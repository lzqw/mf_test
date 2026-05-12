import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def flatten_rollouts(arr, lens):
    return np.concatenate([arr[i,:int(lens[i])] for i in range(arr.shape[0])], axis=0)


def entropy(hist):
    p=hist/(hist.sum()+1e-8); nz=p>0
    return float(-(p[nz]*np.log(p[nz]+1e-8)).sum())


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--runs_roots', nargs='+', default=['runs/exp3','runs/exp4','runs/exp5','runs/exp6','runs/exp7']); ap.add_argument('--grid_x', type=int, default=60); ap.add_argument('--grid_y', type=int, default=40); ap.add_argument('--out_dir', default='paper_outputs/figures'); args=ap.parse_args()
    files=[]
    for r in args.runs_roots: files += list(Path(r).rglob('final_eval/rollouts.npz'))
    rows=[]; residual_all=[]; y_pairs=[]; occ_maps=[]
    for f in files:
        d=np.load(f)
        if 'positions' not in d: continue
        pos=d['positions']; lens=d['valid_lengths'] if 'valid_lengths' in d else np.full((pos.shape[0],), pos.shape[1])
        raw=d['raw_actions'] if 'raw_actions' in d else None
        exe=d['executed_actions'] if 'executed_actions' in d else (d['exec_actions'] if 'exec_actions' in d else None)
        res=d['projection_residual'] if 'projection_residual' in d else None
        pf=f.parts; method=pf[pf.index('exp'+pf[pf.index('exp3')][-1])+1] if any(x.startswith('exp') for x in pf) else f.parent.parent.name
        pflat=flatten_rollouts(pos,lens)
        H,_,_=np.histogram2d(pflat[:,0], pflat[:,1], bins=[args.grid_x,args.grid_y], range=[[-3.5,3.5],[-2,2]])
        occ_maps.append(H)
        cov=float((H>0).sum()/H.size); ent=entropy(H)
        y0=np.nanmean(pos[:,0,1])
        yc=[]
        for i in range(pos.shape[0]):
            tr=pos[i,:int(lens[i])]; m=np.abs(tr[:,0])<0.5
            if np.any(m): yc.append(np.mean(tr[m,1]))
        y_cross=float(np.nanmean(yc)) if yc else np.nan
        if raw is not None: rawf=flatten_rollouts(raw,lens); d_raw=float(np.mean(np.std(rawf,axis=0)))
        else: d_raw=np.nan
        if exe is not None: exef=flatten_rollouts(exe,lens); d_exec=float(np.mean(np.std(exef,axis=0)))
        else: d_exec=np.nan
        rows.append({'method':method,'D_raw':d_raw,'D_exec':d_exec,'R_eff':d_exec/(d_raw+1e-8) if np.isfinite(d_exec) and np.isfinite(d_raw) else np.nan,'coverage':cov,'occ_entropy':ent})
        y_pairs.append((y0,y_cross,method))
        if res is not None: residual_all.append(flatten_rollouts(res[...,None],lens).reshape(-1))
    out=Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    rdf=pd.DataFrame(rows)
    agg=rdf.groupby('method')[['D_raw','D_exec']].mean(numeric_only=True)
    x=np.arange(len(agg)); w=0.35
    fig,ax=plt.subplots(figsize=(10,4)); ax.bar(x-w/2, agg['D_raw'], w, label='raw'); ax.bar(x+w/2, agg['D_exec'], w, label='executed'); ax.set_xticks(x); ax.set_xticklabels(agg.index, rotation=45, ha='right', fontsize=8); ax.legend(); fig.tight_layout(); fig.savefig(out/'exp37_raw_exec_diversity.png',dpi=220); fig.savefig(out/'exp37_raw_exec_diversity.pdf')
    fig,ax=plt.subplots(figsize=(8,4));
    if residual_all: ax.hist(np.concatenate(residual_all), bins=40); ax.set_title('projection residual histogram')
    fig.tight_layout(); fig.savefig(out/'exp37_projection_residual_hist.png',dpi=220); fig.savefig(out/'exp37_projection_residual_hist.pdf')
    fig,ax=plt.subplots(figsize=(6,5));
    for y0,yc,m in y_pairs: ax.scatter(y0,yc,s=18,label=m)
    ax.set_xlabel('y0'); ax.set_ylabel('y_cross'); fig.tight_layout(); fig.savefig(out/'exp37_y0_ycross_scatter.png',dpi=220); fig.savefig(out/'exp37_y0_ycross_scatter.pdf')
    fig,ax=plt.subplots(figsize=(7,4));
    if occ_maps:
        H=np.mean(np.stack(occ_maps),axis=0).T
        im=ax.imshow(H, origin='lower', extent=[-3.5,3.5,-2,2], aspect='auto'); fig.colorbar(im,ax=ax)
    fig.tight_layout(); fig.savefig(out/'exp37_occupancy_heatmap.png',dpi=220); fig.savefig(out/'exp37_occupancy_heatmap.pdf')
    rdf.to_csv(Path('paper_outputs/tables/exp37_diagnostics_metrics.csv'), index=False)

if __name__=='__main__':
    main()

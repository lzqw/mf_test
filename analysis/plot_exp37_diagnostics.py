import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import re


def flatten_rollouts(arr, lens):
    return np.concatenate([arr[i,:int(lens[i])] for i in range(arr.shape[0])], axis=0)


def entropy(hist):
    p=hist/(hist.sum()+1e-8); nz=p>0
    return float(-(p[nz]*np.log(p[nz]+1e-8)).sum())


def parse_path_meta(path):
    parts = list(Path(path).parts)
    exp = None
    exp_idx = None
    for i, part in enumerate(parts):
        m = re.fullmatch(r"exp([3-7])", part)
        if m:
            exp = part
            exp_idx = i
            break
    method = parts[exp_idx + 1] if exp_idx is not None and exp_idx + 1 < len(parts) else Path(path).parent.parent.name
    seed = None
    for part in reversed(parts):
        m = re.search(r"seed[_-]?(\d+)", part)
        if m:
            seed = int(m.group(1))
            break
    if seed is None:
        for part in reversed(parts):
            m = re.fullmatch(r"(\d+)", part)
            if m:
                seed = int(m.group(1))
                break
    return {"exp": exp, "method": method, "seed": seed}


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--runs_roots', nargs='+', default=['runs/exp3','runs/exp4','runs/exp5','runs/exp6','runs/exp7']); ap.add_argument('--grid_x', type=int, default=60); ap.add_argument('--grid_y', type=int, default=40); ap.add_argument('--out_dir', default='paper_outputs/figures'); args=ap.parse_args()
    files=[]
    for r in args.runs_roots: files += list(Path(r).rglob('final_eval/rollouts.npz'))
    rows=[]; residual_by_method={}; y_pairs=[]; occ_by_method={}
    for f in files:
        d=np.load(f)
        if 'positions' not in d: continue
        pos=d['positions']; lens=d['valid_lengths'] if 'valid_lengths' in d else np.full((pos.shape[0],), pos.shape[1])
        raw=d['raw_actions'] if 'raw_actions' in d else None
        exe=d['executed_actions'] if 'executed_actions' in d else (d['exec_actions'] if 'exec_actions' in d else None)
        res=d['projection_residual'] if 'projection_residual' in d else None
        fa=d['filter_active'] if 'filter_active' in d else None
        meta = parse_path_meta(f)
        method=meta['method']
        pflat=flatten_rollouts(pos,lens)
        H,_,_=np.histogram2d(pflat[:,0], pflat[:,1], bins=[args.grid_x,args.grid_y], range=[[-3.5,3.5],[-2,2]])
        occ_by_method.setdefault(method, []).append(H)
        cov=float((H>0).sum()/H.size); ent=entropy(H)
        y0_ep=[]; yc_ep=[]
        for i in range(pos.shape[0]):
            tr=pos[i,:int(lens[i])]; m=np.abs(tr[:,0])<0.5
            y0=float(tr[0,1])
            yc=float(np.mean(tr[m,1])) if np.any(m) else np.nan
            y0_ep.append(y0); yc_ep.append(yc)
            y_pairs.append((y0, yc, method))
        action_lens=np.maximum(lens-1, 0)
        if raw is not None: rawf=flatten_rollouts(raw,action_lens); d_raw=float(np.mean(np.std(rawf,axis=0))) if rawf.size else np.nan
        else: d_raw=np.nan
        if exe is not None: exef=flatten_rollouts(exe,action_lens); d_exec=float(np.mean(np.std(exef,axis=0))) if exef.size else np.nan
        else: d_exec=np.nan
        rows.append({'exp':meta['exp'],'method':method,'seed':meta['seed'],'D_raw':d_raw,'D_exec':d_exec,'R_eff':d_exec/(d_raw+1e-8) if np.isfinite(d_exec) and np.isfinite(d_raw) else np.nan,'coverage':cov,'occ_entropy':ent,
                     'y0_mean':float(np.nanmean(y0_ep)) if y0_ep else np.nan,'y_cross_mean':float(np.nanmean(yc_ep)) if yc_ep else np.nan,
                     'filter_activation_rate':float(np.mean(flatten_rollouts(fa.astype(np.float32), action_lens))) if fa is not None and action_lens.sum()>0 else np.nan})
        if res is not None: residual_by_method.setdefault(method, []).append(flatten_rollouts(res[...,None],action_lens).reshape(-1))
    out=Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    rdf=pd.DataFrame(rows)
    agg=rdf.groupby('method')[['D_raw','D_exec']].mean(numeric_only=True)
    x=np.arange(len(agg)); w=0.35
    fig,ax=plt.subplots(figsize=(10,4)); ax.bar(x-w/2, agg['D_raw'], w, label='raw'); ax.bar(x+w/2, agg['D_exec'], w, label='executed'); ax.set_xticks(x); ax.set_xticklabels(agg.index, rotation=45, ha='right', fontsize=8); ax.legend(); fig.tight_layout(); fig.savefig(out/'exp37_raw_exec_diversity.png',dpi=220); fig.savefig(out/'exp37_raw_exec_diversity.pdf')
    fig,ax=plt.subplots(figsize=(8,4));
    for method, arrs in residual_by_method.items():
        ax.hist(np.concatenate(arrs), bins=40, alpha=0.4, label=method)
    if residual_by_method: ax.legend(fontsize=8); ax.set_title('projection residual histogram')
    fig.tight_layout(); fig.savefig(out/'exp37_projection_residual_hist.png',dpi=220); fig.savefig(out/'exp37_projection_residual_hist.pdf')
    fig,ax=plt.subplots(figsize=(6,5));
    methods = sorted({m for _,_,m in y_pairs})
    for m in methods:
        pts=[(y0,yc) for y0,yc,mm in y_pairs if mm==m]
        ax.scatter([p[0] for p in pts],[p[1] for p in pts],s=14,label=m,alpha=0.6)
    ax.set_xlabel('y0'); ax.set_ylabel('y_cross'); fig.tight_layout(); fig.savefig(out/'exp37_y0_ycross_scatter.png',dpi=220); fig.savefig(out/'exp37_y0_ycross_scatter.pdf')
    fig,ax=plt.subplots(figsize=(7,4));
    if occ_by_method:
        H=np.mean(np.stack([h for v in occ_by_method.values() for h in v]),axis=0).T
        im=ax.imshow(H, origin='lower', extent=[-3.5,3.5,-2,2], aspect='auto'); fig.colorbar(im,ax=ax)
    fig.tight_layout(); fig.savefig(out/'exp37_occupancy_heatmap.png',dpi=220); fig.savefig(out/'exp37_occupancy_heatmap.pdf')
    for method, hmaps in occ_by_method.items():
        fig, ax = plt.subplots(figsize=(7,4))
        Hm=np.mean(np.stack(hmaps),axis=0).T
        im=ax.imshow(Hm, origin='lower', extent=[-3.5,3.5,-2,2], aspect='auto')
        ax.set_title(f'occupancy heatmap: {method}')
        fig.colorbar(im,ax=ax)
        fig.tight_layout(); fig.savefig(out/f'exp37_occupancy_heatmap_{method}.png',dpi=220); fig.savefig(out/f'exp37_occupancy_heatmap_{method}.pdf')
    table_dir=Path('paper_outputs/tables'); table_dir.mkdir(parents=True, exist_ok=True)
    rdf.to_csv(table_dir/'exp37_diagnostics_metrics.csv', index=False)

if __name__=='__main__':
    main()

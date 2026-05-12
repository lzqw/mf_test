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

def parse_path_meta(path):
    parts = list(Path(path).parts)
    method = Path(path).parent.parent.name
    seed = None
    for part in reversed(parts):
        m = re.search(r"seed[_-]?(\d+)", part)
        if m: seed = int(m.group(1)); break
    return {"method": method, "seed": seed}

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--runs_roots', nargs='+', default=['runs/exp3','runs/exp4','runs/exp5','runs/exp6','runs/exp7']); ap.add_argument('--out_dir', default='paper_outputs/figures'); args=ap.parse_args()
    files=[]
    for r in args.runs_roots: files += list(Path(r).rglob('final_eval/rollouts.npz'))
    rows=[]; residual_by_method={}; y_pairs=[]
    for f in files:
        d=np.load(f)
        if 'positions' not in d: continue
        pos=d['positions']; lens=d['valid_lengths'] if 'valid_lengths' in d else np.full((pos.shape[0],), pos.shape[1])
        raw=d['raw_actions'] if 'raw_actions' in d else None
        exe=d['executed_actions'] if 'executed_actions' in d else (d['exec_actions'] if 'exec_actions' in d else None)
        res=d['projection_residual'] if 'projection_residual' in d else None
        meta = parse_path_meta(f); method=meta['method']
        for i in range(pos.shape[0]):
            tr=pos[i,:int(lens[i])]; m=np.abs(tr[:,0])<0.5
            y_pairs.append((float(tr[0,1]), float(np.mean(tr[m,1])) if np.any(m) else np.nan, method))
        action_lens=np.maximum(lens-1, 0)
        d_raw=d_exec=np.nan
        if raw is not None and action_lens.sum()>0:
            rawf=flatten_rollouts(raw,action_lens); d_raw=float(np.mean(np.std(rawf,axis=0))) if rawf.size else np.nan
        if exe is not None and action_lens.sum()>0:
            exef=flatten_rollouts(exe,action_lens); d_exec=float(np.mean(np.std(exef,axis=0))) if exef.size else np.nan
        rows.append({'method':method,'seed':meta['seed'],'D_raw':d_raw,'D_exec':d_exec})
        if res is not None and action_lens.sum()>0: residual_by_method.setdefault(method, []).append(flatten_rollouts(res[...,None],action_lens).reshape(-1))

    out=Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    rdf=pd.DataFrame(rows)
    agg=rdf.groupby('method')[['D_raw','D_exec']].mean(numeric_only=True)
    x=np.arange(len(agg)); w=0.35
    fig,ax=plt.subplots(figsize=(10,4)); ax.bar(x-w/2, agg['D_raw'], w, label='raw'); ax.bar(x+w/2, agg['D_exec'], w, label='executed'); ax.set_xticks(x); ax.set_xticklabels(agg.index, rotation=45, ha='right', fontsize=8); ax.legend(); fig.tight_layout(); fig.savefig(out/'exp37_rollout_level_raw_exec_spread.png',dpi=220); fig.savefig(out/'exp37_rollout_level_raw_exec_spread.pdf')

    if residual_by_method:
        stat_rows=[]
        for m, arrs in residual_by_method.items():
            v=np.concatenate(arrs)
            stat_rows.append((m, float(np.mean(v)), float(np.percentile(v,95)), float(np.percentile(v,99))))
        sdf=pd.DataFrame(stat_rows, columns=['method','mean','p95','p99']).sort_values('method')
        xx=np.arange(len(sdf)); ww=0.25
        fig,ax=plt.subplots(figsize=(10,4)); ax.bar(xx-ww, sdf['mean'], ww, label='mean'); ax.bar(xx, sdf['p95'], ww, label='95%'); ax.bar(xx+ww, sdf['p99'], ww, label='99%')
        ax.set_xticks(xx); ax.set_xticklabels(sdf['method'], rotation=45, ha='right', fontsize=8); ax.set_title('projection residual summary'); ax.legend(); fig.tight_layout(); fig.savefig(out/'exp37_projection_residual_stats.png',dpi=220); fig.savefig(out/'exp37_projection_residual_stats.pdf')

    methods=sorted({m for _,_,m in y_pairs})
    cols=3; rows_n=int(np.ceil(len(methods)/cols)) if methods else 1
    fig,axs=plt.subplots(rows_n, cols, figsize=(4*cols,3.5*rows_n)); axs=np.array(axs).reshape(-1)
    for ax in axs: ax.axis('off')
    for i,m in enumerate(methods):
        ax=axs[i]; ax.axis('on')
        pts=[(y0,yc) for y0,yc,mm in y_pairs if mm==m and np.isfinite(yc)]
        if pts:
            ax.scatter([p[0] for p in pts],[p[1] for p in pts],s=12,alpha=0.6)
        ax.set_title(m, fontsize=9); ax.set_xlabel('y0'); ax.set_ylabel('y_cross')
    fig.tight_layout(); fig.savefig(out/'exp37_y0_ycross_scatter_facets.png',dpi=220); fig.savefig(out/'exp37_y0_ycross_scatter_facets.pdf')

if __name__=='__main__':
    main()

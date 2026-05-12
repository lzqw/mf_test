import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

METHOD_MAP = {
    "likelihood_tn_alpha005_temp007": "Full-LTN",
    "main_likelihood_tn_alpha005_temp007": "Full-LTN",
    "main_likelihood_tn_alpha005_temp007_mix010": "Full-LTN (mix=0.10)",
    "flac_tn_alpha002_temp012_mix010": "FLAC-TN",
    "ablation_no_tn_entropy": "w/o TN entropy",
    "ablation_no_weight_mix": "w/o weight mix",
    "ablation_flac_normal_tangent": "normal+tangent energy",
}
METHOD_ORDER = ["Full-LTN","Full-LTN (mix=0.10)","FLAC-TN","w/o TN entropy","w/o weight mix","normal+tangent energy"]
TYPICAL_ROLLOUTS = {
    "Full-LTN": ["runs/exp7/main_likelihood_tn_alpha005_temp007_mix010/seed4/final_eval/rollouts.npz"],
    "w/o TN entropy": ["runs/exp7/ablation_no_tn_entropy/seed1/final_eval/rollouts.npz"],
    "w/o weight mix": ["runs/exp7/ablation_no_weight_mix/seed2/final_eval/rollouts.npz", "runs/exp7/ablation_no_weight_mix/seed1/final_eval/rollouts.npz"],
    "normal+tangent energy": ["runs/exp7/ablation_flac_normal_tangent/seed1/final_eval/rollouts.npz"],
}

def save(fig, p):
    p.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(p.with_suffix('.png'), dpi=220)
    fig.savefig(p.with_suffix('.pdf'))

def scene(ax):
    ax.add_patch(plt.Circle((0, 0), 0.8, fill=False, color='k', lw=1.2))
    ax.scatter([-2.6], [0], marker='*', c='r', s=80)
    ax.scatter([2.6], [0], marker='o', c='g', s=40)
    ax.set_xlim(-3.5, 3.5); ax.set_ylim(-2.0, 2.0); ax.set_aspect('equal')

def classify(traj):
    y=traj[:,1]
    if np.nanmean(y) > 0.15:
        return 'upper','#d62728'
    if np.nanmean(y) < -0.15:
        return 'lower','#1f77b4'
    return 'failure','#7f7f7f'

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--summary_csv', default='paper_outputs/tables/exp37_all_eval_summary.csv')
    ap.add_argument('--out_dir', default='paper_outputs/figures')
    args = ap.parse_args()

    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(args.summary_csv)
    dff = df[df.eval_type == 'final_eval'].copy()
    dff['method_short'] = dff['method'].map(METHOD_MAP).fillna(dff['method'])
    dff = dff[dff['method_short'].isin(METHOD_ORDER)].copy()

    metrics = ['success_rate','episode_return_mean','FAR','APR']
    fig, axs = plt.subplots(2,2, figsize=(11,7)); axs=axs.reshape(-1)
    for ax, m in zip(axs, metrics):
        agg=dff.groupby('method_short')[m].agg(['mean','std']).reindex(METHOD_ORDER).dropna(subset=['mean'])
        x=np.arange(len(agg))
        yerr=(agg['std']/np.sqrt(np.maximum(dff.groupby('method_short')[m].count().reindex(agg.index),1))).to_numpy()
        ax.bar(x, agg['mean'].to_numpy(), yerr=yerr, capsize=3)
        ax.set_title(m); ax.set_xticks(x); ax.set_xticklabels(agg.index, rotation=28, ha='right', fontsize=9)
    save(fig, out/'exp37_metrics_bar')

    fig, ax = plt.subplots(figsize=(10,4))
    route = dff[['method_short','seed','route_upper_ratio','route_lower_ratio']].dropna()
    route=route.sort_values(['method_short','seed'], key=lambda s: s.map({m:i for i,m in enumerate(METHOD_ORDER)}) if s.name=='method_short' else s)
    labels = [f"{r.method_short}|s{int(r.seed)}" for r in route.itertuples()]
    x = np.arange(len(route)); up=route['route_upper_ratio'].to_numpy(float); lo=route['route_lower_ratio'].to_numpy(float)
    ax.bar(x, lo, label='lower'); ax.bar(x, up, bottom=lo, label='upper')
    ax.set_xticks(x); ax.set_xticklabels(labels, rotation=70, fontsize=7); ax.legend(); ax.set_title('route stacked ratio (selected)')
    save(fig, out/'exp37_route_stacked')

    fig, ax = plt.subplots(figsize=(9,4))
    ent = dff.groupby('method_short')['effective_route_entropy'].agg(['mean','std','count']).reindex(METHOD_ORDER).dropna(subset=['mean'])
    x=np.arange(len(ent)); ax.bar(x, ent['mean'].to_numpy(), yerr=(ent['std']/np.sqrt(np.maximum(ent['count'],1))).to_numpy(), capsize=3)
    ax.axhline(np.log(2.0), ls='--', c='k', lw=1, label='log(2)')
    ax.set_xticks(x); ax.set_xticklabels(ent.index, rotation=28, ha='right', fontsize=9); ax.set_title('effective route entropy'); ax.legend()
    save(fig, out/'exp37_effective_entropy')

    methods=list(TYPICAL_ROLLOUTS.keys()); cols=2; rows=int(np.ceil(len(methods)/cols))
    fig, axs = plt.subplots(rows, cols, figsize=(5*cols,4*rows)); axs=np.array(axs).reshape(-1)
    for ax in axs: ax.axis('off')
    for i,m in enumerate(methods):
        ax=axs[i]; ax.axis('on'); rp=None
        for cand in TYPICAL_ROLLOUTS[m]:
            if Path(cand).exists(): rp=Path(cand); break
        if rp is None:
            ax.set_title(m+' (no rollout)'); continue
        data=np.load(rp); pos=data['positions']; lens=data['valid_lengths'] if 'valid_lengths' in data else np.full((pos.shape[0],), pos.shape[1])
        for j in range(min(25,pos.shape[0])):
            t=int(lens[j]); tr=pos[j,:t,:2]
            lbl,c=classify(tr)
            ax.plot(tr[:,0], tr[:,1], alpha=0.55, lw=1.1, c=c)
        scene(ax); ax.set_title(m, fontsize=10)
    save(fig, out/'exp37_trajectory_grid')

if __name__=='__main__':
    main()

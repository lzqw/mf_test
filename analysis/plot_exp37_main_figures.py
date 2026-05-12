import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--summary_csv', default='paper_outputs/tables/exp37_all_eval_summary.csv')
    ap.add_argument('--runs_roots', nargs='+', default=['runs/exp3','runs/exp4','runs/exp5','runs/exp6','runs/exp7'])
    ap.add_argument('--out_dir', default='paper_outputs/figures')
    args = ap.parse_args()

    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(args.summary_csv)
    dff = df[df.eval_type == 'final_eval'].copy()

    metrics = ['success_rate','episode_return_mean','time_to_goal_mean','FAR','APR']
    agg = dff.groupby('method')[metrics].mean(numeric_only=True)
    fig, axs = plt.subplots(2,2, figsize=(11,7)); axs=axs.reshape(-1)
    for ax, m in zip(axs, ['success_rate','episode_return_mean','FAR','APR']):
        ax.bar(np.arange(len(agg.index)), agg[m].to_numpy())
        ax.set_title(m); ax.set_xticks(np.arange(len(agg.index))); ax.set_xticklabels(agg.index, rotation=45, ha='right', fontsize=8)
    save(fig, out/'exp37_metrics_bar')

    fig, ax = plt.subplots(figsize=(10,4))
    route = dff[['method','seed','route_upper_ratio','route_lower_ratio']].dropna()
    labels = [f"{r.method}|{int(r.seed) if pd.notna(r.seed) else 'na'}" for r in route.itertuples()]
    x = np.arange(len(route))
    up = route['route_upper_ratio'].to_numpy(float); lo=route['route_lower_ratio'].to_numpy(float)
    ax.bar(x, lo, label='lower'); ax.bar(x, up, bottom=lo, label='upper'); ax.set_xticks(x); ax.set_xticklabels(labels, rotation=75, fontsize=6)
    ax.legend(); ax.set_title('route stacked ratio')
    save(fig, out/'exp37_route_stacked')

    fig, ax = plt.subplots(figsize=(9,4))
    ent = dff.groupby('method')['effective_route_entropy'].mean(numeric_only=True)
    x=np.arange(len(ent)); ax.bar(x, ent.to_numpy()); ax.axhline(np.log(2.0), ls='--', c='k'); ax.set_xticks(x); ax.set_xticklabels(ent.index, rotation=45, ha='right', fontsize=8)
    ax.set_title('effective route entropy')
    save(fig, out/'exp37_effective_entropy')

    methods = list(dff['method'].dropna().unique())[:9]
    cols=3; rows=int(np.ceil(len(methods)/cols))
    fig, axs = plt.subplots(rows, cols, figsize=(4*cols,3.5*rows)); axs=np.array(axs).reshape(-1)
    for ax in axs: ax.axis('off')
    for i,m in enumerate(methods):
        ax=axs[i]; ax.axis('on')
        subset=dff[dff['method']==m]
        if subset.empty: continue
        rp = Path(subset.iloc[0]['summary_path']).parent/'rollouts.npz'
        if not rp.exists():
            ax.set_title(m+' (no rollout)'); continue
        data=np.load(rp)
        pos=data['positions']; lens=data['valid_lengths'] if 'valid_lengths' in data else np.full((pos.shape[0],), pos.shape[1])
        for j in range(min(20,pos.shape[0])):
            t=int(lens[j]); ax.plot(pos[j,:t,0], pos[j,:t,1], alpha=0.6, lw=1)
        scene(ax); ax.set_title(m, fontsize=9)
    save(fig, out/'exp37_trajectory_grid')

if __name__=='__main__':
    main()

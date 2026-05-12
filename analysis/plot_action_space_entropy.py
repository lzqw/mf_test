import argparse
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from relax.safety.obstacle_navigation_filter import ObstacleNavConfig, make_action_grid, project_action_np, is_action_feasible_np


def entropy2d(samples, bins=40):
    H,_,_=np.histogram2d(samples[:,0], samples[:,1], bins=bins, range=[[-1,1],[-1,1]])
    p=H/(H.sum()+1e-8); nz=p>0
    return float(-(p[nz]*np.log(p[nz]+1e-8)).sum())


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--checkpoint', default='')
    ap.add_argument('--state', nargs=2, type=float, action='append', required=True)
    ap.add_argument('--num_samples', type=int, default=800)
    ap.add_argument('--out_dir', default='paper_outputs/figures')
    ap.add_argument('--heatmap', action='store_true')
    args=ap.parse_args()

    cfg=ObstacleNavConfig(); grid=make_action_grid(81)
    out=Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    rng=np.random.default_rng(0)
    for si,st in enumerate(args.state):
        pos=np.asarray(st,dtype=np.float32)
        raw=np.clip(rng.normal(size=(args.num_samples,2)).astype(np.float32),-1,1)
        exe=[]; residual=[]
        for a in raw:
            ea,_,gap,_,_=project_action_np(pos,a,grid,cfg); exe.append(ea); residual.append(gap)
        exe=np.asarray(exe); residual=np.asarray(residual)
        feas=np.asarray([is_action_feasible_np(pos,a,cfg) for a in grid])
        D_raw=entropy2d(raw); D_exec=entropy2d(exe); R_eff=D_exec/(D_raw+1e-8)
        dirn=pos-cfg.obstacle_center; dirn=dirn/(np.linalg.norm(dirn)+1e-8); tangent=np.array([-dirn[1],dirn[0]])
        proj=exe-raw; D_N=float(np.std(proj@dirn)); D_T=float(np.std(proj@tangent))
        print(f"state={pos.tolist()} D_raw={D_raw:.6f} D_exec={D_exec:.6f} R_eff={R_eff:.6f} D_N={D_N:.6f} D_T={D_T:.6f} D_T/(D_N+eps)={D_T/(D_N+1e-8):.6f}")
        fig,ax=plt.subplots(figsize=(6,6))
        ax.scatter(grid[feas,0], grid[feas,1], s=6, c='lightgray', label='feasible set')
        ax.scatter(raw[:,0], raw[:,1], s=8, alpha=0.35, label='raw')
        ax.scatter(exe[:,0], exe[:,1], s=8, alpha=0.35, label='exec')
        idx=np.linspace(0,args.num_samples-1,min(120,args.num_samples),dtype=int)
        for i in idx: ax.plot([raw[i,0],exe[i,0]],[raw[i,1],exe[i,1]], c='k', alpha=0.15, lw=0.5)
        ax.arrow(0,0,dirn[0]*0.7,dirn[1]*0.7,color='r',width=0.01); ax.arrow(0,0,tangent[0]*0.7,tangent[1]*0.7,color='b',width=0.01)
        ax.set_xlim(-1,1); ax.set_ylim(-1,1); ax.set_title(f"state=({pos[0]:.2f},{pos[1]:.2f})")
        ax.legend(fontsize=8, loc='upper left')
        fig.tight_layout(); fig.savefig(out/f'action_space_entropy_state{si}.png',dpi=220); fig.savefig(out/f'action_space_entropy_state{si}.pdf')

if __name__=='__main__':
    main()

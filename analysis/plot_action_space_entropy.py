import argparse
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import jax

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from envs.safe_obstacle_navigation_2d import SafeObstacleNavigation2DEnv
from eval.eval_safe_obstacle_navigation import load_agent
from relax.safety.obstacle_navigation_filter import ObstacleNavConfig, make_action_grid, project_action_np, is_action_feasible_np

def entropy2d(samples, bins=40):
    H,_,_=np.histogram2d(samples[:,0], samples[:,1], bins=bins, range=[[-1,1],[-1,1]])
    p=H/(H.sum()+1e-8); nz=p>0
    return float(-(p[nz]*np.log(p[nz]+1e-8)).sum())

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--checkpoint', default='')
    ap.add_argument('--algo', default='safe_pullback_rf2')
    ap.add_argument('--state', nargs=2, type=float, action='append', required=True)
    ap.add_argument('--num_samples', type=int, default=800)
    ap.add_argument('--out_dir', default='paper_outputs/figures')
    ap.add_argument('--random_policy', action='store_true')
    ap.add_argument('--method_name', default='unknown_method')
    args=ap.parse_args()

    cfg=ObstacleNavConfig(); grid=make_action_grid(81)
    out=Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    rng=np.random.default_rng(0)

    env = SafeObstacleNavigation2DEnv()
    agent = None
    if not args.random_policy:
        if not args.checkpoint:
            raise ValueError('--checkpoint is required unless --random_policy is set')
        agent = load_agent(args.checkpoint, args.algo)
    rows=[]
    for si,st in enumerate(args.state):
        pos=np.asarray(st,dtype=np.float32)
        obs = env._get_obs_from_state(pos)[None, :]
        if args.random_policy:
            raw=np.clip(rng.normal(size=(args.num_samples,2)).astype(np.float32),-1,1)
        else:
            keys = jax.random.split(jax.random.PRNGKey(2026 + si), args.num_samples)
            raw = np.asarray([np.asarray(agent.get_action(keys[i], obs)[0], dtype=np.float32) for i in range(args.num_samples)], dtype=np.float32)
            raw = np.clip(raw, -1.0, 1.0)

        exe=[]; residual=[]
        for a in raw:
            ea,_,gap,_,_=project_action_np(pos,a,grid,cfg); exe.append(ea); residual.append(gap)
        exe=np.asarray(exe); residual=np.asarray(residual)
        feas=np.asarray([is_action_feasible_np(pos,a,cfg) for a in grid])
        D_raw=entropy2d(raw); D_exec=entropy2d(exe); R_eff=D_exec/(D_raw+1e-8)

        dirn=pos-cfg.obstacle_center; dirn=dirn/(np.linalg.norm(dirn)+1e-8); tangent=np.array([-dirn[1],dirn[0]])
        raw_n, raw_t = raw@dirn, raw@tangent
        exe_n, exe_t = exe@dirn, exe@tangent
        proj = exe - raw
        proj_n, proj_t = proj@dirn, proj@tangent

        raw_D_N, raw_D_T = float(np.std(raw_n)), float(np.std(raw_t))
        exec_D_N, exec_D_T = float(np.std(exe_n)), float(np.std(exe_t))
        proj_D_N, proj_D_T = float(np.std(proj_n)), float(np.std(proj_t))
        residual_mean=float(np.mean(residual))
        rows.append({
            'method':args.method_name,'checkpoint':args.checkpoint,'state_x':float(pos[0]),'state_y':float(pos[1]),
            'D_raw':D_raw,'D_exec':D_exec,'R_eff':R_eff,
            'raw_D_N':raw_D_N,'raw_D_T':raw_D_T,'exec_D_N':exec_D_N,'exec_D_T':exec_D_T,
            'proj_D_N':proj_D_N,'proj_D_T':proj_D_T,'residual_mean':residual_mean,
        })

        fig,ax=plt.subplots(figsize=(6,6))
        ax.scatter(grid[feas,0], grid[feas,1], s=6, c='lightgray', label='feasible set')
        ax.scatter(raw[:,0], raw[:,1], s=8, alpha=0.35, label='raw')
        ax.scatter(exe[:,0], exe[:,1], s=8, alpha=0.35, label='exec')
        idx=np.linspace(0,args.num_samples-1,min(120,args.num_samples),dtype=int)
        for i in idx:
            ax.plot([raw[i,0],exe[i,0]],[raw[i,1],exe[i,1]], c='k', alpha=0.15, lw=0.5)
        ax.arrow(0,0,dirn[0]*0.7,dirn[1]*0.7,color='r',width=0.01)
        ax.arrow(0,0,tangent[0]*0.7,tangent[1]*0.7,color='b',width=0.01)
        ax.set_xlim(-1,1); ax.set_ylim(-1,1)
        ax.set_title(f"{args.method_name} state=({pos[0]:.2f},{pos[1]:.2f})")
        ax.legend(fontsize=8, loc='upper left')
        fig.tight_layout(); fig.savefig(out/f'action_space_entropy_state{si}.png',dpi=220); fig.savefig(out/f'action_space_entropy_state{si}.pdf')
    pd.DataFrame(rows).to_csv(out/'action_space_entropy_state_stats.csv', index=False)

if __name__=='__main__':
    main()

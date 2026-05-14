import argparse, csv, json, pickle, sys, time
from pathlib import Path
from typing import NamedTuple
import jax, jax.numpy as jnp, numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
import safety_gymnasium
try:
    from tensorboardX import SummaryWriter
except ImportError:
    from torch.utils.tensorboard import SummaryWriter

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from envs.safety_gym_safe_wrapper import SafeSafetyGymWrapper
from relax.algorithm.safe_pullback_rf2_sac_ent_safetygym import SafePullbackRF2SACENTSafetyGym
from relax.network.safe_pullback_rf2_sac_ent import create_safe_pullback_rf2_sac_ent_net
from scripts.safe_pullback_experience import SafePullbackExperience

class Batch(NamedTuple):
    obs:jnp.ndarray; raw_action:jnp.ndarray; action:jnp.ndarray; reward:jnp.ndarray; done:jnp.ndarray; next_obs:jnp.ndarray; projection_residual:jnp.ndarray; projection_cost:jnp.ndarray

def sample_batch(buf,b):
    idx=np.random.randint(0,len(buf),size=b); it=[buf[i] for i in idx]
    return Batch(*(jnp.asarray(np.stack([getattr(x,f) for x in it])) for f in Batch._fields))

def make_algo(args, obs_dim, act_dim):
    key=jax.random.PRNGKey(args.seed)
    net, params = create_safe_pullback_rf2_sac_ent_net(key, obs_dim, act_dim, hidden_sizes=[256]*3, diffusion_hidden_sizes=[256]*3, num_timesteps=args.diffusion_steps, num_ent_timesteps=args.num_ent_timesteps, alpha_value=args.alpha_value, fixed_alpha=args.fixed_alpha, init_alpha=args.init_alpha, noise_scale=args.policy_noise_scale, use_directional_noise=args.use_directional_noise)
    return SafePullbackRF2SACENTSafetyGym(net, params, gamma=args.gamma, gamma_p=args.gamma_p, lr=args.lr, alpha_lr=args.alpha_lr, sample_k=args.sample_k, lambda_p=args.lambda_p, use_projection_critic=args.use_projection_critic, fixed_alpha=args.fixed_alpha, alpha_value=args.alpha_value, lambda_p_warmup_steps=args.lambda_p_warmup_steps, use_tn_energy=args.use_tn_energy, entropy_reg_mode=args.entropy_reg_mode, use_filter_surrogate=args.use_filter_surrogate, surrogate_warmup_steps=args.surrogate_warmup_steps, surrogate_loss_coef=args.surrogate_loss_coef, lambda_raw_norm=args.lambda_raw_norm)

def safe_nanmean(xs):
    arr=np.asarray(xs,dtype=np.float32)
    finite=arr[np.isfinite(arr)]
    return float(np.mean(finite)) if finite.size else np.nan

def safe_info_float(info, key, default=0.0):
    v = info.get(key, default)
    if isinstance(v, (str, bytes)):
        return float(default)
    try:
        return float(v)
    except (TypeError, ValueError):
        return float(default)



def rolling_mean(x, window):
    arr = np.asarray(x, dtype=np.float64)
    if arr.size == 0:
        return arr
    if window is None or window <= 1:
        return arr
    out = np.full(arr.shape, np.nan, dtype=np.float64)
    for i in range(arr.size):
        st = max(0, i - window + 1)
        seg = arr[st:i + 1]
        seg = seg[np.isfinite(seg)]
        if seg.size:
            out[i] = float(np.mean(seg))
    return out


def _extract_series(rows, key):
    xs, ys = [], []
    for i, row in enumerate(rows):
        x = row.get("step", i)
        y = row.get(key, np.nan)
        try:
            xv = float(x)
            yv = float(y)
        except (TypeError, ValueError):
            continue
        if np.isfinite(xv) and np.isfinite(yv):
            xs.append(xv)
            ys.append(yv)
    return np.asarray(xs, dtype=np.float64), np.asarray(ys, dtype=np.float64)


def _save_csv(path, rows):
    if not rows:
        path.write_text("")
        return
    keys = sorted({k for row in rows for k in row.keys()})
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for row in rows:
            w.writerow(row)


def save_curves(log_dir, step_metrics, eval_metrics, train_metrics=None, window=100, curve_dir=None):
    base_dir = Path(curve_dir) if curve_dir else Path(log_dir) / "curves"
    base_dir.mkdir(parents=True, exist_ok=True)

    def plot_rows(rows, keys, filename, use_rolling=False):
        fig = plt.figure(figsize=(10, 6))
        ax = fig.add_subplot(111)
        has_line = False
        for key in keys:
            xs, ys = _extract_series(rows, key)
            if ys.size == 0:
                continue
            if use_rolling:
                ys = rolling_mean(ys, window)
                finite = np.isfinite(ys)
                xs, ys = xs[finite], ys[finite]
            if ys.size == 0:
                continue
            ax.plot(xs, ys, label=key)
            has_line = True
        ax.set_xlabel("step")
        ax.set_ylabel("value")
        ax.grid(True, alpha=0.3)
        if has_line:
            ax.legend()
        fig.tight_layout()
        fig.savefig(base_dir / filename, dpi=150)
        plt.close(fig)

    plot_rows(step_metrics, ["cost", "filter_active", "projection_residual", "raw_action_norm", "exec_action_norm"], "train_safety_curves.png", use_rolling=True)
    plot_rows(step_metrics, ["safe_candidate_ratio", "emergency_active", "current_min_h", "predicted_min_h", "filter_active_005", "filter_active_010"], "shield_curves.png", use_rolling=True)
    plot_rows(eval_metrics, ["return_", "cost_return", "success_rate", "FAR", "APR", "safe_candidate_ratio", "emergency_rate", "safety_score"], "eval_curves.png", use_rolling=False)
    if train_metrics:
        plot_rows(train_metrics, ["q1_loss", "q2_loss", "qp_loss", "vp_loss", "policy_loss", "q_reward_mean", "q_projection_mean", "lambda_eff", "safe_energy_actor_mean"], "critic_actor_curves.png", use_rolling=False)


def save_partial_outputs(log_dir, train_metrics, eval_metrics, step_metrics, agent_state):
    pickle.dump(train_metrics, open(log_dir / "train_metrics.pkl", "wb"))
    pickle.dump(eval_metrics, open(log_dir / "eval_metrics.pkl", "wb"))
    pickle.dump(step_metrics, open(log_dir / "step_metrics.pkl", "wb"))
    pickle.dump(agent_state, open(log_dir / "checkpoint.pkl", "wb"))
    _save_csv(log_dir / "step_metrics.csv", step_metrics)
    _save_csv(log_dir / "eval_metrics.csv", eval_metrics)
    _save_csv(log_dir / "train_metrics.csv", train_metrics)

def eval_agent(agent, args, env):
    ms=[]
    for ep in range(args.eval_episodes):
        obs,_=env.reset(seed=args.seed+1000+ep); done=False; ret=0; l=0; costs=[]; fars=[]; aprs=[]; rns=[]; ens=[]; mins=[]; cbfs=[]; sv=[]; cv=[]; sc_ratio=[]; emergency=[]; gh=[]; fh=[]; lh=[]; rh=[]; info={}
        while not done and l<1000:
            a=np.asarray(agent.get_action(jax.random.PRNGKey(args.seed+ep*10000+l),obs[None])[0]); obs,r,t,tr,info=env.step(a); done=t or tr; ret+=float(r); l+=1
            costs.append(safe_info_float(info, 'cost', 0)); fars.append(safe_info_float(info, 'filter_active', 0)); aprs.append(safe_info_float(info, 'projection_residual', 0)); rns.append(safe_info_float(info, 'raw_action_norm', 0)); ens.append(safe_info_float(info, 'exec_action_norm', 0)); mins.append(safe_info_float(info, 'min_h', np.nan)); cbfs.append(safe_info_float(info, 'cbf_violation', 0)); sv.append(safe_info_float(info, 'safety_violation', 0)); cv.append(safe_info_float(info, 'constraint_violation', 0)); sc_ratio.append(safe_info_float(info, 'safe_candidate_ratio', 0.0)); emergency.append(safe_info_float(info, 'emergency_active', 0.0)); gh.append(safe_info_float(info, 'global_min_h', np.nan)); fh.append(safe_info_float(info, 'front_h', np.nan)); lh.append(safe_info_float(info, 'left_h', np.nan)); rh.append(safe_info_float(info, 'right_h', np.nan))
        success=safe_info_float(info, 'is_success', safe_info_float(info, 'success', safe_info_float(info, 'goal_met', safe_info_float(info, 'task_success', 0.0))))
        ms.append(dict(return_=ret, episode_length=l, success_rate=success, cost_return=float(np.sum(costs)), cost_rate=float(np.mean(np.array(costs)>0)) if costs else 0.0, safety_violation_rate=float(np.mean(sv)) if sv else 0.0, constraint_violation_rate=float(np.mean(cv)) if cv else 0.0, FAR=float(np.mean(fars)) if fars else 0.0, APR=float(np.mean(aprs)) if aprs else 0.0, projection_cost=float(np.mean(np.square(aprs))) if aprs else 0.0, raw_action_norm=float(np.mean(rns)) if rns else 0.0, exec_action_norm=float(np.mean(ens)) if ens else 0.0, min_h=safe_nanmean(mins) if mins else np.nan, cbf_violation_rate=float(np.mean(cbfs)) if cbfs else 0.0, safe_candidate_ratio=float(np.mean(sc_ratio)) if sc_ratio else 0.0, emergency_rate=float(np.mean(emergency)) if emergency else 0.0, global_min_h=safe_nanmean(gh) if gh else np.nan, front_h=safe_nanmean(fh) if fh else np.nan, left_h=safe_nanmean(lh) if lh else np.nan, right_h=safe_nanmean(rh) if rh else np.nan))
    out={k:safe_nanmean([m[k] for m in ms]) for k in ms[0].keys()}
    out['safety_score']=out['cost_return']+10.0*out['safety_violation_rate']+out['FAR']+0.1*out['APR']
    return out

if __name__=='__main__':
    p=argparse.ArgumentParser();
    for k,d,t in [('env_id','SafetyPointGoal1-v0',str),('seed',0,int),('total_steps',50000,int),('start_steps',1000,int),('update_after',1000,int),('batch_size',256,int),('eval_interval',1000,int),('eval_episodes',3,int),('log_dir',None,str)]: p.add_argument(f'--{k}',default=d,type=t,required=(k=='log_dir'))
    p.add_argument('--use_filter',action='store_true'); p.add_argument('--filter_type',default='hybrid'); p.add_argument('--terminate_on_safety_violation',action='store_true'); p.add_argument('--cost_limit_per_step',type=float,default=0.0)
    p.add_argument('--save_curve_interval',type=int,default=0)
    p.add_argument('--curve_dir',type=str,default=None)
    p.add_argument('--save_partial_on_eval',action='store_true',default=True)
    p.add_argument('--plot_train_window',type=int,default=100)
    p.add_argument('--use_tn_energy',action='store_true'); p.add_argument('--use_projection_critic',action='store_true'); p.add_argument('--lambda_p',type=float,default=0.03); p.add_argument('--lambda_p_warmup_steps',type=int,default=0); p.add_argument('--entropy_reg_mode',default='flac_tn'); p.add_argument('--sample_k',type=int,default=256); p.add_argument('--diffusion_steps',type=int,default=10); p.add_argument('--num_ent_timesteps',type=int,default=10); p.add_argument('--policy_noise_scale',type=float,default=0.3); p.add_argument('--use_directional_noise',action='store_true'); p.add_argument('--fixed_alpha',action='store_true'); p.add_argument('--alpha_value',type=float,default=0.1); p.add_argument('--init_alpha',type=float,default=0.1); p.add_argument('--lr',type=float,default=3e-4); p.add_argument('--alpha_lr',type=float,default=1e-2); p.add_argument('--gamma',type=float,default=0.99); p.add_argument('--gamma_p',type=float,default=0.99); p.add_argument('--use_filter_surrogate',action='store_true'); p.add_argument('--surrogate_warmup_steps',type=int,default=0); p.add_argument('--surrogate_loss_coef',type=float,default=1.0); p.add_argument('--lambda_raw_norm',type=float,default=0.0)
    args=p.parse_args(); np.random.seed(args.seed)
    if args.save_curve_interval < 0:
        args.save_curve_interval = 0
    log=Path(args.log_dir); (log/'checkpoints').mkdir(parents=True,exist_ok=True); writer=SummaryWriter(str(log/'tb')); (log/'args.json').write_text(json.dumps(vars(args),indent=2,sort_keys=True))
    env=SafeSafetyGymWrapper(env_id=args.env_id,use_filter=args.use_filter,filter_type=args.filter_type,terminate_on_safety_violation=args.terminate_on_safety_violation,cost_limit_per_step=args.cost_limit_per_step)
    eenv=SafeSafetyGymWrapper(env_id=args.env_id,use_filter=args.use_filter,filter_type=args.filter_type,terminate_on_safety_violation=args.terminate_on_safety_violation,cost_limit_per_step=args.cost_limit_per_step)
    obs,_=env.reset(seed=args.seed); agent=make_algo(args,env.observation_space.shape[0],env.action_space.shape[0]); key=jax.random.PRNGKey(args.seed+7); buf=[]; train=[]; ev=[]; step_metrics=[]; best={'return_':(-1e18,'best_return_checkpoint.pkl',max),'cost_return':(1e18,'best_cost_checkpoint.pkl',min),'success_rate':(-1e18,'best_success_checkpoint.pkl',max),'safety_score':(1e18,'best_safety_checkpoint.pkl',min)}
    start=time.perf_counter(); pbar=tqdm(range(1,args.total_steps+1))
    try:
        for step in pbar:
            if step<args.start_steps:
                raw=env.action_space.sample()
            else:
                key,ak=jax.random.split(key)
                raw=np.asarray(agent.get_action(ak,obs[None])[0])
            next_obs,reward,term,trunc,info=env.step(raw)
            exp=SafePullbackExperience.create(obs,raw,info['exec_action'],reward,term,trunc,next_obs,info)
            buf.append(exp)
            buf=buf[-1_000_000:]
            obs=next_obs
            row={
                "step": step,
                "reward": float(reward),
                "cost": safe_info_float(info, "cost", 0.0),
                "filter_active": safe_info_float(info, "filter_active", 0.0),
                "projection_residual": safe_info_float(info, "projection_residual", 0.0),
                "projection_cost": safe_info_float(info, "projection_cost", 0.0),
                "raw_action_norm": safe_info_float(info, "raw_action_norm", 0.0),
                "exec_action_norm": safe_info_float(info, "exec_action_norm", 0.0),
                "raw_action_abs_max": safe_info_float(info, "raw_action_abs_max", 0.0),
                "exec_action_abs_max": safe_info_float(info, "exec_action_abs_max", 0.0),
                "safe_candidate_ratio": safe_info_float(info, "safe_candidate_ratio", np.nan),
                "emergency_active": safe_info_float(info, "emergency_active", 0.0),
                "gt_known": safe_info_float(info, "gt_known", np.nan),
                "current_min_h": safe_info_float(info, "current_min_h", np.nan),
                "predicted_min_h": safe_info_float(info, "predicted_min_h", np.nan),
                "filter_active_005": safe_info_float(info, "filter_active_005", np.nan),
                "filter_active_010": safe_info_float(info, "filter_active_010", np.nan),
            }
            step_metrics.append(row)
            if term or trunc: obs,_=env.reset()
            for k, v in info.items():
                try:
                    if isinstance(v, (str, bytes)):
                        continue
                    if np.isscalar(v):
                        fv = float(v)
                        if np.isfinite(fv):
                            writer.add_scalar(f'train_env_info/{k}', fv, step)
                except (TypeError, ValueError):
                    continue
            if step>=args.update_after and len(buf)>=args.batch_size:
                out=agent.update(jax.random.PRNGKey(args.seed+step),sample_batch(buf,args.batch_size)); out['step']=step; train.append(out)
            if step%args.eval_interval==0:
                m=eval_agent(agent,args,eenv); m['step']=step; ev.append(m)
                with open(log/'checkpoints'/f'checkpoint_step_{step}.pkl','wb') as f: pickle.dump(agent.state,f)
                if np.isfinite(m['return_']) and m['return_']>best['return_'][0]: best['return_']=(m['return_'],'best_return_checkpoint.pkl',max); pickle.dump(agent.state,open(log/'best_return_checkpoint.pkl','wb'))
                if np.isfinite(m['cost_return']) and m['cost_return']<best['cost_return'][0]: best['cost_return']=(m['cost_return'],'best_cost_checkpoint.pkl',min); pickle.dump(agent.state,open(log/'best_cost_checkpoint.pkl','wb'))
                if np.isfinite(m['success_rate']) and m['success_rate']>best['success_rate'][0]: best['success_rate']=(m['success_rate'],'best_success_checkpoint.pkl',max); pickle.dump(agent.state,open(log/'best_success_checkpoint.pkl','wb'))
                if np.isfinite(m['safety_score']) and m['safety_score']<best['safety_score'][0]: best['safety_score']=(m['safety_score'],'best_safety_checkpoint.pkl',min); pickle.dump(agent.state,open(log/'best_safety_checkpoint.pkl','wb'))
                if args.save_partial_on_eval:
                    save_partial_outputs(log, train, ev, step_metrics, agent.state)
                if args.save_curve_interval == 0 or (args.save_curve_interval > 0 and step % args.save_curve_interval == 0):
                    save_curves(log, step_metrics, ev, train, window=args.plot_train_window, curve_dir=args.curve_dir)
            pbar.set_postfix(r=f"{reward:.2f}",cost=f"{info.get('cost',0):.2f}",FAR=f"{info.get('filter_active',0):.2f}",APR=f"{info.get('projection_residual',0):.3f}",raw_norm=f"{info.get('raw_action_norm',0):.2f}",exec_norm=f"{info.get('exec_action_norm',0):.2f}",eval_ret=f"{(ev[-1]['return_'] if ev else np.nan):.1f}",eval_cost=f"{(ev[-1]['cost_return'] if ev else np.nan):.2f}",succ=f"{(ev[-1]['success_rate'] if ev else 0):.2f}",sps=f"{step/(time.perf_counter()-start):.1f}")
    except KeyboardInterrupt:
        print("[INTERRUPTED] saving partial metrics/checkpoint...")
        save_partial_outputs(log, train, ev, step_metrics, agent.state)
        if ev:
            save_curves(log, step_metrics, ev, train, window=args.plot_train_window, curve_dir=args.curve_dir)
        writer.close()
        raise
    save_partial_outputs(log, train, ev, step_metrics, agent.state)
    if ev:
        save_curves(log, step_metrics, ev, train, window=args.plot_train_window, curve_dir=args.curve_dir)
    writer.close()

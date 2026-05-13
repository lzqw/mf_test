import argparse, json, pickle, sys, time
from pathlib import Path
from typing import NamedTuple
import jax, jax.numpy as jnp, numpy as np
from tqdm.auto import tqdm
try:
    from tensorboardX import SummaryWriter
except ImportError:
    from torch.utils.tensorboard import SummaryWriter

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from envs.metadrive_safe_wrapper import SafeMetaDriveSampleWrapper
from relax.algorithm.safe_pullback_rf2_sac_ent_metadrive import SafePullbackRF2SACENTMetaDrive
from relax.network.safe_pullback_rf2_sac_ent import create_safe_pullback_rf2_sac_ent_net
from relax.safety.metadrive_sample_filter import SampleVehicleFilterConfig
from scripts.safe_pullback_experience import SafePullbackExperience

class Batch(NamedTuple):
    obs:jnp.ndarray; raw_action:jnp.ndarray; action:jnp.ndarray; reward:jnp.ndarray; done:jnp.ndarray; next_obs:jnp.ndarray; projection_residual:jnp.ndarray; projection_cost:jnp.ndarray

def sample_batch(buf,b):
    idx=np.random.randint(0,len(buf),size=b); it=[buf[i] for i in idx]
    return Batch(*(jnp.asarray(np.stack([getattr(x,f) for x in it])) for f in Batch._fields))

def make_algo(args, obs_dim, act_dim):
    key=jax.random.PRNGKey(args.seed)
    net, params = create_safe_pullback_rf2_sac_ent_net(key, obs_dim, act_dim, hidden_sizes=[256]*3, diffusion_hidden_sizes=[256]*3,
        num_timesteps=args.diffusion_steps, num_ent_timesteps=args.num_ent_timesteps, alpha_value=args.alpha_value,
        fixed_alpha=args.fixed_alpha, init_alpha=args.init_alpha, noise_scale=args.policy_noise_scale)
    return SafePullbackRF2SACENTMetaDrive(net, params, gamma=args.gamma, gamma_p=args.gamma_p, lr=args.lr, alpha_lr=args.alpha_lr,
        sample_k=args.sample_k, lambda_p=args.lambda_p, use_projection_critic=args.use_projection_critic, fixed_alpha=args.fixed_alpha,
        alpha_value=args.alpha_value, lambda_p_warmup_steps=args.lambda_p_warmup_steps, use_tn_energy=args.use_tn_energy,
        entropy_reg_mode=args.entropy_reg_mode, candidate_temp=args.candidate_temp, beta_normal_entropy=args.beta_normal_entropy,
        min_effective_entropy=args.min_effective_entropy, target_effective_entropy=args.target_effective_entropy,
        normal_energy_coef=args.normal_energy_coef, weight_mix=args.weight_mix, action_limit=1.0)

def is_finite_number(x):
    try:
        return np.isfinite(float(x))
    except Exception:
        return False

def build_env_kwargs(args):
    return {}


def safe_nanmean(values, default=None):
    finite_values = []
    for v in values:
        try:
            fv = float(v)
            if np.isfinite(fv):
                finite_values.append(fv)
        except Exception:
            pass
    if len(finite_values) == 0:
        return default
    return float(np.mean(finite_values))


def save_state_checkpoint(agent, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(agent.state, f)


def eval_agent(agent,args):
    mets=[]
    env=SafeMetaDriveSampleWrapper(args.env_name,use_filter=args.use_filter,filter_type=args.filter_type,filter_cfg=SampleVehicleFilterConfig(num_local_samples=args.num_local_samples,num_prev_samples=args.num_prev_samples,horizon=args.horizon,dt=args.dt,steer_limit=args.steer_limit,throttle_limit=args.throttle_limit,brake_limit=args.brake_limit,max_dsteer=args.max_dsteer,max_daccel=args.max_daccel,safe_radius=args.safe_radius,ttc_min=args.ttc_min,h_vo_margin=args.h_vo_margin,lane_margin_min=args.lane_margin_min),**build_env_kwargs(args))
    try:
        for ep in range(args.eval_episodes):
            obs,_=env.reset(seed=args.seed+1000+ep); done=False; ret=0.0; steps=0; far=[]; apr=[]; info={}; costs=[]; vels=[]; nsc=[]; fbs=[]; vcr=[]; mpd=[]; mpt=[]; mph=[]; vov=[]; ttcv=[]; lv=[]; ft=[]
            while not done and steps<1000:
                a=env.action_space.sample() if agent is None else np.asarray(agent.get_action(jax.random.PRNGKey(args.seed+ep+steps), obs[None])[0])
                obs,r,term,trunc,info=env.step(a); done=term or trunc; ret+=float(r); steps+=1; far.append(float(info.get('filter_active',0.0))); apr.append(float(info.get('projection_residual',0.0))); costs.append(float(info.get('cost',0.0))); vels.append(float(info.get('velocity',0.0))); nsc.append(float(info.get('no_safe_candidate',0.0))); fbs.append(float(info.get('fallback',0.0))); vcr.append(float(info.get('valid_candidate_ratio',0.0))); mpd.append(float(info.get('min_pred_dist',np.nan))); mpt.append(float(info.get('min_pred_ttc',np.nan))); mph.append(float(info.get('min_pred_h_vo',np.nan))); vov.append(float(info.get('vo_violation',0.0))); ttcv.append(float(info.get('ttc_violation',0.0))); lv.append(float(info.get('lane_violation',0.0))); ft.append(float(info.get('filter_time_ms',0.0)))
            metric = dict(return_=ret,episode_length=steps,success_rate=float(info.get("is_success",0.0)),crash_rate=float(info.get("crash",0.0)),out_of_road_rate=float(info.get("out_of_road",0.0)),cost_mean=float(np.mean(costs)) if costs else 0.0,velocity_mean=float(np.mean(vels)) if vels else 0.0,FAR=float(np.mean(far)) if far else 0.0,APR=float(np.mean(apr)) if apr else 0.0,safe_violation_rate=float(info.get("safe_violation",0.0)),sample_filter_active_rate=float(np.mean(far)) if far else 0.0,no_safe_candidate_rate=float(np.mean(nsc)) if nsc else 0.0,fallback_rate=float(np.mean(fbs)) if fbs else 0.0,valid_candidate_ratio=float(np.mean(vcr)) if vcr else 0.0,min_pred_dist=safe_nanmean(mpd,0.0),min_pred_ttc=safe_nanmean(mpt,0.0),min_pred_h_vo=safe_nanmean(mph,0.0),vo_violation_rate=float(np.mean(vov)) if vov else 0.0,ttc_violation_rate=float(np.mean(ttcv)) if ttcv else 0.0,lane_violation_rate=float(np.mean(lv)) if lv else 0.0,filter_time_ms=float(np.mean(ft)) if ft else 0.0)
            mets.append(metric)
    finally:
        env.close()
    keys = mets[0].keys()
    result = {}
    for k in keys:
        value = safe_nanmean([m[k] for m in mets], default=None)
        if value is not None:
            result[k] = value
    return result

def main():
    p=argparse.ArgumentParser();
    for k,d,t in [('env_name','FlatThreeLaneStraight',str),('seed',0,int),('total_steps',1000000,int),('start_steps',10000,int),('update_after',10000,int),('batch_size',256,int),('eval_interval',10000,int),('eval_episodes',10,int),('log_dir',None,str)]: p.add_argument(f'--{k}',default=d,type=t,required=(k=='log_dir'))
    p.add_argument('--use_filter',action='store_true'); p.add_argument('--filter_type',choices=['none','rate','sample_vo'],default='sample_vo'); p.add_argument('--num_local_samples',type=int,default=64); p.add_argument('--num_prev_samples',type=int,default=32); p.add_argument('--horizon',type=int,default=8); p.add_argument('--dt',type=float,default=0.1); p.add_argument('--steer_limit',type=float,default=0.7); p.add_argument('--throttle_limit',type=float,default=0.8); p.add_argument('--brake_limit',type=float,default=-0.8); p.add_argument('--max_dsteer',type=float,default=0.12); p.add_argument('--max_daccel',type=float,default=0.2); p.add_argument('--safe_radius',type=float,default=4.0); p.add_argument('--ttc_min',type=float,default=1.5); p.add_argument('--h_vo_margin',type=float,default=0.2); p.add_argument('--lane_margin_min',type=float,default=0.3)
    p.add_argument('--lr',type=float,default=3e-4); p.add_argument('--alpha_lr',type=float,default=1e-2); p.add_argument('--gamma',type=float,default=0.99); p.add_argument('--gamma_p',type=float,default=0.99); p.add_argument('--sample_k',type=int,default=256); p.add_argument('--lambda_p',type=float,default=0.1); p.add_argument('--lambda_p_warmup_steps',type=int,default=100000); p.add_argument('--use_projection_critic',action='store_true')
    p.add_argument('--fixed_alpha',action='store_true'); p.add_argument('--alpha_value',type=float,default=0.1); p.add_argument('--init_alpha',type=float,default=0.1)
    p.add_argument('--diffusion_steps',type=int,default=10); p.add_argument('--num_ent_timesteps',type=int,default=10); p.add_argument('--policy_noise_scale',type=float,default=0.3)
    p.add_argument('--entropy_reg_mode',choices=['legacy','likelihood_tn','flac_tn'],default='legacy'); p.add_argument('--use_tn_energy',action='store_true'); p.add_argument('--candidate_temp',type=float,default=0.1); p.add_argument('--beta_normal_entropy',type=float,default=1.0); p.add_argument('--min_effective_entropy',type=float,default=-20.0); p.add_argument('--target_effective_entropy',type=float,default=1.0); p.add_argument('--normal_energy_coef',type=float,default=0.05); p.add_argument('--weight_mix',type=float,default=0.05)
    args=p.parse_args(); np.random.seed(args.seed)
    log=Path(args.log_dir); log.mkdir(parents=True,exist_ok=True); writer=SummaryWriter(str(log/'tb')); (log/'args.json').write_text(json.dumps(vars(args),indent=2,sort_keys=True))
    env=SafeMetaDriveSampleWrapper(args.env_name,use_filter=args.use_filter,filter_type=args.filter_type,filter_cfg=SampleVehicleFilterConfig(num_local_samples=args.num_local_samples,num_prev_samples=args.num_prev_samples,horizon=args.horizon,dt=args.dt,steer_limit=args.steer_limit,throttle_limit=args.throttle_limit,brake_limit=args.brake_limit,max_dsteer=args.max_dsteer,max_daccel=args.max_daccel,safe_radius=args.safe_radius,ttc_min=args.ttc_min,h_vo_margin=args.h_vo_margin,lane_margin_min=args.lane_margin_min),**build_env_kwargs(args)); obs,_=env.reset(seed=args.seed)
    print("=" * 80)
    print("MetaDrive Safe-Pullback Training")
    print("env_name:", args.env_name)
    print("filter_type:", args.filter_type)
    print("observation_space:", env.observation_space)
    print("action_space:", env.action_space)
    print("obs_dim:", env.observation_space.shape[0])
    print("act_dim:", env.action_space.shape[0])
    print("use_filter:", args.use_filter)
    print("entropy_reg_mode:", args.entropy_reg_mode)
    print("=" * 80)
    agent = make_algo(args, env.observation_space.shape[0], env.action_space.shape[0])
    key = jax.random.PRNGKey(args.seed + 7)
    buf = []
    train = []
    ev = []
    train_start_time = time.perf_counter()
    last_eval_metrics = {
        "return_": float("nan"),
        "success_rate": float("nan"),
        "FAR": float("nan"),
        "APR": float("nan"),
        "crash_rate": float("nan"),
        "out_of_road_rate": float("nan"),
    }
    best_metrics = {
        "return_": {"value": -np.inf, "path": log / "best_return_checkpoint.pkl", "mode": "max"},
        "success_rate": {"value": -np.inf, "path": log / "best_success_checkpoint.pkl", "mode": "max"},
        "safety_score": {"value": np.inf, "path": log / "best_safety_checkpoint.pkl", "mode": "min"},
    }
    pbar = tqdm(
        range(1, args.total_steps + 1),
        total=args.total_steps,
        dynamic_ncols=True,
        smoothing=0.05,
        desc=f"{args.env_name} | metadrive",
    )
    for step in pbar:
        if step<args.start_steps: raw=env.action_space.sample()
        else: key,ak=jax.random.split(key); raw=np.asarray(agent.get_action(ak,obs[None])[0])
        nobs,reward,term,trunc,info=env.step(raw); exp=SafePullbackExperience.create(obs,raw,info['exec_action'],reward,term,trunc,nobs,info); buf.append(exp); buf=buf[-1_000_000:]; obs=nobs
        if term or trunc: obs,_=env.reset()
        for k,tg in [('reward','train_env/reward'),('filter_active','train_env/filter_active'),('projection_residual','train_env/projection_residual'),('projection_cost','train_env/projection_cost'),('fall','train_env/fall'),('head_height','train_env/head_height'),('torso_upright','train_env/torso_upright'),('joint_angle_abs_max','train_env/joint_angle_abs_max'),('joint_vel_abs_max','train_env/joint_vel_abs_max')]:
            if k in info:
                val = info[k]
                if is_finite_number(val):
                    writer.add_scalar(tg, float(val), step)
        for k, v in info.items():
            if isinstance(v, (list, tuple, np.ndarray)):
                continue
            if is_finite_number(v):
                writer.add_scalar(f"train_env_info/{k}", float(v), step)
        if step>=args.update_after and len(buf)>=args.batch_size:
            key,uk=jax.random.split(key); out=agent.update(uk,sample_batch(buf,args.batch_size)); out['step']=step; train.append(out)
            for k,v in out.items():
                if k!='step' and is_finite_number(v): writer.add_scalar(f'train/{k}',float(v),step)
        if step % 10 == 0 or step == 1:
            elapsed = time.perf_counter() - train_start_time
            steps_per_sec = step / max(elapsed, 1e-8)
            pbar.set_postfix(
                {
                    "r": f"{float(reward):.2f}",
                    "FAR": f"{float(info.get('filter_active', 0.0)):.2f}",
                    "APR": f"{float(info.get('projection_residual', 0.0)):.3f}",
                    "cost": f"{float(info.get('cost', 0.0)):.2f}",
                    "eval_ret": f"{last_eval_metrics.get('return_', float('nan')):.1f}",
                    "succ": f"{last_eval_metrics.get('success_rate', float('nan')):.2f}",
                    "sps": f"{steps_per_sec:.1f}",
                }
            )
        if step%args.eval_interval==0:
            eval_start = time.perf_counter()
            e=eval_agent(agent,args)
            eval_time = time.perf_counter() - eval_start
            e['step']=step
            e['safety_score']=float(e.get('crash_rate',0.0)+e.get('out_of_road_rate',0.0)+e.get('cost_mean',0.0)+e.get('fallback_rate',0.0))
            e["eval_time_sec"] = float(eval_time)
            ev.append(e)
            last_eval_metrics.update(e)
            elapsed = time.perf_counter() - train_start_time
            avg_steps_per_sec = step / max(elapsed, 1e-8)
            eta_sec = max(args.total_steps - step, 0) / max(avg_steps_per_sec, 1e-8)
            msg = (
                f"[eval step {step}/{args.total_steps}] "
                f"return={e.get('return_', float('nan')):.3f}, "
                f"success={e.get('success_rate', float('nan')):.3f}, "
                f"FAR={e.get('FAR', float('nan')):.3f}, "
                f"APR={e.get('APR', float('nan')):.3f}, "
                f"crash={e.get('crash_rate', float('nan')):.3f}, out={e.get('out_of_road_rate', float('nan')):.3f}"
            )
            msg += (
                f", eval_time={eval_time:.1f}s"
                f", avg_sps={avg_steps_per_sec:.2f}"
                f", eta={eta_sec / 60:.1f}min"
            )
            pbar.write(msg)
            for k,v in e.items():
                if k!='step' and is_finite_number(v): writer.add_scalar(f'eval/{k}',float(v),step)
            writer.add_scalar("time/eval_time_sec", float(eval_time), step)
            writer.add_scalar("time/avg_steps_per_sec", float(avg_steps_per_sec), step)
            writer.add_scalar("time/eta_sec", float(eta_sec), step)
            writer.flush()
            checkpoint_dir = log / "checkpoints"
            step_checkpoint_path = checkpoint_dir / f"checkpoint_step_{step}.pkl"
            save_state_checkpoint(agent, step_checkpoint_path)
            for metric_name, metric_cfg in best_metrics.items():
                if metric_name not in e or not is_finite_number(e[metric_name]):
                    continue
                metric_value = float(e[metric_name])
                is_better = metric_value > metric_cfg["value"] if metric_cfg["mode"] == "max" else metric_value < metric_cfg["value"]
                if is_better:
                    metric_cfg["value"] = metric_value
                    save_state_checkpoint(agent, metric_cfg["path"])
                    print(
                        f"[best] step={step}, metric={metric_name}, value={metric_value}, saved={metric_cfg['path']}"
                    )
    pbar.close()
    total_elapsed = time.perf_counter() - train_start_time
    print("=" * 80)
    print("Training finished.")
    print(f"Total elapsed time: {total_elapsed / 60:.2f} min")
    print(f"Average speed: {args.total_steps / max(total_elapsed, 1e-8):.2f} step/s")
    print("=" * 80)
    writer.add_scalar("time/total_elapsed_sec", float(total_elapsed), args.total_steps)
    writer.flush()
    pickle.dump(train,open(log/'train_metrics.pkl','wb')); pickle.dump(ev,open(log/'eval_metrics.pkl','wb')); pickle.dump(agent.state,open(log/'checkpoint.pkl','wb'))
if __name__=='__main__': main()

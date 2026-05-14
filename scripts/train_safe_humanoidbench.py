import argparse, json, pickle, sys, time
from pathlib import Path
from types import SimpleNamespace
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

from envs.humanoidbench_safe_wrapper import SafeHumanoidBenchWrapper
from relax.algorithm.safe_pullback_rf2_sac_ent_humanoid import SafePullbackRF2SACENTHumanoid
from relax.network.safe_pullback_rf2_sac_ent import create_safe_pullback_rf2_sac_ent_net
from relax.safety.humanoidbench_filter import HumanoidSafeFilterConfig
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
    return SafePullbackRF2SACENTHumanoid(net, params, gamma=args.gamma, gamma_p=args.gamma_p, lr=args.lr, alpha_lr=args.alpha_lr,
        sample_k=args.sample_k, lambda_p=args.lambda_p, use_projection_critic=args.use_projection_critic, fixed_alpha=args.fixed_alpha,
        alpha_value=args.alpha_value, lambda_p_warmup_steps=args.lambda_p_warmup_steps, use_tn_energy=args.use_tn_energy,
        entropy_reg_mode=args.entropy_reg_mode, candidate_temp=args.candidate_temp, beta_normal_entropy=args.beta_normal_entropy,
        min_effective_entropy=args.min_effective_entropy, target_effective_entropy=args.target_effective_entropy,
        normal_energy_coef=args.normal_energy_coef, weight_mix=args.weight_mix, residual_radius=args.residual_radius, action_limit=1.0,
        use_goal_candidate=args.use_goal_candidate, high_level_max_delta=args.max_delta)

def is_finite_number(x):
    try: return np.isfinite(float(x))
    except Exception: return False

def build_env_kwargs(args):
    kwargs = {}
    if getattr(args, "blocked_hands", None) is not None: kwargs["blocked_hands"] = args.blocked_hands
    if getattr(args, "small_obs", None) is not None: kwargs["small_obs"] = args.small_obs
    return kwargs

def make_env(args, seed=None, render_mode=None):
    env = SafeHumanoidBenchWrapper(
        args.env_name, use_filter=args.use_filter, render_mode=render_mode,
        filter_cfg=HumanoidSafeFilterConfig(residual_radius=args.residual_radius,smooth_radius=args.smooth_radius,max_delta=args.max_delta,target_step_radius=args.target_step_radius,reachable_radius=args.reachable_radius,z_min_safe=args.z_min_safe,z_max_safe=args.z_max_safe),
        policy_path=args.policy_path, mean_path=args.mean_path, var_path=args.var_path, policy_type=args.policy_type,
        augment_reach_obs=args.augment_reach_obs,
        reference_filter_mode=args.reference_filter_mode,
        reference_filter_threshold=args.reference_filter_threshold,
        reference_filter_type=args.reference_filter_type,
        **build_env_kwargs(args),
    )
    return env

def safe_nanmean(values, default=None):
    finite=[]
    for v in values:
        try:
            fv=float(v)
            if np.isfinite(fv): finite.append(fv)
        except Exception: pass
    return default if not finite else float(np.mean(finite))

def save_state_checkpoint(agent, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f: pickle.dump(agent.state, f)

def eval_agent(agent,args):
    mets=[]; env=make_env(args, seed=args.seed+1000)
    try:
        for ep in range(args.eval_episodes):
            obs,_=env.reset(seed=args.seed+1000+ep); done=False; ret=0.0; steps=0; falls=[]; far=[]; apr=[]; ref_active=[]; ref_dist=[]; total_apr=[]; info={}; last_term=False
            while not done and steps<1000:
                a=env.action_space.sample() if agent is None else np.asarray(agent.get_action(jax.random.PRNGKey(args.seed+ep+steps), obs[None])[0])
                obs,r,term,trunc,info=env.step(a); done=term or trunc; ret+=float(r); steps+=1
                falls.append(float(info.get('fall',0.0))); far.append(float(info.get('filter_active',0.0))); apr.append(float(info.get('projection_residual',0.0)))
                ref_active.append(float(info.get("reference_correction_active", 0.0)))
                ref_dist.append(float(info.get("raw_to_reference_dist", np.nan)))
                total_apr.append(float(info.get("total_projection_residual", info.get("projection_residual", 0.0))))
                last_term=bool(term)
            mets.append(dict(return_=ret,episode_length=steps,FAR=safe_nanmean(far,0.0),APR=safe_nanmean(apr,0.0),total_APR=safe_nanmean(total_apr,0.0),fall_rate=float(np.max(falls)) if falls else 0.0,success_rate=float(info.get("is_success",0.0)),hand_dist=float(info.get("hand_dist",np.nan)),target_dist=float(info.get("target_dist",np.nan)),reference_correction_rate=safe_nanmean(ref_active,0.0),raw_to_reference_dist=safe_nanmean(ref_dist,np.nan),early_termination_rate=float(last_term and float(info.get("is_success",0.0))<=0.0)))
    finally: env.close()
    out={}
    for k in mets[0].keys():
        v=safe_nanmean([m[k] for m in mets],None)
        if v is not None: out[k]=v
    return out

def main():
    p=argparse.ArgumentParser()
    for k,d,t in [('env_name','h1hand-reach-v0',str),('seed',0,int),('total_steps',1000000,int),('start_steps',10000,int),('update_after',10000,int),('batch_size',256,int),('eval_interval',10000,int),('eval_episodes',10,int),('log_dir',None,str)]: p.add_argument(f'--{k}',default=d,type=t,required=(k=='log_dir'))
    p.add_argument('--use_filter',action='store_true'); p.add_argument('--residual_radius',type=float,default=0.35); p.add_argument('--smooth_radius',type=float,default=0.25); p.add_argument('--max_delta',type=float,default=0.1); p.add_argument('--target_step_radius',type=float,default=0.08); p.add_argument('--reachable_radius',type=float,default=0.45); p.add_argument('--z_min_safe',type=float,default=0.4); p.add_argument('--z_max_safe',type=float,default=1.8)
    p.add_argument('--lr',type=float,default=3e-4); p.add_argument('--alpha_lr',type=float,default=1e-2); p.add_argument('--gamma',type=float,default=0.99); p.add_argument('--gamma_p',type=float,default=0.99); p.add_argument('--sample_k',type=int,default=256); p.add_argument('--lambda_p',type=float,default=0.1); p.add_argument('--lambda_p_warmup_steps',type=int,default=100000); p.add_argument('--use_projection_critic',action='store_true')
    p.add_argument('--fixed_alpha',action='store_true'); p.add_argument('--alpha_value',type=float,default=0.1); p.add_argument('--init_alpha',type=float,default=0.1)
    p.add_argument('--diffusion_steps',type=int,default=10); p.add_argument('--num_ent_timesteps',type=int,default=10); p.add_argument('--policy_noise_scale',type=float,default=0.3)
    p.add_argument('--entropy_reg_mode',choices=['legacy','likelihood_tn','flac_tn'],default='legacy'); p.add_argument('--use_tn_energy',action='store_true'); p.add_argument('--candidate_temp',type=float,default=0.1); p.add_argument('--beta_normal_entropy',type=float,default=1.0); p.add_argument('--min_effective_entropy',type=float,default=-20.0); p.add_argument('--target_effective_entropy',type=float,default=1.0); p.add_argument('--normal_energy_coef',type=float,default=0.05); p.add_argument('--weight_mix',type=float,default=0.05)
    p.add_argument('--policy_path',type=str,default=None); p.add_argument('--mean_path',type=str,default=None); p.add_argument('--var_path',type=str,default=None); p.add_argument('--policy_type',type=str,default=None)
    p.add_argument("--blocked_hands", type=str, default=None); p.add_argument("--small_obs", type=str, default=None); p.add_argument("--augment_reach_obs", action="store_true"); p.add_argument("--use_goal_candidate", action="store_true")
    p.add_argument("--reference_filter_mode", type=str, default="none", choices=["none", "goal"]); p.add_argument("--reference_filter_threshold", type=float, default=0.25); p.add_argument("--reference_filter_type", type=str, default="replace", choices=["replace", "ball"])
    p.add_argument("--num_envs", type=int, default=1); p.add_argument("--updates_per_step", type=int, default=1)
    args=p.parse_args(); np.random.seed(args.seed)
    log=Path(args.log_dir); log.mkdir(parents=True,exist_ok=True); writer=SummaryWriter(str(log/'tb')); (log/'args.json').write_text(json.dumps(vars(args),indent=2,sort_keys=True))

    env0 = make_env(args, seed=args.seed); obs0,_=env0.reset(seed=args.seed)
    print("num_envs:", args.num_envs); print("updates_per_step:", args.updates_per_step); print("reference_filter_mode:", args.reference_filter_mode); print("reference_filter_threshold:", args.reference_filter_threshold); print("reference_filter_type:", args.reference_filter_type)
    print("obs_dim:", env0.observation_space.shape[0]); print("act_dim:", env0.action_space.shape[0])
    agent = make_algo(args, env0.observation_space.shape[0], env0.action_space.shape[0]); key = jax.random.PRNGKey(args.seed + 7)

    envs=[env0] if args.num_envs==1 else [env0]+[make_env(args,seed=args.seed+i) for i in range(1,args.num_envs)]
    obs_list=[obs0] + [env.reset(seed=args.seed+i)[0] for i,env in enumerate(envs[1:], start=1)]
    buf=[]; train=[]; ev=[]; start=time.perf_counter(); global_env_steps=0
    pbar=tqdm(range(1,args.total_steps+1),total=args.total_steps,dynamic_ncols=True,smoothing=0.05,desc=f"{args.env_name} | {args.policy_type or 'flat'}")
    for step in pbar:
        step_infos=[]
        for i,env in enumerate(envs):
            obs=obs_list[i]
            if step<args.start_steps: raw=env.action_space.sample()
            else:
                key,ak=jax.random.split(key)
                raw=np.asarray(agent.get_action(jax.random.fold_in(ak,i),obs[None])[0])
            nobs,reward,term,trunc,info=env.step(raw)
            buf.append(SafePullbackExperience.create(obs,raw,info['exec_action'],reward,term,trunc,nobs,info)); buf=buf[-1_000_000:]
            obs_list[i]=env.reset(seed=args.seed+step+i)[0] if (term or trunc) else nobs
            info=dict(info); info['reward']=float(reward); step_infos.append(info)
        global_env_steps += len(envs)
        mean_keys=['reward','filter_active','projection_residual','total_projection_residual','hand_dist','reward_success','reference_correction_active','raw_to_reference_dist','reference_corrected_to_ref_dist']
        for k in mean_keys:
            mv=safe_nanmean([inf.get(k,np.nan) for inf in step_infos],None)
            if mv is not None and is_finite_number(mv): writer.add_scalar(f'train_env/{k}_mean',float(mv),step)
        if global_env_steps>=args.update_after and len(buf)>=args.batch_size:
            for _ in range(args.updates_per_step):
                key,uk=jax.random.split(key); out=agent.update(uk,sample_batch(buf,args.batch_size)); out['step']=step; train.append(out)
                for k,v in out.items():
                    if k!='step' and is_finite_number(v): writer.add_scalar(f'train/{k}',float(v),step)
        if step % 10 == 0 or step == 1:
            elapsed=time.perf_counter()-start; sps=global_env_steps/max(elapsed,1e-8)
            pbar.set_postfix({'outer_step':step,'global_env_steps':global_env_steps,'buffer_size':len(buf),'sps':f'{sps:.1f}'})
        if step%args.eval_interval==0:
            e=eval_agent(agent,args); e['step']=step; ev.append(e)
            for k,v in e.items():
                if k!='step' and is_finite_number(v): writer.add_scalar(f'eval/{k}',float(v),step)
    for env in envs: env.close()
    writer.flush(); pickle.dump(train,open(log/'train_metrics.pkl','wb')); pickle.dump(ev,open(log/'eval_metrics.pkl','wb')); pickle.dump(agent.state,open(log/'checkpoint.pkl','wb'))

if __name__=='__main__': main()

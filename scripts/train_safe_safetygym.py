import argparse, json, pickle, sys, time
from pathlib import Path
from typing import NamedTuple
import jax, jax.numpy as jnp, numpy as np
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
    net, params = create_safe_pullback_rf2_sac_ent_net(key, obs_dim, act_dim, hidden_sizes=[256]*3, diffusion_hidden_sizes=[256]*3, num_timesteps=args.diffusion_steps, num_ent_timesteps=args.num_ent_timesteps, alpha_value=args.alpha_value, fixed_alpha=args.fixed_alpha, init_alpha=args.init_alpha, noise_scale=args.policy_noise_scale)
    return SafePullbackRF2SACENTSafetyGym(net, params, gamma=args.gamma, gamma_p=args.gamma_p, lr=args.lr, alpha_lr=args.alpha_lr, sample_k=args.sample_k, lambda_p=args.lambda_p, use_projection_critic=args.use_projection_critic, fixed_alpha=args.fixed_alpha, alpha_value=args.alpha_value, lambda_p_warmup_steps=args.lambda_p_warmup_steps, use_tn_energy=args.use_tn_energy, entropy_reg_mode=args.entropy_reg_mode)

def safe_nanmean(xs):
    arr=np.asarray(xs,dtype=np.float32)
    finite=arr[np.isfinite(arr)]
    return float(np.mean(finite)) if finite.size else np.nan

def eval_agent(agent, args, env):
    ms=[]
    for ep in range(args.eval_episodes):
        obs,_=env.reset(seed=args.seed+1000+ep); done=False; ret=0; l=0; costs=[]; fars=[]; aprs=[]; rns=[]; ens=[]; mins=[]; cbfs=[]; sv=[]; cv=[]; sc_ratio=[]; emergency=[]; gh=[]; fh=[]; lh=[]; rh=[]; info={}
        while not done and l<1000:
            a=np.asarray(agent.get_action(jax.random.PRNGKey(args.seed+ep*10000+l),obs[None])[0]); obs,r,t,tr,info=env.step(a); done=t or tr; ret+=float(r); l+=1
            costs.append(float(info.get('cost',0))); fars.append(float(info.get('filter_active',0))); aprs.append(float(info.get('projection_residual',0))); rns.append(float(info.get('raw_action_norm',0))); ens.append(float(info.get('exec_action_norm',0))); mins.append(float(info.get('min_h',np.nan))); cbfs.append(float(info.get('cbf_violation',0))); sv.append(float(info.get('safety_violation',0))); cv.append(float(info.get('constraint_violation',0))); sc_ratio.append(float(info.get('safe_candidate_ratio',0.0))); emergency.append(float(info.get('emergency_active',0.0))); gh.append(float(info.get('global_min_h',np.nan))); fh.append(float(info.get('front_h',np.nan))); lh.append(float(info.get('left_h',np.nan))); rh.append(float(info.get('right_h',np.nan)))
        success=float(info.get('is_success', info.get('success', info.get('goal_met', info.get('task_success', 0.0)))))
        ms.append(dict(return_=ret, episode_length=l, success_rate=success, cost_return=float(np.sum(costs)), cost_rate=float(np.mean(np.array(costs)>0)) if costs else 0.0, safety_violation_rate=float(np.mean(sv)) if sv else 0.0, constraint_violation_rate=float(np.mean(cv)) if cv else 0.0, FAR=float(np.mean(fars)) if fars else 0.0, APR=float(np.mean(aprs)) if aprs else 0.0, projection_cost=float(np.mean(np.square(aprs))) if aprs else 0.0, raw_action_norm=float(np.mean(rns)) if rns else 0.0, exec_action_norm=float(np.mean(ens)) if ens else 0.0, min_h=safe_nanmean(mins) if mins else np.nan, cbf_violation_rate=float(np.mean(cbfs)) if cbfs else 0.0, safe_candidate_ratio=float(np.mean(sc_ratio)) if sc_ratio else 0.0, emergency_rate=float(np.mean(emergency)) if emergency else 0.0, global_min_h=safe_nanmean(gh) if gh else np.nan, front_h=safe_nanmean(fh) if fh else np.nan, left_h=safe_nanmean(lh) if lh else np.nan, right_h=safe_nanmean(rh) if rh else np.nan))
    out={k:safe_nanmean([m[k] for m in ms]) for k in ms[0].keys()}
    out['safety_score']=out['cost_return']+10.0*out['safety_violation_rate']+out['FAR']+0.1*out['APR']
    return out

if __name__=='__main__':
    p=argparse.ArgumentParser();
    for k,d,t in [('env_id','SafetyPointGoal1-v0',str),('seed',0,int),('total_steps',50000,int),('start_steps',1000,int),('update_after',1000,int),('batch_size',256,int),('eval_interval',1000,int),('eval_episodes',3,int),('log_dir',None,str)]: p.add_argument(f'--{k}',default=d,type=t,required=(k=='log_dir'))
    p.add_argument('--use_filter',action='store_true'); p.add_argument('--filter_type',default='hybrid'); p.add_argument('--terminate_on_safety_violation',action='store_true'); p.add_argument('--cost_limit_per_step',type=float,default=0.0)
    p.add_argument('--use_tn_energy',action='store_true'); p.add_argument('--use_projection_critic',action='store_true'); p.add_argument('--lambda_p',type=float,default=0.03); p.add_argument('--lambda_p_warmup_steps',type=int,default=0); p.add_argument('--entropy_reg_mode',default='flac_tn'); p.add_argument('--sample_k',type=int,default=256); p.add_argument('--diffusion_steps',type=int,default=10); p.add_argument('--num_ent_timesteps',type=int,default=10); p.add_argument('--policy_noise_scale',type=float,default=0.3); p.add_argument('--fixed_alpha',action='store_true'); p.add_argument('--alpha_value',type=float,default=0.1); p.add_argument('--init_alpha',type=float,default=0.1); p.add_argument('--lr',type=float,default=3e-4); p.add_argument('--alpha_lr',type=float,default=1e-2); p.add_argument('--gamma',type=float,default=0.99); p.add_argument('--gamma_p',type=float,default=0.99)
    args=p.parse_args(); np.random.seed(args.seed)
    log=Path(args.log_dir); (log/'checkpoints').mkdir(parents=True,exist_ok=True); writer=SummaryWriter(str(log/'tb')); (log/'args.json').write_text(json.dumps(vars(args),indent=2,sort_keys=True))
    env=SafeSafetyGymWrapper(env_id=args.env_id,use_filter=args.use_filter,filter_type=args.filter_type,terminate_on_safety_violation=args.terminate_on_safety_violation,cost_limit_per_step=args.cost_limit_per_step)
    eenv=SafeSafetyGymWrapper(env_id=args.env_id,use_filter=args.use_filter,filter_type=args.filter_type,terminate_on_safety_violation=args.terminate_on_safety_violation,cost_limit_per_step=args.cost_limit_per_step)
    obs,_=env.reset(seed=args.seed); agent=make_algo(args,env.observation_space.shape[0],env.action_space.shape[0]); key=jax.random.PRNGKey(args.seed+7); buf=[]; train=[]; ev=[]; best={'return_':(-1e18,'best_return_checkpoint.pkl',max),'cost_return':(1e18,'best_cost_checkpoint.pkl',min),'success_rate':(-1e18,'best_success_checkpoint.pkl',max),'safety_score':(1e18,'best_safety_checkpoint.pkl',min)}
    start=time.perf_counter(); pbar=tqdm(range(1,args.total_steps+1))
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
        if term or trunc: obs,_=env.reset()
        for k,v in info.items():
            if np.isscalar(v) and np.isfinite(float(v)): writer.add_scalar(f'train_env_info/{k}',float(v),step)
        if step>=args.update_after and len(buf)>=args.batch_size:
            out=agent.update(jax.random.PRNGKey(args.seed+step),sample_batch(buf,args.batch_size)); out['step']=step; train.append(out)
        if step%args.eval_interval==0:
            m=eval_agent(agent,args,eenv); m['step']=step; ev.append(m)
            with open(log/'checkpoints'/f'checkpoint_step_{step}.pkl','wb') as f: pickle.dump(agent.state,f)
            for mk,(bv,name,fn) in list(best.items()):
                if fn(mk in m and m[mk] or bv,bv)!=(bv):
                    pass
            if np.isfinite(m['return_']) and m['return_']>best['return_'][0]: best['return_']=(m['return_'],'best_return_checkpoint.pkl',max); pickle.dump(agent.state,open(log/'best_return_checkpoint.pkl','wb'))
            if np.isfinite(m['cost_return']) and m['cost_return']<best['cost_return'][0]: best['cost_return']=(m['cost_return'],'best_cost_checkpoint.pkl',min); pickle.dump(agent.state,open(log/'best_cost_checkpoint.pkl','wb'))
            if np.isfinite(m['success_rate']) and m['success_rate']>best['success_rate'][0]: best['success_rate']=(m['success_rate'],'best_success_checkpoint.pkl',max); pickle.dump(agent.state,open(log/'best_success_checkpoint.pkl','wb'))
            if np.isfinite(m['safety_score']) and m['safety_score']<best['safety_score'][0]: best['safety_score']=(m['safety_score'],'best_safety_checkpoint.pkl',min); pickle.dump(agent.state,open(log/'best_safety_checkpoint.pkl','wb'))
        pbar.set_postfix(r=f"{reward:.2f}",cost=f"{info.get('cost',0):.2f}",FAR=f"{info.get('filter_active',0):.2f}",APR=f"{info.get('projection_residual',0):.3f}",raw_norm=f"{info.get('raw_action_norm',0):.2f}",exec_norm=f"{info.get('exec_action_norm',0):.2f}",eval_ret=f"{(ev[-1]['return_'] if ev else np.nan):.1f}",eval_cost=f"{(ev[-1]['cost_return'] if ev else np.nan):.2f}",succ=f"{(ev[-1]['success_rate'] if ev else 0):.2f}",sps=f"{step/(time.perf_counter()-start):.1f}")
    pickle.dump(train,open(log/'train_metrics.pkl','wb')); pickle.dump(ev,open(log/'eval_metrics.pkl','wb')); pickle.dump(agent.state,open(log/'checkpoint.pkl','wb'))

import argparse, json, pickle, sys
from pathlib import Path
from typing import NamedTuple
import jax, jax.numpy as jnp, numpy as np
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
        normal_energy_coef=args.normal_energy_coef, weight_mix=args.weight_mix, residual_radius=args.residual_radius, action_limit=1.0)

def eval_agent(agent,args):
    mets=[]
    for ep in range(args.eval_episodes):
        env=SafeHumanoidBenchWrapper(args.env_name,use_filter=args.use_filter,filter_cfg=HumanoidSafeFilterConfig(residual_radius=args.residual_radius,smooth_radius=args.smooth_radius),policy_path=args.policy_path,mean_path=args.mean_path,var_path=args.var_path,policy_type=args.policy_type)
        obs,_=env.reset(seed=args.seed+1000+ep); done=False; ret=0; steps=0; falls=[]; far=[]; apr=[]; info={}
        while not done and steps<1000:
            a=env.action_space.sample() if agent is None else np.asarray(agent.get_action(jax.random.PRNGKey(args.seed+ep+steps), obs[None])[0])
            obs,r,term,trunc,info=env.step(a); done=term or trunc; ret+=r; steps+=1; falls.append(float(info.get('fall',0))); far.append(float(info.get('filter_active',0))); apr.append(float(info.get('projection_residual',0)))
        mets.append(dict(return_=ret,episode_length=steps,FAR=np.mean(far),APR=np.mean(apr),fall_rate=np.max(falls) if falls else 0.0,success_rate=float(info.get("is_success", 0.0)),hand_dist=float(info.get('hand_dist',np.nan)),target_dist=float(info.get('target_dist',np.nan))))
    keys=mets[0].keys(); return {k:float(np.nanmean([m[k] for m in mets])) for k in keys}

def main():
    p=argparse.ArgumentParser();
    for k,d,t in [('env_name','h1hand-reach-v0',str),('seed',0,int),('total_steps',1000000,int),('start_steps',10000,int),('update_after',10000,int),('batch_size',256,int),('eval_interval',10000,int),('eval_episodes',10,int),('log_dir',None,str)]: p.add_argument(f'--{k}',default=d,type=t,required=(k=='log_dir'))
    p.add_argument('--use_filter',action='store_true'); p.add_argument('--residual_radius',type=float,default=0.35); p.add_argument('--smooth_radius',type=float,default=0.25)
    p.add_argument('--lr',type=float,default=3e-4); p.add_argument('--alpha_lr',type=float,default=1e-2); p.add_argument('--gamma',type=float,default=0.99); p.add_argument('--gamma_p',type=float,default=0.99); p.add_argument('--sample_k',type=int,default=256); p.add_argument('--lambda_p',type=float,default=0.1); p.add_argument('--lambda_p_warmup_steps',type=int,default=100000); p.add_argument('--use_projection_critic',action='store_true')
    p.add_argument('--fixed_alpha',action='store_true'); p.add_argument('--alpha_value',type=float,default=0.1); p.add_argument('--init_alpha',type=float,default=0.1)
    p.add_argument('--diffusion_steps',type=int,default=10); p.add_argument('--num_ent_timesteps',type=int,default=10); p.add_argument('--policy_noise_scale',type=float,default=0.3)
    p.add_argument('--entropy_reg_mode',choices=['legacy','likelihood_tn','flac_tn'],default='legacy'); p.add_argument('--use_tn_energy',action='store_true'); p.add_argument('--candidate_temp',type=float,default=0.1); p.add_argument('--beta_normal_entropy',type=float,default=1.0); p.add_argument('--min_effective_entropy',type=float,default=-20.0); p.add_argument('--target_effective_entropy',type=float,default=1.0); p.add_argument('--normal_energy_coef',type=float,default=0.05); p.add_argument('--weight_mix',type=float,default=0.05)
    p.add_argument('--policy_path',type=str,default=None); p.add_argument('--mean_path',type=str,default=None); p.add_argument('--var_path',type=str,default=None); p.add_argument('--policy_type',type=str,default=None)
    args=p.parse_args(); np.random.seed(args.seed)
    log=Path(args.log_dir); log.mkdir(parents=True,exist_ok=True); writer=SummaryWriter(str(log/'tb')); (log/'args.json').write_text(json.dumps(vars(args),indent=2,sort_keys=True))
    env=SafeHumanoidBenchWrapper(args.env_name,use_filter=args.use_filter,filter_cfg=HumanoidSafeFilterConfig(residual_radius=args.residual_radius,smooth_radius=args.smooth_radius),policy_path=args.policy_path,mean_path=args.mean_path,var_path=args.var_path,policy_type=args.policy_type); obs,_=env.reset(seed=args.seed)
    print("=" * 80)
    print("HumanoidBench Safe-Pullback Training")
    print("env_name:", args.env_name)
    print("policy_type:", args.policy_type)
    print("policy_path:", args.policy_path)
    print("observation_space:", env.observation_space)
    print("action_space:", env.action_space)
    print("obs_dim:", env.observation_space.shape[0])
    print("act_dim:", env.action_space.shape[0])
    print("use_filter:", args.use_filter)
    print("residual_radius:", args.residual_radius)
    print("smooth_radius:", args.smooth_radius)
    print("entropy_reg_mode:", args.entropy_reg_mode)
    print("=" * 80)
    agent=make_algo(args, env.observation_space.shape[0], env.action_space.shape[0]); key=jax.random.PRNGKey(args.seed+7); buf=[]; train=[]; ev=[]
    for step in range(1,args.total_steps+1):
        if step<args.start_steps: raw=env.action_space.sample()
        else: key,ak=jax.random.split(key); raw=np.asarray(agent.get_action(ak,obs[None])[0])
        nobs,reward,term,trunc,info=env.step(raw); exp=SafePullbackExperience.create(obs,raw,info['exec_action'],reward,term,trunc,nobs,info); buf.append(exp); buf=buf[-1_000_000:]; obs=nobs
        if term or trunc: obs,_=env.reset()
        for k,tg in [('reward','train_env/reward'),('filter_active','train_env/filter_active'),('projection_residual','train_env/projection_residual'),('projection_cost','train_env/projection_cost'),('fall','train_env/fall'),('head_height','train_env/head_height'),('torso_upright','train_env/torso_upright'),('joint_angle_abs_max','train_env/joint_angle_abs_max'),('joint_vel_abs_max','train_env/joint_vel_abs_max')]:
            if k in info: writer.add_scalar(tg,float(info[k]),step)
        if step>=args.update_after and len(buf)>=args.batch_size:
            key,uk=jax.random.split(key); out=agent.update(uk,sample_batch(buf,args.batch_size)); out['step']=step; train.append(out)
            for k,v in out.items():
                if k!='step': writer.add_scalar(f'train/{k}',float(v),step)
        if step%args.eval_interval==0:
            e=eval_agent(agent,args); e['step']=step; ev.append(e); print(step,e)
            for k,v in e.items():
                if k!='step': writer.add_scalar(f'eval/{k}',float(v),step)
    pickle.dump(train,open(log/'train_metrics.pkl','wb')); pickle.dump(ev,open(log/'eval_metrics.pkl','wb')); pickle.dump(agent.state,open(log/'checkpoint.pkl','wb'))
if __name__=='__main__': main()

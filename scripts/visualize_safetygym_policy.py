import argparse, pickle, sys
from pathlib import Path
import jax, numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from envs.safety_gym_safe_wrapper import SafeSafetyGymWrapper
from scripts.train_safe_safetygym import make_algo

if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--env_id',default='SafetyPointGoal1-v0')
    p.add_argument('--checkpoint',required=True)
    p.add_argument('--episodes',type=int,default=3)
    p.add_argument('--render_mode',default='human')
    p.add_argument('--use_filter',action='store_true')
    p.add_argument('--filter_type',default='hybrid')
    args=p.parse_args()
    env=SafeSafetyGymWrapper(env_id=args.env_id,use_filter=args.use_filter,filter_type=args.filter_type,render_mode=args.render_mode)
    dummy=argparse.Namespace(seed=0,diffusion_steps=10,num_ent_timesteps=10,alpha_value=0.1,fixed_alpha=False,init_alpha=0.1,policy_noise_scale=0.3,gamma=0.99,gamma_p=0.99,lr=3e-4,alpha_lr=1e-2,sample_k=256,lambda_p=0.0,use_projection_critic=False,lambda_p_warmup_steps=0,use_tn_energy=False,entropy_reg_mode='legacy')
    agent=make_algo(dummy,env.observation_space.shape[0],env.action_space.shape[0])
    agent.state=pickle.load(open(args.checkpoint,'rb'))
    for ep in range(args.episodes):
        obs,_=env.reset(); done=False; ret=0; costs=[]; fars=[]; aprs=[]; sv=[]; info={}; l=0
        while not done and l<1000:
            a=np.asarray(agent.get_action(jax.random.PRNGKey(ep*10000+l),obs[None])[0])
            obs,r,t,tr,info=env.step(a); done=t or tr; l+=1; ret+=float(r); costs.append(float(info.get('cost',0))); fars.append(float(info.get('filter_active',0))); aprs.append(float(info.get('projection_residual',0))); sv.append(float(info.get('safety_violation',0)))
        succ=float(info.get('is_success', info.get('success', info.get('goal_met', info.get('task_success', 0.0)))))
        print(dict(return_=ret,cost_return=float(np.sum(costs)),episode_length=l,success=succ,FAR=float(np.mean(fars)) if fars else 0.0,APR=float(np.mean(aprs)) if aprs else 0.0,safety_violation=float(np.mean(sv)) if sv else 0.0))

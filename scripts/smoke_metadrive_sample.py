import argparse
import numpy as np
from envs.metadrive_safe_wrapper import SafeMetaDriveSampleWrapper

p=argparse.ArgumentParser()
p.add_argument('--env_name',default='FlatThreeLaneStraight')
p.add_argument('--filter_type',default='sample_vo')
p.add_argument('--steps',type=int,default=1000)
p.add_argument('--seed',type=int,default=0)
args=p.parse_args()

env=SafeMetaDriveSampleWrapper(env_name=args.env_name,use_filter=True,filter_type=args.filter_type)
obs,_=env.reset(seed=args.seed)
print('obs shape', np.shape(obs))
print('action space', env.action_space)
ret=0.0; episodes=[]; stats=[]
for t in range(args.steps):
    a=env.action_space.sample()
    obs,r,term,trunc,info=env.step(a)
    ret += float(r)
    stats.append(info)
    if term or trunc:
        episodes.append((ret,info)); ret=0.0; obs,_=env.reset()
print('info keys', sorted(list(stats[-1].keys())))
last=episodes[-1][1] if episodes else stats[-1]
print('episode return', episodes[-1][0] if episodes else ret)
for k in ['is_success','crash','out_of_road','cost','FAR','APR','num_valid_candidates','valid_candidate_ratio','min_pred_ttc','min_pred_h_vo','filter_time_ms']:
    print(k, float(last.get(k,0.0)))
print('no_safe_candidate_rate', float(np.mean([x.get('no_safe_candidate',0.0) for x in stats])))
print('fallback_rate', float(np.mean([x.get('fallback',0.0) for x in stats])))
env.close()

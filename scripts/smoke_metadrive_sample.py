from pathlib import Path
import sys
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path: sys.path.insert(0, str(PROJECT_ROOT))
import argparse
import numpy as np
from envs.metadrive_safe_wrapper import SafeMetaDriveSampleWrapper

p=argparse.ArgumentParser(); p.add_argument('--env_name',default='FlatThreeLaneStraight'); p.add_argument('--filter_type',default='sample_vo'); p.add_argument('--steps',type=int,default=1000); p.add_argument('--seed',type=int,default=0); args=p.parse_args()
env=SafeMetaDriveSampleWrapper(env_name=args.env_name,use_filter=True,filter_type=args.filter_type)
obs,_=env.reset(seed=args.seed)
ret=0.0; stats=[]
for _ in range(args.steps):
    obs,r,term,trunc,info=env.step(env.action_space.sample()); ret += float(r); stats.append(info)
    if term or trunc: obs,_=env.reset()
print('FAR', float(np.mean([x.get('filter_active',0.0) for x in stats])))
print('APR', float(np.mean([x.get('projection_residual',0.0) for x in stats])))
print('valid_candidate_ratio', float(np.mean([x.get('valid_candidate_ratio',0.0) for x in stats])))
print('no_safe_candidate_rate', float(np.mean([x.get('no_safe_candidate',0.0) for x in stats])))
print('fallback_rate', float(np.mean([x.get('fallback',0.0) for x in stats])))
print('min_pred_ttc_mean', float(np.nanmean([x.get('min_pred_ttc',np.nan) for x in stats])))
print('min_pred_h_vo_mean', float(np.nanmean([x.get('min_pred_h_vo',np.nan) for x in stats])))
print('filter_time_ms_mean', float(np.mean([x.get('filter_time_ms',0.0) for x in stats])))
env.close()

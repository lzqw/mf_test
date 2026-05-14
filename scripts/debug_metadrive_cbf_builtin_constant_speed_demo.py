from pathlib import Path
import sys,argparse,json,csv
import gymnasium as gym
import numpy as np
PROJECT_ROOT=Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path: sys.path.insert(0,str(PROJECT_ROOT))
import relax.env.drive.lane_change  # noqa
from relax.safety.metadrive_cbf_builtin_filter import CBFBuiltinSafetyFilter, CBFBuiltinFilterConfig

def args():
 p=argparse.ArgumentParser(); p.add_argument('--env_name',default='FlatThreeLaneStraight'); p.add_argument('--seed',type=int,default=0); p.add_argument('--use_filter',action='store_true'); p.add_argument('--filter_type',default='cbf_builtin'); p.add_argument('--max_steps',type=int,default=500); p.add_argument('--render',action=argparse.BooleanOptionalAction,default=False); p.add_argument('--constant_target_speed',type=float,default=3.0); p.add_argument('--constant_speed_kp',type=float,default=0.25); p.add_argument('--constant_raw_accel_max',type=float,default=0.45); p.add_argument('--builtin_policy_name',choices=['idm','lane_change','trajectory_idm','expert','fallback_lane'],default='idm'); p.add_argument('--horizon',type=int,default=10); p.add_argument('--dt',type=float,default=0.1); p.add_argument('--cbf_obstacle_radius',type=float,default=1.35); p.add_argument('--cbf_safe_distance',type=float,default=1.35); p.add_argument('--cbf_activation_distance',type=float,default=20.0); p.add_argument('--ttc_activation_threshold',type=float,default=5.0); p.add_argument('--min_closing_speed',type=float,default=0.05); p.add_argument('--max_steer_angle',type=float,default=0.25); p.add_argument('--max_dsteer',type=float,default=0.20); p.add_argument('--max_daccel',type=float,default=0.30); return p.parse_args()

def main():
 a=args(); env=gym.make(a.env_name,use_render=a.render); env.reset(seed=a.seed)
 f=CBFBuiltinSafetyFilter(CBFBuiltinFilterConfig(dt=a.dt,horizon=a.horizon,obstacle_radius=a.cbf_obstacle_radius,safe_distance=a.cbf_safe_distance,cbf_activation_distance=a.cbf_activation_distance,ttc_activation_threshold=a.ttc_activation_threshold,min_closing_speed=a.min_closing_speed,max_steer_angle=a.max_steer_angle,max_dsteer=a.max_dsteer,max_daccel=a.max_daccel,builtin_policy_name=a.builtin_policy_name)); f.reset(); prev=np.zeros(2,np.float32)
 rows=[]; cost_sum=0; min_d=1e9; intv=0; built_ok=0; resid=[]; tms=[]; passed=False
 for t in range(a.max_steps):
  speed=float(getattr(env.unwrapped.agent,'speed',0.0)); raw=np.array([0.0,np.clip(a.constant_speed_kp*(a.constant_target_speed-speed),0.0,a.constant_raw_accel_max)],dtype=np.float32)
  exe,fi=f.project(raw,env=env.unwrapped,prev_exec_action=prev) if a.use_filter else (raw,{'filter_active':0.0,'builtin_policy_success':0.0,'projection_residual':0.0,'filter_time_ms':0.0,'min_pred_dist':1e9})
  prev=exe.copy(); _,_,term,trunc,info=env.step(exe); 
  if a.render: env.render()
  intv+=int(fi.get('filter_active',0)>0.5); built_ok+=int(fi.get('builtin_policy_success',0)>0.5); resid.append(float(fi.get('projection_residual',0))); tms.append(float(fi.get('filter_time_ms',0))); min_d=min(min_d,float(fi.get('min_pred_dist',1e9))); cost_sum+=float(info.get('cost',0));
  rows.append({'step':t,'raw_steer':raw[0],'raw_accel':raw[1],'exec_steer':exe[0],'exec_accel':exe[1],**{k:fi.get(k) for k in ['cbf_safe','filter_active','builtin_policy_success','min_pred_dist','projection_residual','filter_time_ms']},'crash':info.get('crash',0),'cost':info.get('cost',0)})
  if getattr(env.unwrapped.agent,'position',[0,0])[0]>30: passed=True
  if term or trunc: break
 with open('trace.csv','w',newline='') as fp: w=csv.DictWriter(fp,fieldnames=rows[0].keys()); w.writeheader(); w.writerows(rows)
 summary={'success':float(not rows[-1]['crash']),'passed_obstacle':bool(passed),'crash':float(rows[-1]['crash']),'cost_sum':float(cost_sum),'min_distance':float(min_d),'intervention_rate':intv/max(1,len(rows)),'builtin_policy_success_rate':built_ok/max(1,len(rows)),'mean_projection_residual':float(np.mean(resid)),'mean_filter_time_ms':float(np.mean(tms))}
 with open('summary.json','w') as fp: json.dump(summary,fp,indent=2)
 print(summary)
if __name__=='__main__': main()

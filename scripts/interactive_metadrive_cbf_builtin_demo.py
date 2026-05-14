from pathlib import Path
import sys, argparse, os
import gymnasium as gym
import numpy as np
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path: sys.path.insert(0, str(PROJECT_ROOT))
import relax.env.drive.lane_change  # noqa
from relax.safety.metadrive_cbf_builtin_filter import CBFBuiltinSafetyFilter, CBFBuiltinFilterConfig

def parse_args():
 p=argparse.ArgumentParser(); p.add_argument('--env_name',default='FlatThreeLaneStraight'); p.add_argument('--seed',type=int,default=0); p.add_argument('--filter_type',default='cbf_builtin'); p.add_argument('--use_filter',action='store_true'); p.add_argument('--judge_only_cbf',action='store_true'); p.add_argument('--print_every',type=int,default=5); p.add_argument('--show_status_panel',action=argparse.BooleanOptionalAction,default=False)
 for n,d,t in [('horizon',10,int),('dt',0.1,float),('cbf_obstacle_radius',1.35,float),('cbf_safe_distance',1.35,float),('cbf_activation_distance',20.0,float),('ttc_activation_threshold',5.0,float),('min_closing_speed',0.05,float),('cbf_h_margin',0.0,float),('cbf_min_margin',0.0,float),('steer_limit',0.7,float),('throttle_limit',0.8,float),('brake_limit',-0.8,float),('max_steer_angle',0.25,float),('max_dsteer',0.20,float),('max_daccel',0.30,float),('builtin_action_scale',1.0,float),('blend_with_raw',0.0,float),('correction_min_throttle',0.0,float)]: p.add_argument(f'--{n}',type=t,default=d)
 p.add_argument('--enable_ttc_lateral_gate',action=argparse.BooleanOptionalAction,default=True); p.add_argument('--ttc_lateral_gate',type=float,default=1.8); p.add_argument('--ttc_front_min_longitudinal',type=float,default=0.0); p.add_argument('--ttc_front_max_longitudinal',type=float,default=30.0); p.add_argument('--enable_cbf_lateral_gate',action=argparse.BooleanOptionalAction,default=True); p.add_argument('--cbf_lateral_gate',type=float,default=2.0); p.add_argument('--cbf_front_min_longitudinal',type=float,default=-1.0); p.add_argument('--cbf_front_max_longitudinal',type=float,default=30.0); p.add_argument('--cbf_close_distance_margin',type=float,default=0.3); p.add_argument('--builtin_policy_name',choices=['idm','lane_change','trajectory_idm','expert','fallback_lane','fallback_obstacle_avoid'],default='idm'); p.add_argument('--require_approaching',action=argparse.BooleanOptionalAction,default=True); p.add_argument('--preserve_raw_accel_if_positive',action=argparse.BooleanOptionalAction,default=False)
 return p.parse_args()

def get_current_policy(env):
 base=env.unwrapped; aid=getattr(getattr(base,'agent',None),'id',None) or 'default_agent'; return base.engine.get_policy(aid)

def attach_filter_to_active_policy(env,filt,args):
 policy=get_current_policy(env)
 if getattr(policy,'_safe_filter_is_patched',False): return policy
 original_act=policy.act
 policy._safe_filter_is_patched=True; policy._safe_filter_original_act=original_act; policy._safe_filter_prev_exec_action=np.zeros(2,np.float32)
 policy._safe_filter_last_raw_action=np.zeros(2,np.float32); policy._safe_filter_last_exec_action=np.zeros(2,np.float32); policy._safe_filter_last_info={}; policy._safe_filter_act_call_count=0
 def filtered_act(*a,**kw):
  raw_action=np.asarray(original_act(*a,**kw),dtype=np.float32).reshape(2)
  if args.judge_only_cbf:
   _,filter_info=filt.project(raw_action,env=env.unwrapped,prev_exec_action=policy._safe_filter_prev_exec_action)
   exec_action=raw_action
   filter_info=dict(filter_info)
   filter_info['judge_only_cbf']=1.0
   filter_info['selected_candidate_type']='judge_only_raw'
   filter_info['filter_active']=0.0
   filter_info['exec_action']=raw_action.copy()
   filter_info['projection_residual']=0.0
  elif args.use_filter: exec_action,filter_info=filt.project(raw_action,env=env.unwrapped,prev_exec_action=policy._safe_filter_prev_exec_action)
  else: exec_action,filter_info=raw_action,{}
  policy._safe_filter_prev_exec_action=np.asarray(exec_action,dtype=np.float32).copy(); policy._safe_filter_last_raw_action=raw_action.copy(); policy._safe_filter_last_exec_action=policy._safe_filter_prev_exec_action.copy(); policy._safe_filter_last_info=dict(filter_info); policy._safe_filter_act_call_count+=1
  return exec_action
 policy.act=filtered_act; return policy

def main():
 a=parse_args(); cfg=CBFBuiltinFilterConfig(dt=a.dt,horizon=a.horizon,obstacle_radius=a.cbf_obstacle_radius,safe_distance=a.cbf_safe_distance,cbf_activation_distance=a.cbf_activation_distance,ttc_activation_threshold=a.ttc_activation_threshold,min_closing_speed=a.min_closing_speed,cbf_h_margin=a.cbf_h_margin,cbf_min_margin=a.cbf_min_margin,require_approaching=a.require_approaching,steer_limit=a.steer_limit,throttle_limit=a.throttle_limit,brake_limit=a.brake_limit,max_steer_angle=a.max_steer_angle,max_dsteer=a.max_dsteer,max_daccel=a.max_daccel,builtin_policy_name=a.builtin_policy_name,builtin_action_scale=a.builtin_action_scale,blend_with_raw=a.blend_with_raw,correction_min_throttle=a.correction_min_throttle,preserve_raw_accel_if_positive=a.preserve_raw_accel_if_positive,enable_ttc_lateral_gate=a.enable_ttc_lateral_gate,ttc_lateral_gate=a.ttc_lateral_gate,ttc_front_min_longitudinal=a.ttc_front_min_longitudinal,ttc_front_max_longitudinal=a.ttc_front_max_longitudinal,enable_cbf_lateral_gate=a.enable_cbf_lateral_gate,cbf_lateral_gate=a.cbf_lateral_gate,cbf_front_min_longitudinal=a.cbf_front_min_longitudinal,cbf_front_max_longitudinal=a.cbf_front_max_longitudinal,cbf_close_distance_margin=a.cbf_close_distance_margin)
 print({'enable_cbf_lateral_gate':a.enable_cbf_lateral_gate,'cbf_lateral_gate':a.cbf_lateral_gate,'cbf_front_min_longitudinal':a.cbf_front_min_longitudinal,'cbf_front_max_longitudinal':a.cbf_front_max_longitudinal,'cbf_close_distance_margin':a.cbf_close_distance_margin})
 f=CBFBuiltinSafetyFilter(cfg); env=gym.make(a.env_name,use_render=True,manual_control=True,controller='keyboard'); env.reset(seed=a.seed); f.reset(); policy=attach_filter_to_active_policy(env,f,a)
 while True:
  _,_,term,trunc,info=env.step([0.0,0.0]); env.render(); fi=getattr(policy,'_safe_filter_last_info',{}) or {}
  s=int(info.get('episode_length',0))
  if s%a.print_every==0:
   diagnostics={'raw_action':getattr(policy,'_safe_filter_last_raw_action',None),'exec_action':getattr(policy,'_safe_filter_last_exec_action',None),**{k:fi.get(k) for k in ['cbf_safe','cbf_active','cbf_violation','predicted_collision','min_pred_dist','min_pred_ttc','min_pred_h_cbf','min_pred_cbf','obstacle_distance','closing_speed','ttc_relevant','in_ttc_corridor','cbf_relevant','in_cbf_corridor','close_distance_relevant','obstacle_longitudinal','obstacle_lateral','ttc_lateral_gate','cbf_lateral_gate','selected_candidate_type','judge_only_cbf','filter_active','projection_residual','builtin_policy_name','builtin_policy_success','builtin_policy_status','builtin_policy_class','builtin_policy_exception']}}
   print(diagnostics)
   if a.show_status_panel:
    print('CBF STATUS | safe={cbf_safe} active={cbf_active} violation={cbf_violation} collision={predicted_collision} min_dist={min_pred_dist} min_ttc={min_pred_ttc} raw={raw_action} exec={exec_action} candidate={selected_candidate_type} judge_only={judge_only_cbf}'.format(**diagnostics))
  if term or trunc: env.reset(seed=a.seed); f.reset(); policy=attach_filter_to_active_policy(env,f,a)

if __name__=='__main__': main()

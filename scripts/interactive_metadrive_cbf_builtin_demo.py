from pathlib import Path
import sys, argparse, os
import gymnasium as gym
import numpy as np
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path: sys.path.insert(0, str(PROJECT_ROOT))
import relax.env.drive.lane_change  # noqa
from relax.safety.metadrive_cbf_builtin_filter import CBFBuiltinSafetyFilter, CBFBuiltinFilterConfig

def parse_args():
 p=argparse.ArgumentParser(); p.add_argument('--env_name',default='FlatThreeLaneStraight'); p.add_argument('--seed',type=int,default=0); p.add_argument('--filter_type',default='cbf_builtin'); p.add_argument('--use_filter',action='store_true'); p.add_argument('--print_every',type=int,default=5); p.add_argument('--show_status_panel',action=argparse.BooleanOptionalAction,default=False)
 for n,d,t in [('horizon',10,int),('dt',0.1,float),('cbf_obstacle_radius',1.35,float),('cbf_safe_distance',1.35,float),('cbf_activation_distance',20.0,float),('ttc_activation_threshold',5.0,float),('min_closing_speed',0.05,float),('cbf_h_margin',0.0,float),('cbf_min_margin',0.0,float),('steer_limit',0.7,float),('throttle_limit',0.8,float),('brake_limit',-0.8,float),('max_steer_angle',0.25,float),('max_dsteer',0.20,float),('max_daccel',0.30,float),('builtin_action_scale',1.0,float),('blend_with_raw',0.0,float),('correction_min_throttle',0.0,float)]: p.add_argument(f'--{n}',type=t,default=d)
 p.add_argument('--builtin_policy_name',choices=['idm','lane_change','trajectory_idm','expert','fallback_lane'],default='idm'); p.add_argument('--require_approaching',action=argparse.BooleanOptionalAction,default=True); p.add_argument('--preserve_raw_accel_if_positive',action=argparse.BooleanOptionalAction,default=False)
 return p.parse_args()

def main():
 a=parse_args(); cfg=CBFBuiltinFilterConfig(dt=a.dt,horizon=a.horizon,obstacle_radius=a.cbf_obstacle_radius,safe_distance=a.cbf_safe_distance,cbf_activation_distance=a.cbf_activation_distance,ttc_activation_threshold=a.ttc_activation_threshold,min_closing_speed=a.min_closing_speed,cbf_h_margin=a.cbf_h_margin,cbf_min_margin=a.cbf_min_margin,require_approaching=a.require_approaching,steer_limit=a.steer_limit,throttle_limit=a.throttle_limit,brake_limit=a.brake_limit,max_steer_angle=a.max_steer_angle,max_dsteer=a.max_dsteer,max_daccel=a.max_daccel,builtin_policy_name=a.builtin_policy_name,builtin_action_scale=a.builtin_action_scale,blend_with_raw=a.blend_with_raw,correction_min_throttle=a.correction_min_throttle,preserve_raw_accel_if_positive=a.preserve_raw_accel_if_positive)
 f=CBFBuiltinSafetyFilter(cfg); env=gym.make(a.env_name,use_render=True,manual_control=True,controller='keyboard'); env.reset(seed=a.seed); f.reset(); prev=np.zeros(2,np.float32)
 cv2=None
 if a.show_status_panel and os.environ.get('DISPLAY',''):
  try: import cv2
  except: pass
 while True:
  raw=np.array([0.,0.],dtype=np.float32); exec_action,fi=f.project(raw,env=env.unwrapped,prev_exec_action=prev) if a.use_filter else (raw,{})
  prev=exec_action.copy(); _,_,term,trunc,info=env.step(exec_action); env.render()
  s=int(info.get('episode_length',0))
  if s%a.print_every==0:
   print({k:fi.get(k) for k in ['raw_action','builtin_action','exec_action','cbf_safe','cbf_active','filter_active','selected_candidate_type','builtin_policy_name','builtin_policy_success','builtin_policy_status','min_pred_dist','min_pred_ttc','min_pred_h_cbf','min_pred_cbf','cbf_violation','predicted_collision','obstacle_distance','closing_speed','projection_residual']},'speed',getattr(env.unwrapped.agent,'speed',0),'crash',info.get('crash',0),'cost',info.get('cost',0))
  if cv2 is not None:
   panel=np.zeros((260,900,3),dtype=np.uint8); lines=[f"CBF {'SAFE' if fi.get('cbf_safe',1)>0.5 else 'UNSAFE'}",f"INTERVENTION {'ON' if fi.get('filter_active',0)>0.5 else 'OFF'}",f"raw={fi.get('raw_action',raw)} builtin={fi.get('builtin_action',raw)}",f"exec={fi.get('exec_action',exec_action)}",f"d={fi.get('min_pred_dist',0):.3f} ttc={fi.get('min_pred_ttc',0):.3f} h={fi.get('min_pred_h_cbf',0):.3f} cbf={fi.get('min_pred_cbf',0):.3f}"]
   y=30
   for ln in lines: cv2.putText(panel,ln,(10,y),cv2.FONT_HERSHEY_SIMPLEX,0.7,(255,255,255),1); y+=40
   cv2.imshow('cbf_builtin_status',panel); cv2.waitKey(1)
  if term or trunc: env.reset(seed=a.seed); f.reset(); prev*=0

if __name__=='__main__': main()

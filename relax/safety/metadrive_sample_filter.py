from dataclasses import dataclass
import time
import numpy as np


@dataclass
class SampleVehicleFilterConfig:
    action_limit: float = 1.0
    steer_limit: float = 0.7
    throttle_limit: float = 0.8
    brake_limit: float = -0.8
    max_dsteer: float = 0.12
    max_daccel: float = 0.20
    dt: float = 0.1
    horizon: int = 8
    v_max: float = 20.0
    k_steer: float = 0.35
    k_accel: float = 3.0
    num_local_samples: int = 64
    num_prev_samples: int = 32
    local_sigma_steer: float = 0.20
    local_sigma_accel: float = 0.25
    prev_sigma: float = 0.15
    num_obstacles: int = 4
    safe_radius: float = 4.0
    ego_radius: float = 1.5
    obs_radius: float = 1.5
    ttc_min: float = 1.5
    h_vo_margin: float = 0.2
    lane_margin_min: float = 0.3
    vo_activation_distance: float = 12.0
    ttc_activation_threshold: float = 3.0
    min_closing_speed: float = 0.5
    h_vo_tolerance: float = 0.05
    allow_far_vo_pass: bool = True
    w_raw: float = 1.0
    w_rate: float = 0.3
    w_lane: float = 0.2
    w_progress: float = 0.1
    w_margin: float = 0.05
    forbid_opposite_lane: bool = True
    allowed_lane_change: int = 1
    road_edge_margin: float = 0.25
    lane_corridor_margin: float = 0.30
    max_abs_lateral_from_start_lane: float = 5.8
    w_seq_smooth: float = 0.2
    w_longitudinal_progress: float = 0.5
    w_clearance: float = 0.1
    w_pass_obstacle: float = 1.0
    w_opposite_lane: float = 1000.0
    w_road_edge: float = 1000.0
    num_maneuver_samples: int = 32
    maneuver_steer_values: tuple = (-0.6, -0.4, -0.25, 0.25, 0.4, 0.6)
    maneuver_accel_values: tuple = (-0.2, 0.0, 0.3, 0.6)
    eps: float = 1e-6
    seed: int = 0


class SampleBasedVehicleSafetyFilter:
    def __init__(self, cfg): self.cfg = cfg; self.rng = np.random.default_rng(cfg.seed); self.prev_exec_action = np.zeros(2, dtype=np.float32)
    def reset(self): self.prev_exec_action = np.zeros(2, dtype=np.float32)
    def _box_rate(self, a, prev):
        a=np.asarray(a,dtype=np.float32).copy(); a[0]=np.clip(a[0],-self.cfg.steer_limit,self.cfg.steer_limit); a[1]=np.clip(a[1],self.cfg.brake_limit,self.cfg.throttle_limit)
        d=np.clip(a-prev,[-self.cfg.max_dsteer,-self.cfg.max_daccel],[self.cfg.max_dsteer,self.cfg.max_daccel]); return np.clip(prev+d,-1.0,1.0)
    def _extract_ego_state(self, env):
        try:
            u=getattr(env,"unwrapped",env); v=getattr(u,"vehicle",None) or getattr(u,"agent",None); pos=np.asarray(getattr(v,"position",[0.0,0.0]),dtype=np.float32)
            heading=float(getattr(v,"heading_theta",getattr(v,"heading",0.0))); speed=float(getattr(v,"speed",np.linalg.norm(getattr(v,"velocity",[0.0,0.0]))))
            vel=np.asarray(getattr(v,"velocity",[speed*np.cos(heading),speed*np.sin(heading)]),dtype=np.float32); return dict(x=float(pos[0]),y=float(pos[1]),heading=heading,speed=speed,v=vel,obj=v)
        except Exception: return dict(x=0.0,y=0.0,heading=0.0,speed=0.0,v=np.zeros(2,dtype=np.float32),obj=None)
    def _extract_obstacles(self, env):
        out=[]
        try:
            u=getattr(env,'unwrapped',env); ego=getattr(u,'vehicle',None) or getattr(u,'agent',None); cands=[]; pk=getattr(u,'_parked_obj',None)
            if pk is not None: cands.append(pk)
            tm=getattr(u,'traffic_manager',None)
            if tm is not None and hasattr(tm,'vehicles'): cands += list(getattr(tm,'vehicles',{}).values())
            seen=set()
            for v in cands:
                if v is None or v is ego: continue
                vid=str(getattr(v,'id',id(v)))
                if vid in seen: continue
                seen.add(vid); pos=np.asarray(getattr(v,'position',[0.0,0.0]),dtype=np.float32); heading=float(getattr(v,'heading_theta',getattr(v,'heading',0.0)))
                speed=float(getattr(v,'speed',np.linalg.norm(getattr(v,'velocity',[0.0,0.0])))); vel=np.asarray(getattr(v,'velocity',[speed*np.cos(heading),speed*np.sin(heading)]),dtype=np.float32)
                out.append(dict(x=float(pos[0]),y=float(pos[1]),heading=heading,speed=speed,v=vel,d=float(np.linalg.norm(pos-np.asarray(getattr(ego,'position',pos),dtype=np.float32)))))
            out=sorted(out,key=lambda x:x['d'])[: self.cfg.num_obstacles]
        except Exception: return []
        return out

    def _extract_road_corridor(self, env, ego_state):
        c = dict(lane_available=False, start_lane=None, start_lane_index=None, start_lateral=0.0, lane_width=3.7,
                 road_left_bound=np.inf, road_right_bound=-np.inf, allowed_lateral_min=-self.cfg.max_abs_lateral_from_start_lane,
                 allowed_lateral_max=self.cfg.max_abs_lateral_from_start_lane, forbidden_opposite_boundary=None,
                 heading_ref=np.array([np.cos(ego_state['heading']), np.sin(ego_state['heading'])], dtype=np.float32), start_y=ego_state['y'])
        try:
            u=getattr(env,'unwrapped',env); ego=getattr(u,'vehicle',None) or getattr(u,'agent',None); lane=getattr(ego,'lane',None) or getattr(getattr(ego,'navigation',None),'current_lane',None)
            c['start_lane']=lane
            if lane is not None:
                c['lane_available']=True; c['lane_width']=float(getattr(lane,'width',3.7) or 3.7)
                if hasattr(lane,'local_coordinates'):
                    l,lat = lane.local_coordinates(np.asarray(getattr(ego,'position',[ego_state['x'],ego_state['y']])))
                    c['start_lateral']=float(lat)
                    if hasattr(lane,'heading_theta'): c['heading_ref']=np.array([np.cos(float(lane.heading_theta(l))),np.sin(float(lane.heading_theta(l)))],dtype=np.float32)
            aw = self.cfg.allowed_lane_change*c['lane_width'] + 0.5*c['lane_width']
            c['allowed_lateral_min']=c['start_lateral'] - aw + self.cfg.road_edge_margin
            c['allowed_lateral_max']=c['start_lateral'] + aw - self.cfg.road_edge_margin
            c['road_left_bound']=c['allowed_lateral_max']; c['road_right_bound']=c['allowed_lateral_min']; c['forbidden_opposite_boundary']=c['allowed_lateral_max']
        except Exception:
            pass
        return c

    def _as_sequence(self, cand):
        H=self.cfg.horizon; c=np.asarray(cand,dtype=np.float32)
        if c.ndim==1: seq=np.repeat(c.reshape(1,2),H,axis=0)
        else: seq=c.copy()
        if seq.shape[0]<H: seq=np.concatenate([seq,np.repeat(seq[-1:].copy(),H-seq.shape[0],axis=0)],axis=0)
        return seq[:H]

    def _rate_limit_sequence(self, seq, prev):
        out=[]; p=np.asarray(prev,dtype=np.float32)
        for t in range(seq.shape[0]):
            a=self._box_rate(seq[t],p); out.append(a); p=a
        return np.asarray(out,dtype=np.float32)

    def project(self, raw_action, env=None, prev_exec_action=None):
        t0=time.perf_counter(); raw=np.asarray(raw_action,dtype=np.float32).reshape(2); prev=self.prev_exec_action if prev_exec_action is None else np.asarray(prev_exec_action,dtype=np.float32).reshape(2)
        ego=self._extract_ego_state(env); obstacles=self._extract_obstacles(env); corridor=self._extract_road_corridor(env,ego); H=self.cfg.horizon
        candidates=[]
        base=[raw,np.clip(raw,-1,1),prev,np.zeros(2),np.array([0.0,self.cfg.brake_limit]),np.array([raw[0],-0.4]),np.array([0.3,-0.4]),np.array([-0.3,-0.4])]
        for b in base: candidates.append(dict(type='single',is_maneuver=False,seq=self._as_sequence(b)))
        for m in [0.25,0.4,0.6]:
            for th in [0.0,0.3,0.6]:
                k=H//3
                left=np.vstack([np.tile([m,th],(k,1)),np.tile([0.0,th],(k,1)),np.tile([-0.5*m,th],(H-2*k,1))])
                right=np.vstack([np.tile([-m,th],(k,1)),np.tile([0.0,th],(k,1)),np.tile([0.5*m,th],(H-2*k,1))])
                candidates += [dict(type='left_avoid',is_maneuver=True,seq=left),dict(type='right_avoid',is_maneuver=True,seq=right)]
                candidates += [dict(type='left_pass_accel',is_maneuver=True,seq=np.vstack([np.tile([m,max(th,0.3)],(k,1)),np.tile([0.0,max(th,0.3)],(k,1)),np.tile([-0.4*m,max(th,0.0)],(H-2*k,1))])),dict(type='right_pass_accel',is_maneuver=True,seq=np.vstack([np.tile([-m,max(th,0.3)],(k,1)),np.tile([0.0,max(th,0.3)],(k,1)),np.tile([0.4*m,max(th,0.0)],(H-2*k,1))]))]
        for br in [-0.2,-0.4,-0.8]: candidates.append(dict(type='brake_keep_lane',is_maneuver=True,seq=np.tile([0.0,br],(H,1))))
        candidates += [dict(type='slow_steer_left',is_maneuver=True,seq=np.tile([0.25,0.0],(H,1))),dict(type='slow_steer_right',is_maneuver=True,seq=np.tile([-0.25,0.0],(H,1)))]
        k=H//3
        for _ in range(self.cfg.num_maneuver_samples):
            s1=float(self.rng.choice(self.cfg.maneuver_steer_values)); a1=float(self.rng.choice(self.cfg.maneuver_accel_values)); a2=float(self.rng.choice(self.cfg.maneuver_accel_values)); a3=float(self.rng.choice(self.cfg.maneuver_accel_values)); s3=float(self.rng.choice([0.0,-0.5*s1]))
            seq=np.vstack([np.tile([s1,a1],(k,1)),np.tile([self.rng.normal(0,0.08),a2],(k,1)),np.tile([s3,a3],(H-2*k,1))])
            candidates.append(dict(type='random_maneuver',is_maneuver=True,seq=seq))

        for c in candidates: c['seq']=self._rate_limit_sequence(self._as_sequence(c['seq']),prev)

        def eval_candidate(c):
            seq=c['seq']; speed,hdg,x,y=ego['speed'],ego['heading'],ego['x'],ego['y']; min_d,min_ttc,min_h=np.inf,np.inf,np.inf; min_lane_margin=np.inf; min_corridor_margin=np.inf; max_abs_lat=0.0
            predicted_offroad=False; predicted_opp=False; nearest_obs_lon=None
            if obstacles:
                obs0=min(obstacles,key=lambda o:o['d']); nearest_obs_lon=np.dot(np.array([obs0['x']-ego['x'],obs0['y']-ego['y']]),corridor['heading_ref'])
            vo_active_any=False; ttc_active_any=False; min_h_con=np.inf
            for t in range(H):
                a=seq[t]; hdg += self.cfg.k_steer*float(a[0])*self.cfg.dt; speed=np.clip(speed+self.cfg.k_accel*float(a[1])*self.cfg.dt,0.0,self.cfg.v_max)
                x += speed*np.cos(hdg)*self.cfg.dt; y += speed*np.sin(hdg)*self.cfg.dt; ego_v=np.array([speed*np.cos(hdg),speed*np.sin(hdg)])
                lat=(y-corridor['start_y'])+corridor['start_lateral']
                if corridor['lane_available'] and corridor['start_lane'] is not None and hasattr(corridor['start_lane'],'local_coordinates'):
                    try: lat=float(corridor['start_lane'].local_coordinates(np.array([x,y]))[1])
                    except Exception: pass
                lane_margin_t=0.5*corridor['lane_width']-abs(lat-corridor['start_lateral'])
                corr_margin_t=min(lat-corridor['allowed_lateral_min'],corridor['allowed_lateral_max']-lat)
                min_lane_margin=min(min_lane_margin,lane_margin_t); min_corridor_margin=min(min_corridor_margin,corr_margin_t); max_abs_lat=max(max_abs_lat,abs(lat-corridor['start_lateral']))
                predicted_offroad = predicted_offroad or (corr_margin_t < self.cfg.lane_corridor_margin)
                if self.cfg.forbid_opposite_lane: predicted_opp = predicted_opp or (lat>corridor['allowed_lateral_max'] or lat<corridor['allowed_lateral_min'])
                for ob in obstacles:
                    p_rel=np.array([ob['x']-x,ob['y']-y]); v_rel=ego_v-ob['v']; d=np.linalg.norm(p_rel); min_d=min(min_d,d); closing=float(np.dot(p_rel,v_rel)/(d+self.cfg.eps))
                    if closing>self.cfg.min_closing_speed:
                        ttc=(d-(self.cfg.ego_radius+self.cfg.obs_radius))/(closing+self.cfg.eps); min_ttc=min(min_ttc,ttc); ttc_active_any=ttc_active_any or (ttc<self.cfg.ttc_activation_threshold)
                        if d<self.cfg.vo_activation_distance:
                            vo_active_any=True; h=abs(p_rel[0]*v_rel[1]-p_rel[1]*v_rel[0])-(self.cfg.ego_radius+self.cfg.obs_radius)*np.linalg.norm(v_rel); min_h=min(min_h,h); min_h_con=min(min_h_con,h)
            lane_unsafe=predicted_offroad or predicted_opp or min_lane_margin<self.cfg.lane_margin_min or max_abs_lat>self.cfg.max_abs_lateral_from_start_lane
            collision=min_d<self.cfg.safe_radius; ttc_unsafe=ttc_active_any and (min_ttc<self.cfg.ttc_min); vo_unsafe=vo_active_any and (min_h_con<(self.cfg.h_vo_margin-self.cfg.h_vo_tolerance))
            valid=not(collision or ttc_unsafe or vo_unsafe or lane_unsafe or (self.cfg.forbid_opposite_lane and predicted_opp))
            end=np.array([x,y]); start=np.array([ego['x'],ego['y']]); longitudinal_progress=float(np.dot(end-start,corridor['heading_ref']))
            end_lon=longitudinal_progress; pass_bonus=float(nearest_obs_lon is not None and end_lon>nearest_obs_lon and min_d>=self.cfg.safe_radius)
            smooth=float(np.sum(np.square(np.diff(seq,axis=0)))) if H>1 else 0.0
            first=seq[0]
            score=self.cfg.w_raw*np.sum((first-raw)**2)+self.cfg.w_rate*np.sum((first-prev)**2)+self.cfg.w_seq_smooth*smooth+self.cfg.w_lane*abs(corridor['start_lateral'])-self.cfg.w_longitudinal_progress*longitudinal_progress-self.cfg.w_clearance*min_d-self.cfg.w_pass_obstacle*pass_bonus
            risk=(1000*float(collision)+1000*float(predicted_opp)+1000*float(predicted_offroad)+500*float(lane_unsafe)+50*max(0,self.cfg.ttc_min-(min_ttc if np.isfinite(min_ttc) else self.cfg.ttc_min))+20*max(0,self.cfg.h_vo_margin-(min_h_con if np.isfinite(min_h_con) else self.cfg.h_vo_margin))+10*max(0,self.cfg.safe_radius-min_d)-0.2*longitudinal_progress)
            return dict(valid=valid,score=score,risk=risk,seq=seq,first=first,last=seq[-1],type=c['type'],is_maneuver=c['is_maneuver'],predicted_opposite_lane=predicted_opp,predicted_offroad=predicted_offroad,lane_safe=not lane_unsafe,min_corridor_margin=min_corridor_margin,max_abs_lateral=max_abs_lat,min_d=min_d,min_ttc=min_ttc,min_h=min_h_con if vo_active_any else np.inf,lane_margin=min_lane_margin,collision=collision,ttc_unsafe=ttc_unsafe,vo_unsafe=vo_unsafe,lane_unsafe=lane_unsafe,longitudinal_progress=longitudinal_progress,pass_obstacle_bonus=pass_bonus)

        evals=[eval_candidate(c) for c in candidates]; valids=[e for e in evals if e['valid']]; fallback=len(valids)==0
        best=min(valids,key=lambda e:e['score']) if valids else min(evals,key=lambda e:e['risk'])
        exec_action=np.clip(best['first'],-1.0,1.0).astype(np.float32); self.prev_exec_action=exec_action.copy(); diff=exec_action-raw
        return exec_action, dict(raw_action=raw,exec_action=exec_action,projection_residual=float(np.linalg.norm(diff)),projection_cost=float(np.sum(diff**2)),filter_active=float(np.linalg.norm(diff)>1e-6),sample_filter_active=1.0,num_candidates=len(candidates),num_valid_candidates=len(valids),valid_candidate_ratio=float(len(valids)/max(len(candidates),1)),fallback=float(fallback),no_safe_candidate=float(fallback),least_risk_score=float(best['risk']),min_pred_dist=float(best['min_d']),min_pred_ttc=float(best['min_ttc']),min_pred_h_vo=float(best['min_h']),min_lane_margin=float(best['lane_margin']),predicted_collision=float(best['collision']),predicted_offroad=float(best['predicted_offroad']),predicted_opposite_lane=float(best['predicted_opposite_lane']),lane_safe=float(best['lane_safe']),min_corridor_margin=float(best['min_corridor_margin']),max_abs_lateral=float(best['max_abs_lateral']),longitudinal_progress=float(best['longitudinal_progress']),pass_obstacle_bonus=float(best['pass_obstacle_bonus']),selected_candidate_type=best['type'],selected_is_maneuver=float(best['is_maneuver']),chosen_sequence_first_action=np.asarray(best['first'],dtype=np.float32),chosen_sequence_last_action=np.asarray(best['last'],dtype=np.float32),filter_time_ms=float((time.perf_counter()-t0)*1000.0))

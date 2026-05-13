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
    w_end_lateral: float = 1.0
    w_max_lateral: float = 0.5
    w_heading_error: float = 0.3
    w_edge_buffer: float = 2.0
    edge_buffer: float = 0.8
    maneuver_commit_steps: int = 8
    maneuver_switch_penalty: float = 2.0
    same_maneuver_bonus: float = 1.0
    enable_lane_recovery: bool = True
    recovery_lateral_threshold: float = 0.7
    recovery_steer_gain: float = 0.25
    recovery_max_steer: float = 0.4
    enable_stuck_escape: bool = True
    stuck_speed_threshold: float = 0.2
    stuck_steps_threshold: int = 15
    stuck_escape_throttle: float = 0.35
    num_maneuver_samples: int = 32
    maneuver_steer_values: tuple = (-0.6, -0.4, -0.25, 0.25, 0.4, 0.6)
    maneuver_accel_values: tuple = (-0.2, 0.0, 0.3, 0.6)
    eps: float = 1e-6
    seed: int = 0


class SampleBasedVehicleSafetyFilter:
    def __init__(self, cfg):
        self.cfg = cfg
        self.rng = np.random.default_rng(cfg.seed)
        self.prev_exec_action = np.zeros(2, dtype=np.float32)
        self.active_maneuver_type = None
        self.active_maneuver_sign = 0
        self.active_maneuver_steps_left = 0
        self.low_speed_count = 0

    def reset(self):
        self.prev_exec_action = np.zeros(2, dtype=np.float32)
        self.active_maneuver_type = None
        self.active_maneuver_sign = 0
        self.active_maneuver_steps_left = 0
        self.low_speed_count = 0

    def _maneuver_sign(self, candidate_type, steer_hint=0.0):
        if candidate_type.startswith("left_"):
            return 1
        if candidate_type.startswith("right_"):
            return -1
        if candidate_type.startswith("lane_recovery_"):
            return int(np.sign(steer_hint))
        return 0

    def _box_rate(self, a, prev):
        a=np.asarray(a,dtype=np.float32).copy(); a[0]=np.clip(a[0],-self.cfg.steer_limit,self.cfg.steer_limit); a[1]=np.clip(a[1],self.cfg.brake_limit,self.cfg.throttle_limit)
        d=np.clip(a-prev,[-self.cfg.max_dsteer,-self.cfg.max_daccel],[self.cfg.max_dsteer,self.cfg.max_daccel]); return np.clip(prev+d,-1.0,1.0)
    # ... unchanged extract helpers ...
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
        c = dict(lane_available=False, start_lane=None, start_lateral=0.0, lane_width=3.7,
                 allowed_lateral_min=-self.cfg.max_abs_lateral_from_start_lane, allowed_lateral_max=self.cfg.max_abs_lateral_from_start_lane,
                 heading_ref=np.array([np.cos(ego_state['heading']), np.sin(ego_state['heading'])], dtype=np.float32), start_y=ego_state['y'])
        try:
            u=getattr(env,'unwrapped',env); ego=getattr(u,'vehicle',None) or getattr(u,'agent',None); lane=getattr(ego,'lane',None) or getattr(getattr(ego,'navigation',None),'current_lane',None)
            c['start_lane']=lane
            if lane is not None and hasattr(lane,'local_coordinates'):
                c['lane_available']=True; c['lane_width']=float(getattr(lane,'width',3.7) or 3.7)
                l,lat = lane.local_coordinates(np.asarray(getattr(ego,'position',[ego_state['x'],ego_state['y']]))); c['start_lateral']=float(lat)
                if hasattr(lane,'heading_theta'): c['heading_ref']=np.array([np.cos(float(lane.heading_theta(l))),np.sin(float(lane.heading_theta(l)))],dtype=np.float32)
            aw = self.cfg.allowed_lane_change*c['lane_width'] + 0.5*c['lane_width']
            c['allowed_lateral_min']=c['start_lateral'] - aw + self.cfg.road_edge_margin
            c['allowed_lateral_max']=c['start_lateral'] + aw - self.cfg.road_edge_margin
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
        self.low_speed_count = self.low_speed_count + 1 if ego['speed'] < self.cfg.stuck_speed_threshold else 0
        current_lateral = corridor['start_lateral']
        corridor_center = 0.5 * (corridor['allowed_lateral_min'] + corridor['allowed_lateral_max'])
        lateral_error_to_center = current_lateral - corridor_center
        steer_to_center = float(np.clip(-self.cfg.recovery_steer_gain * lateral_error_to_center, -self.cfg.recovery_max_steer, self.cfg.recovery_max_steer))

        candidates=[]
        for b in [raw,np.clip(raw,-1,1),prev,np.zeros(2),np.array([0.0,self.cfg.brake_limit]),np.array([raw[0],-0.4]),np.array([0.3,-0.4]),np.array([-0.3,-0.4])]:
            candidates.append(dict(type='single',is_maneuver=False,seq=self._as_sequence(b),steer_hint=float(b[0])))
        k=H//3
        for m in [0.25,0.4,0.6]:
            for th in [0.0,0.3,0.6]:
                left=np.vstack([np.tile([m,th],(k,1)),np.tile([0.0,th],(k,1)),np.tile([-0.5*m,th],(H-2*k,1))])
                right=np.vstack([np.tile([-m,th],(k,1)),np.tile([0.0,th],(k,1)),np.tile([0.5*m,th],(H-2*k,1))])
                candidates += [dict(type='left_avoid',is_maneuver=True,seq=left,steer_hint=m),dict(type='right_avoid',is_maneuver=True,seq=right,steer_hint=-m)]
        if self.cfg.enable_lane_recovery and abs(lateral_error_to_center) > self.cfg.recovery_lateral_threshold:
            for tname,accel in [('lane_recovery_brake',-0.2),('lane_recovery_hold',0.0),('lane_recovery_throttle',0.3)]:
                seq=np.vstack([np.tile([steer_to_center,accel],(k,1)),np.tile([0.6*steer_to_center,accel],(k,1)),np.tile([0.3*steer_to_center,accel],(H-2*k,1))])
                candidates.append(dict(type=tname,is_maneuver=True,seq=seq,steer_hint=steer_to_center))
        if self.cfg.enable_stuck_escape and self.low_speed_count >= self.cfg.stuck_steps_threshold:
            for tname,st in [('stuck_escape_straight',0.0),('stuck_escape_center',steer_to_center),('stuck_escape_left',0.25),('stuck_escape_right',-0.25)]:
                candidates.append(dict(type=tname,is_maneuver=True,seq=np.tile([st,self.cfg.stuck_escape_throttle],(H,1)),steer_hint=st))
        for c in candidates: c['seq']=self._rate_limit_sequence(self._as_sequence(c['seq']),prev)

        def wrap(a): return np.arctan2(np.sin(a),np.cos(a))
        def eval_candidate(c):
            seq=c['seq']; speed,hdg,x,y=ego['speed'],ego['heading'],ego['x'],ego['y']; min_d=np.inf; min_ttc=np.inf; min_h=np.inf; min_corridor_margin=np.inf; max_abs_lat=0.0; end_lateral=corridor['start_lateral']
            predicted_offroad=False; predicted_opp=False; nearest_obs_lon=None
            if obstacles: obs0=min(obstacles,key=lambda o:o['d']); nearest_obs_lon=np.dot(np.array([obs0['x']-ego['x'],obs0['y']-ego['y']]),corridor['heading_ref'])
            min_h_con=np.inf
            for a in seq:
                hdg += self.cfg.k_steer*float(a[0])*self.cfg.dt; speed=np.clip(speed+self.cfg.k_accel*float(a[1])*self.cfg.dt,0.0,self.cfg.v_max); x += speed*np.cos(hdg)*self.cfg.dt; y += speed*np.sin(hdg)*self.cfg.dt
                lat=(y-corridor['start_y'])+corridor['start_lateral']; end_lateral=lat
                corr_margin_t=min(lat-corridor['allowed_lateral_min'],corridor['allowed_lateral_max']-lat); min_corridor_margin=min(min_corridor_margin,corr_margin_t); max_abs_lat=max(max_abs_lat,abs(lat-corridor_center))
                predicted_offroad |= corr_margin_t < self.cfg.lane_corridor_margin; predicted_opp |= (lat>corridor['allowed_lateral_max'] or lat<corridor['allowed_lateral_min'])
                for ob in obstacles:
                    p_rel=np.array([ob['x']-x,ob['y']-y]); d=np.linalg.norm(p_rel); min_d=min(min_d,d)
            collision=min_d<self.cfg.safe_radius; lane_unsafe=predicted_offroad or predicted_opp or min_corridor_margin<self.cfg.lane_corridor_margin
            end=np.array([x,y]); start=np.array([ego['x'],ego['y']]); longitudinal_progress=float(np.dot(end-start,corridor['heading_ref'])); pass_bonus=float(nearest_obs_lon is not None and longitudinal_progress>nearest_obs_lon and min_d>=self.cfg.safe_radius)
            heading_ref_angle=float(np.arctan2(corridor['heading_ref'][1], corridor['heading_ref'][0])); heading_error=float(wrap(hdg-heading_ref_angle)); end_lateral_error_to_center=float(end_lateral-corridor_center); edge_penalty=float(max(0.0,self.cfg.edge_buffer-min_corridor_margin))
            score=self.cfg.w_raw*np.sum((seq[0]-raw)**2)+self.cfg.w_rate*np.sum((seq[0]-prev)**2)+self.cfg.w_seq_smooth*float(np.sum(np.square(np.diff(seq,axis=0))))+self.cfg.w_end_lateral*abs(end_lateral_error_to_center)+self.cfg.w_max_lateral*max_abs_lat+self.cfg.w_heading_error*abs(heading_error)+self.cfg.w_edge_buffer*edge_penalty-self.cfg.w_longitudinal_progress*longitudinal_progress-self.cfg.w_clearance*min_d-self.cfg.w_pass_obstacle*pass_bonus
            sign=self._maneuver_sign(c['type'],c.get('steer_hint',0.0))
            if self.active_maneuver_steps_left>0:
                if sign!=0 and sign==self.active_maneuver_sign: score -= self.cfg.same_maneuver_bonus
                elif sign!=0 and self.active_maneuver_sign!=0 and sign!=self.active_maneuver_sign: score += self.cfg.maneuver_switch_penalty
            risk=(1000*float(collision)+1000*float(predicted_opp)+1000*float(predicted_offroad)+100*edge_penalty+50*abs(end_lateral_error_to_center)+20*abs(heading_error)-0.2*longitudinal_progress)
            return dict(valid=not(collision or lane_unsafe),score=score,risk=risk,seq=seq,first=seq[0],last=seq[-1],type=c['type'],is_maneuver=c['is_maneuver'],predicted_opposite_lane=predicted_opp,predicted_offroad=predicted_offroad,min_corridor_margin=min_corridor_margin,max_abs_lateral=max_abs_lat,min_d=min_d,min_ttc=min_ttc,min_h=min_h_con if np.isfinite(min_h_con) else np.inf,lane_margin=min_corridor_margin,collision=collision,lane_unsafe=lane_unsafe,longitudinal_progress=longitudinal_progress,pass_obstacle_bonus=pass_bonus,end_lateral=end_lateral,end_lateral_error_to_center=end_lateral_error_to_center,heading_error=heading_error,edge_penalty=edge_penalty,candidate_sign=sign)

        raw_eval=eval_candidate(dict(type='raw',is_maneuver=False,seq=self._rate_limit_sequence(self._as_sequence(raw),prev),steer_hint=float(raw[0])))
        evals=[eval_candidate(c) for c in candidates]; valids=[e for e in evals if e['valid']]
        near_term_risk=raw_eval['collision'] or raw_eval['lane_unsafe'] or raw_eval['predicted_opposite_lane']
        if raw_eval['valid'] and not near_term_risk: best=raw_eval; fallback=False
        else:
            fallback=len(valids)==0
            if valids: best=min(valids,key=lambda e:e['score'])
            else:
                for e in evals:
                    if (e['predicted_offroad'] or e['min_corridor_margin'] < 0.0) and e['type'].startswith('lane_recovery_') and abs(e['end_lateral_error_to_center']) < abs(lateral_error_to_center):
                        e['risk'] -= 40.0
                best=min(evals,key=lambda e:e['risk'])
        maneuver_types={"left_avoid","right_avoid","left_pass_accel","right_pass_accel"}
        if best['type'] in maneuver_types:
            self.active_maneuver_type=best['type']; self.active_maneuver_sign=best['candidate_sign']; self.active_maneuver_steps_left=self.cfg.maneuver_commit_steps
        else:
            self.active_maneuver_steps_left=max(0,self.active_maneuver_steps_left-1)
        exec_action=np.clip(best['first'],-1.0,1.0).astype(np.float32); self.prev_exec_action=exec_action.copy(); diff=exec_action-raw
        return exec_action, dict(raw_action=raw,exec_action=exec_action,projection_residual=float(np.linalg.norm(diff)),projection_cost=float(np.sum(diff**2)),filter_active=float(np.linalg.norm(diff)>1e-6),sample_filter_active=1.0,num_candidates=len(candidates),num_valid_candidates=len(valids),valid_candidate_ratio=float(len(valids)/max(len(candidates),1)),fallback=float(fallback),no_safe_candidate=float(fallback),least_risk_score=float(best['risk']),min_pred_dist=float(best['min_d']),min_pred_ttc=float(best['min_ttc']),min_pred_h_vo=float(best['min_h']),min_lane_margin=float(best['lane_margin']),predicted_collision=float(best['collision']),predicted_offroad=float(best['predicted_offroad']),predicted_opposite_lane=float(best['predicted_opposite_lane']),lane_safe=float(not best['lane_unsafe']),min_corridor_margin=float(best['min_corridor_margin']),max_abs_lateral=float(best['max_abs_lateral']),longitudinal_progress=float(best['longitudinal_progress']),pass_obstacle_bonus=float(best['pass_obstacle_bonus']),selected_candidate_type=best['type'],selected_is_maneuver=float(best['is_maneuver']),chosen_sequence_first_action=np.asarray(best['first'],dtype=np.float32),chosen_sequence_last_action=np.asarray(best['last'],dtype=np.float32),end_lateral=float(best['end_lateral']),end_lateral_error_to_center=float(best['end_lateral_error_to_center']),heading_error=float(best['heading_error']),edge_penalty=float(best['edge_penalty']),lateral_error_to_center=float(lateral_error_to_center),active_maneuver_type=self.active_maneuver_type,active_maneuver_steps_left=int(self.active_maneuver_steps_left),low_speed_count=int(self.low_speed_count),filter_time_ms=float((time.perf_counter()-t0)*1000.0))

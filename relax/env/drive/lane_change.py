import logging, importlib, inspect
import sys
from typing import Tuple

from gymnasium.spaces.utils import FlatType
from metadrive.envs.safe_metadrive_env import SafeMetaDriveEnv
from metadrive.component.map.base_map import BaseMap
from metadrive.component.map.pg_map import MapGenerateMethod, FirstPGBlock

# ===== 可调参数（如需更远就把 REL 调大一些）=====
OBSTACLE_OFFSET_REL = 1.5       # 相对距离倍数：s = s0 + REL*L + M
OBSTACLE_OFFSET_M   = 0.0       # 绝对偏移（米，正向前、负向后）
SAFETY_MARGIN_END   = 2.0       # 距离段尾安全余量（米）
import gymnasium as gym
from gymnasium.wrappers import FlattenObservation
from gymnasium.envs.registration import register
# 注册你的新环境
sys.path.append(".")  # 确保当前目录在路径中，以便找到 lane_change 模块
register(
    id='FlatThreeLaneStraight',  # <-- 这是你将在训练脚本中使用的新 ID
    entry_point='relax.env.drive.lane_change:make_flat_metadrive_env',  # 指向你的创建函数
    max_episode_steps=1000  # 你可以根据需要调整这个值
)

def make_flat_metadrive_env(**kwargs):
    """
    这是一个辅助函数，用于创建 MetaDrive 环境
    并应用 FlattenObservation 包装器。
    """
    # 你可以在这里传递配置，例如禁用渲染以加快训练速度
    kwargs.pop("render_mode", None)
    config = {
        "use_render": False,
        "log_level": 50  # 设置为 ERROR，减少不必要的日志输出
    }
    config.update(kwargs)
    env = ThreeLaneStraightEnv(config)

    # 关键步骤：将 Dict 观测空间 扁平化为 Box 空间
    env = FlattenObservation(env)
    return env


def _pick_vehicle_class() -> Tuple[type, str]:
    """
    优先：TrafficVehicle -> IDMVehicle -> 其他非 DefaultVehicle 子类 -> DefaultVehicle(兜底)
    绝不返回抽象 BaseVehicle。
    """
    try:
        from metadrive.component.vehicle.vehicle_type import TrafficVehicle
        return TrafficVehicle, "TrafficVehicle"
    except Exception:
        pass
    try:
        from metadrive.component.vehicle.vehicle_type import IDMVehicle
        return IDMVehicle, "IDMVehicle"
    except Exception:
        pass
    # 其他子类（排除 DefaultVehicle 与抽象 BaseVehicle）
    try:
        vt = importlib.import_module("metadrive.component.vehicle.vehicle_type")
        from metadrive.component.vehicle.base_vehicle import BaseVehicle
        for name in dir(vt):
            obj = getattr(vt, name)
            if inspect.isclass(obj):
                try:
                    if issubclass(obj, BaseVehicle) and obj is not BaseVehicle and name.lower() != "defaultvehicle":
                        return obj, name
                except Exception:
                    continue
    except Exception:
        pass
    # 兜底
    from metadrive.component.vehicle.vehicle_type import DefaultVehicle
    return DefaultVehicle, "DefaultVehicle(FALLBACK)"


class ThreeLaneStraightEnv(SafeMetaDriveEnv):
    """固定单段直线 S + 三车道，中间车道前方有一辆静止大车。"""

    def default_config(self):
        cfg = super().default_config()

        # —— 固定为我们自定义的直线地图 ——
        cfg["map"] = None
        cfg["map_config"].update({
            BaseMap.GENERATE_TYPE:   MapGenerateMethod.BIG_BLOCK_SEQUENCE,
            BaseMap.GENERATE_CONFIG: "S",   # 只有一个直线区块
            BaseMap.LANE_NUM:        3,
            BaseMap.LANE_WIDTH:      3.7,
        })

        # —— 固定出生在 S 段中间车道起点附近 ——
        cfg["random_spawn_lane_index"] = False
        cfg["agent_configs"]["default_agent"].update({
            "spawn_lane_index": (FirstPGBlock.NODE_2, FirstPGBlock.NODE_3, 1),  # 中间车道
            "spawn_longitude":  1.0,   # s0：离段起点 1m
            "spawn_lateral":    0.0,
        })

        # 其他参数（如 manual_control/use_render 等）由外部传入 cfg 控制，这里不强制覆盖
        # 严格禁用除中线静止车外的任何额外交通/障碍
        cfg.update(dict(
            start_seed=0,
            log_level=logging.ERROR,
            num_scenarios=1,
            traffic_density=0.0,
            random_traffic=False,
            accident_prob=0.0,
            traffic_mode="NoTraffic",
        ))
        return cfg

    # 兼容 Gym / Gymnasium 的 reset（MetaDrive 0.4.3 无 options 参数）
    def reset(self, seed=None, options=None):
        # 在 reset 前仅清理我们手动生成的上轮“停靠车”，并强制 flush 清理队列
        seed=0
        try:
            eng = getattr(self, "engine", None)
            pk = getattr(self, "_parked_obj", None)
            if eng is not None and pk is not None:
                try:
                    eng.clear_objects([pk.id])
                except Exception:
                    pass
                try:
                    # 触发 managers.before_step() 使清理真正生效
                    if hasattr(eng, "before_step"):
                        eng.before_step({})
                except Exception:
                    pass
                try:
                    self._parked_obj = None
                except Exception:
                    pass
        except Exception:
            pass

        # 不传 options，以兼容 0.4.3 的 BaseEnv.reset 签名
        out = super().reset(seed=seed) if seed is not None else super().reset()
        obs, info = (out if (isinstance(out, tuple) and len(out) == 2) else (out, {}))

        # ===== 计算 S 段（NODE_2 -> NODE_3）中间车道上的目标点 =====
        lane_S = self.current_map.road_network.get_lane((FirstPGBlock.NODE_2, FirstPGBlock.NODE_3, 1))

        # 段总长度 L
        L = getattr(lane_S, "length", None)
        if L is None:
            L = lane_S.get_length() if hasattr(lane_S, "get_length") else 0.0
        L = float(L)

        # 出生点 s0
        s0 = float(self.config["agent_configs"]["default_agent"]["spawn_longitude"])

        # 目标位置：s0 + REL*L + M（并夹到 [0, L - SAFETY_MARGIN_END]）
        s_raw = s0 + OBSTACLE_OFFSET_REL * L + OBSTACLE_OFFSET_M
        s_target = max(0.0, min(s_raw, max(L - SAFETY_MARGIN_END, 0.0)))

        # 世界位姿（x,y,heading）
        x, y = lane_S.position(s_target, 0.0)
        heading = float(lane_S.heading_theta_at(s_target))

        # 车型（非 DefaultVehicle 优先）
        VehCls, vname = _pick_vehicle_class()

        # 0.4.3：vehicle_config 指定“车道 + 纵向位置”，并显式给 spawn_position_heading
        v_cfg = {
            "spawn_lane_index": (FirstPGBlock.NODE_2, FirstPGBlock.NODE_3, 1),
            "spawn_longitude":  float(s_target),
            "spawn_lateral":    0.0,

            "spawn_velocity":   [0.0, 0.0],             # 速度必须是二维向量
            "spawn_velocity_car_frame": True,
            "spawn_position_heading":   ([float(x), float(y)], heading),  # ([x,y], heading)

            "enable_reverse":   False,
            "max_speed_km_h":   0.0,
            "random_color":     False,

            # 尽量使用“大车”外观；若该模型不存在会被忽略
            "vehicle_model":    "bus",

            # 放大碰撞盒/外形
            "length": 12.0,
            "width":  2.5,
        }

        parked = self.engine.spawn_object(VehCls, vehicle_config=v_cfg)
        try:
            self._parked_obj = parked
        except Exception:
            pass

        # 不 set_static(True)，让其自然贴地，仅“刹死/去控制”
        try:
            if hasattr(parked, "set_break_down"): parked.set_break_down(True)
            if hasattr(parked, "policy"):         parked.policy = None
            if hasattr(parked, "controller"):     parked.controller = None
            if hasattr(parked, "enable_navigation"):
                parked.enable_navigation(False)
            if hasattr(parked, "set_angular_velocity"):
                parked.set_angular_velocity([0.0, 0.0, 0.0])
        except Exception:
            pass

        # 一次性调试输出
        if not hasattr(self, "_printed"):
            try:
                print(f"[ThreeLaneStraightEnv] L={L:.2f}, s0={s0:.2f}, raw={s_raw:.2f}, "
                      f"s={s_target:.2f}, veh_cls={vname}, pos=({x:.2f},{y:.2f}), head={heading:.3f}")
            except Exception:
                pass
            self._printed = True

        return obs, info

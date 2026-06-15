import numpy as np

import pytest

from envs.safe_obstacle_double_integrator_2d import SafeObstacleDoubleIntegrator2DEnv


def make_env():
    return SafeObstacleDoubleIntegrator2DEnv(
        noise_sigma=(0.0, 0.0),
        use_filter=True,
        seed=0,
        start_y_range=0.1,
        dt=0.1,
        a_max=3.0,
        v_max=2.0,
        damping=0.98,
        episode_len=50,
    )


def test_double_integrator_env_reset_and_obs():
    env = make_env()
    obs, info = env.reset(seed=123)

    assert obs.shape == (10,)
    assert "success" in info
    assert info["success"] == bool(info["success"])
    assert "clearance" in info and "h_sq" in info and "distance_to_goal" in info
    assert np.isfinite(obs).all()
    assert np.all(np.isfinite([info[k] for k in ["clearance", "h_sq", "distance_to_goal"]]))


def test_double_integrator_env_step_updates_state_and_info():
    env = make_env()
    env.reset(seed=1)
    env.state = np.array([0.0, 0.5, 0.0, 0.0], dtype=np.float32)

    raw_action = np.array([0.2, -0.2], dtype=np.float32)
    obs_next, reward, terminated, truncated, info = env.step(raw_action)

    assert obs_next.shape == (10,)
    assert isinstance(reward, float)
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)
    assert "exec_action" in info
    assert "filter_activated" in info
    assert "action_gain" in info
    assert np.isfinite(obs_next).all()


def test_double_integrator_env_filter_correction_near_obstacle():
    env = make_env()
    obs, _ = env.reset(seed=2)
    # place near obstacle but with feasible lateral correction.
    env.state = np.array([0.0, 0.875, 0.0, 0.0], dtype=np.float32)
    raw = np.array([1.0, 0.0], dtype=np.float32)
    _, _, _, _, info = env.step(raw)

    assert info["filter_activated"]
    # filter should reduce direct obstacle-forward residual in most cases
    assert info["projection_residual"] >= 0.0
    assert np.linalg.norm(info["exec_action"] - raw) > 0.0 or info["filter_fallback"]


def test_double_integrator_env_success_and_collision_flags():
    env = make_env()
    obs, _ = env.reset(seed=3)
    env.state = np.array([2.6, 0.0, 0.0, 0.0], dtype=np.float32)
    # next step at the goal should terminate by reward shaping and success metric
    _obs2, _r, terminated, truncated, info = env.step(np.zeros(2, dtype=np.float32))
    assert bool(info["success"]) or bool(terminated)
    assert isinstance(info["collision"], bool)
    assert info["distance_to_goal"] <= env.goal_radius + 1e-6

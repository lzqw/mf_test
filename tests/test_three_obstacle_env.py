import numpy as np

from envs.safe_obstacle_double_integrator_2d import SafeObstacleDoubleIntegrator2DEnv


def test_single_circle_still_constructs():
    env = SafeObstacleDoubleIntegrator2DEnv(noise_sigma=(0.0, 0.0), map_id="single_circle")
    obs, info = env.reset(seed=0)
    assert obs.shape == (10,)
    assert info["map_id"] == "single_circle"


def test_three_circles_constructs_with_16d_obs():
    env = SafeObstacleDoubleIntegrator2DEnv(
        noise_sigma=(0.0, 0.0),
        map_id="three_circles",
        reward_mode="multi_route_progress",
    )
    obs, info = env.reset(seed=0)
    assert obs.shape == (16,)
    assert info["map_id"] == "three_circles"
    assert info["clearances"].shape == (3,)


def test_three_circles_collision_detects_any_obstacle_and_min_clearance():
    env = SafeObstacleDoubleIntegrator2DEnv(
        noise_sigma=(0.0, 0.0),
        use_filter=False,
        map_id="three_circles",
        reward_mode="multi_route_progress",
    )
    env.reset(seed=1)
    env.state = np.array([0.70, 0.80, 0.0, 0.0], dtype=np.float32)
    _obs, _reward, terminated, truncated, info = env.step(np.zeros(2, dtype=np.float32))
    assert info["collision"]
    assert info["state_violation"]
    assert terminated
    assert not truncated
    assert info["nearest_obstacle_id"] in {1, 2}
    assert np.isclose(info["clearance"], np.min(info["clearances"]), atol=1e-6)


def test_three_circles_reset_records_start_y():
    env = SafeObstacleDoubleIntegrator2DEnv(
        noise_sigma=(0.0, 0.0),
        map_id="three_circles",
        reward_mode="multi_route_progress",
    )
    _obs, info = env.reset(seed=123)
    assert "start_y" in info
    assert np.isfinite(info["start_y"])


def test_three_circles_route_potential_respects_start_height_symmetry():
    env = SafeObstacleDoubleIntegrator2DEnv(
        noise_sigma=(0.0, 0.0),
        map_id="three_circles",
        reward_mode="multi_route_progress",
    )
    p_upper = np.array([-3.0, 0.8], dtype=np.float32)
    p_lower = np.array([-3.0, -0.8], dtype=np.float32)
    d_upper_from_upper = env._route_potential(p_upper, env.upper_route)
    d_lower_from_upper = env._route_potential(p_upper, env.lower_route)
    d_upper_from_lower = env._route_potential(p_lower, env.upper_route)
    d_lower_from_lower = env._route_potential(p_lower, env.lower_route)
    assert d_upper_from_upper < d_lower_from_upper
    assert d_lower_from_lower < d_upper_from_lower


def test_three_circles_start_conditioned_route_bias_changes_preferred_route():
    env = SafeObstacleDoubleIntegrator2DEnv(
        noise_sigma=(0.0, 0.0),
        map_id="three_circles",
        reward_mode="multi_route_progress",
        reward_cfg={"route_start_bias_scale": 0.5},
        start_y_range=1.0,
    )
    pos = np.array([-3.0, 0.0], dtype=np.float32)
    env.start_y = 0.8
    d_pos = env._multi_route_potential(pos)
    d_upper = env._route_potential(pos, env.upper_route) - 0.5 * np.tanh(0.8 / 1.0)
    d_lower = env._route_potential(pos, env.lower_route) + 0.5 * np.tanh(0.8 / 1.0)
    assert np.isclose(d_pos, min(d_upper, d_lower), atol=1e-6)
    assert d_upper < d_lower

    env.start_y = -0.8
    d_neg = env._multi_route_potential(pos)
    d_upper = env._route_potential(pos, env.upper_route) - 0.5 * np.tanh(-0.8 / 1.0)
    d_lower = env._route_potential(pos, env.lower_route) + 0.5 * np.tanh(-0.8 / 1.0)
    assert np.isclose(d_neg, min(d_upper, d_lower), atol=1e-6)
    assert d_lower < d_upper


def test_three_circles_route_variant_can_add_exit_pull_waypoint():
    env = SafeObstacleDoubleIntegrator2DEnv(
        noise_sigma=(0.0, 0.0),
        map_id="three_circles",
        route_variant="exit_pull_v1",
        reward_mode="multi_route_progress",
    )
    assert env.route_variant == "exit_pull_v1"
    assert len(env.upper_route) == 4
    assert len(env.lower_route) == 4
    assert env.upper_route[-1][0] > 2.0
    assert env.lower_route[-1][0] > 2.0


def test_three_circles_route_variant_exit_pull_v3_adds_goal_alignment_waypoint():
    env = SafeObstacleDoubleIntegrator2DEnv(
        noise_sigma=(0.0, 0.0),
        map_id="three_circles",
        route_variant="exit_pull_v3",
        reward_mode="multi_route_progress",
    )
    assert env.route_variant == "exit_pull_v3"
    assert len(env.upper_route) == 5
    assert len(env.lower_route) == 5
    assert env.upper_route[-1][0] > env.upper_route[-2][0]
    assert env.lower_route[-1][0] > env.lower_route[-2][0]
    assert env.upper_route[-1][1] < env.upper_route[-2][1]
    assert env.lower_route[-1][1] > env.lower_route[-2][1]


def test_three_circles_route_variant_exit_pull_v4_keeps_four_waypoints_but_lowers_exit():
    env = SafeObstacleDoubleIntegrator2DEnv(
        noise_sigma=(0.0, 0.0),
        map_id="three_circles",
        route_variant="exit_pull_v4",
        reward_mode="multi_route_progress",
    )
    assert env.route_variant == "exit_pull_v4"
    assert len(env.upper_route) == 4
    assert len(env.lower_route) == 4
    assert env.upper_route[-1][0] > 2.0
    assert env.lower_route[-1][0] > 2.0
    assert abs(float(env.upper_route[-1][1])) < 0.2
    assert abs(float(env.lower_route[-1][1])) < 0.2


def test_three_circles_goal_progress_mix_changes_multi_route_reward():
    env_plain = SafeObstacleDoubleIntegrator2DEnv(
        noise_sigma=(0.0, 0.0),
        map_id="three_circles",
        reward_mode="multi_route_progress",
        reward_cfg={"goal_progress_mix": 0.0},
    )
    env_mix = SafeObstacleDoubleIntegrator2DEnv(
        noise_sigma=(0.0, 0.0),
        map_id="three_circles",
        reward_mode="multi_route_progress",
        reward_cfg={"goal_progress_mix": 0.5},
    )
    state = np.array([-0.1, 1.1, 0.0, 0.0], dtype=np.float32)
    next_state = np.array([0.1, 1.0, 0.0, 0.0], dtype=np.float32)
    route_delta = env_plain._multi_route_potential(state[:2]) - env_plain._multi_route_potential(next_state[:2])
    goal_delta = float(np.linalg.norm(state[:2] - env_plain.goal) - np.linalg.norm(next_state[:2] - env_plain.goal))
    plain_reward, _ = env_plain._compute_reward(
        state, next_state, np.zeros(2, dtype=np.float32), np.zeros(2, dtype=np.float32), False, False, float(np.linalg.norm(state[:2] - env_plain.goal))
    )
    mix_reward, _ = env_mix._compute_reward(
        state, next_state, np.zeros(2, dtype=np.float32), np.zeros(2, dtype=np.float32), False, False, float(np.linalg.norm(state[:2] - env_mix.goal))
    )
    expected_gain = env_mix.reward_cfg["progress_coef"] * 0.5 * (goal_delta - route_delta)
    assert np.isclose(route_delta, env_plain._multi_route_potential(state[:2]) - env_plain._multi_route_potential(next_state[:2]), atol=1e-6)
    assert np.isclose(mix_reward - plain_reward, expected_gain, atol=1e-5)

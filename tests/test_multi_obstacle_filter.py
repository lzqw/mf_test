import numpy as np

from relax.safety.obstacle_double_integrator_filter import (
    DoubleIntegratorObstacleConfig,
    DoubleIntegratorObstacleFilter,
)


def make_three_obstacle_filter():
    cfg = DoubleIntegratorObstacleConfig(
        dt=0.08,
        a_max=3.5,
        obstacle_centers=np.array(
            [
                [-0.70, 0.00],
                [0.70, 0.80],
                [0.70, -0.80],
            ],
            dtype=np.float32,
        ),
        obstacle_radii=np.array([0.45, 0.42, 0.42], dtype=np.float32),
        eps_obs=0.06,
        x_min=-3.8,
        x_max=3.8,
        y_min=-2.4,
        y_max=2.4,
        grid_size=41,
    )
    return DoubleIntegratorObstacleFilter(cfg)


def test_multi_obstacle_filter_keeps_feasible_action():
    filt = make_three_obstacle_filter()
    state = np.array([-2.8, 1.2, 0.0, 0.0], dtype=np.float32)
    raw = np.array([0.2, -0.1], dtype=np.float32)
    projected, active, gap, safe_violation, safe_set_empty, fallback = filt.project_action_np(state, raw)
    assert not active
    assert gap == 0.0
    assert np.allclose(projected, raw)
    assert not safe_violation
    assert not safe_set_empty
    assert not fallback


def test_multi_obstacle_filter_projection_respects_all_obstacles():
    filt = make_three_obstacle_filter()
    state = np.array([0.25, 0.72, 1.0, 0.0], dtype=np.float32)
    raw = np.array([1.0, 0.0], dtype=np.float32)
    projected, active, gap, safe_violation, safe_set_empty, fallback, details = filt.project_action_np(
        state, raw, return_details=True
    )
    predicted_tight_clearances = filt.predicted_tight_clearances(state, projected)
    assert active
    assert safe_violation
    assert gap >= 0.0
    assert np.min(predicted_tight_clearances) >= -1e-6 or fallback
    assert details["nearest_obstacle_id"] in {0, 1, 2}


def test_multi_obstacle_filter_fallback_maximizes_min_clearance():
    cfg = DoubleIntegratorObstacleConfig(
        dt=0.1,
        a_max=3.0,
        obstacle_centers=np.array([[0.0, 0.0], [0.5, 0.0]], dtype=np.float32),
        obstacle_radii=np.array([10.0, 10.0], dtype=np.float32),
        eps_obs=0.08,
        grid_size=21,
    )
    filt = DoubleIntegratorObstacleFilter(cfg)
    state = np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32)
    raw = np.array([0.3, 0.4], dtype=np.float32)
    projected, active, gap, safe_violation, safe_set_empty, fallback, details = filt.project_action_np(
        state, raw, return_details=True
    )
    assert active
    assert safe_set_empty
    assert fallback
    assert np.isfinite(details["min_predicted_tight_clearance"])
    assert projected.shape == (2,)

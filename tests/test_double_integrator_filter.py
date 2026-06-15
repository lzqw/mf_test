import numpy as np

from relax.safety.obstacle_double_integrator_filter import (
    DoubleIntegratorObstacleConfig,
    DoubleIntegratorObstacleFilter,
)


def test_filter_far_action_kept():
    cfg = DoubleIntegratorObstacleConfig(dt=0.1, a_max=3.0)
    filt = DoubleIntegratorObstacleFilter(cfg)
    state = np.array([2.0, 1.0, 0.0, 0.0], dtype=np.float32)
    raw = np.array([0.5, -0.5], dtype=np.float32)
    projected, active, gap, safe_violation, safe_set_empty, fallback = filt.project_action_np(state, raw)

    assert not active
    assert gap == 0.0
    assert projected.shape == (2,)
    assert np.allclose(projected, raw)
    assert not safe_violation
    assert not safe_set_empty
    assert not fallback


def test_filter_corrects_toward_obstacle_action():
    cfg = DoubleIntegratorObstacleConfig(dt=0.1, a_max=3.0)
    filt = DoubleIntegratorObstacleFilter(cfg)
    # near obstacle in front, raw push towards obstacle; lateral correction should exist.
    state = np.array([0.0, 0.875, 0.0, 0.0], dtype=np.float32)
    raw = np.array([1.0, 0.0], dtype=np.float32)
    projected, active, gap, safe_violation, safe_set_empty, fallback = filt.project_action_np(state, raw)

    assert safe_violation
    assert active
    assert gap > 0.0
    # should be a different feasible action or explicit fallback
    assert np.linalg.norm(projected - raw) > 0.0
    assert not safe_set_empty


def test_filter_fallback_if_empty_set():
    # create overly tight safe set by placing action_gain extremely small? use huge obstacle radius so that no action works
    cfg = DoubleIntegratorObstacleConfig(dt=0.1, a_max=3.0, obstacle_radius=10.0)
    filt = DoubleIntegratorObstacleFilter(cfg)
    state = np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32)
    raw = np.array([0.5, 0.5], dtype=np.float32)
    projected, active, gap, safe_violation, safe_set_empty, fallback = filt.project_action_np(state, raw)

    assert safe_set_empty
    assert fallback
    assert active
    assert projected.shape == (2,)
    assert np.all(np.isfinite(projected))

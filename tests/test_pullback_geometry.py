import numpy as np

from relax.utils.pullback_geometry import (
    action_normal_double_integrator,
    empirical_covariance_local,
    local_curvature_covariances_double_integrator,
    local_normal_tangent,
    obstacle_margin_sq,
)


def test_normal_tangent_are_orthogonal():
    normal, tangent = local_normal_tangent(np.array([1.0, 1.0]), np.zeros(2))
    assert np.isclose(np.linalg.norm(normal), 1.0)
    assert np.isclose(np.linalg.norm(tangent), 1.0)
    assert np.isclose(float(normal @ tangent), 0.0, atol=1e-8)


def test_obstacle_margin_is_zero_on_circle():
    margin = obstacle_margin_sq(np.array([0.8, 0.0]), np.zeros(2), radius=0.8)
    assert np.isclose(margin, 0.0, atol=1e-8)


def test_action_normal_aligns_with_obstacle_normal():
    pos = np.array([1.2, 0.4])
    vel = np.array([0.1, -0.05])
    center = np.zeros(2)
    action_normal = action_normal_double_integrator(pos, vel, center, dt=0.1, a_max=3.0)
    obstacle_normal, _ = local_normal_tangent(pos, center)
    cosine = float(action_normal @ obstacle_normal) / max(np.linalg.norm(action_normal), 1e-12)
    assert cosine > 0.95


def test_curvature_covariances_reduce_normal_variance_and_shift_ratios():
    stats = local_curvature_covariances_double_integrator(
        pos=np.array([1.0, 0.3]),
        vel=np.array([0.2, 0.0]),
        center=np.zeros(2),
        dt=0.1,
        a_max=3.0,
        m0_scale=1.0,
        lambda_safe=80.0,
        lambda_robust=40.0,
        robust_iso=0.25,
    )

    assert stats["safety_normal_var"] < stats["nominal_normal_var"]
    assert stats["robust_normal_var"] < stats["safety_normal_var"]
    assert stats["safety_nlr"] < stats["nominal_nlr"]
    assert stats["robust_nlr"] < stats["safety_nlr"]
    assert stats["safety_tcr"] > stats["nominal_tcr"]
    assert stats["robust_tcr"] > stats["safety_tcr"]


def test_empirical_covariance_local_detects_tangent_dominance():
    actions = np.array(
        [
            [0.0, 1.0],
            [0.0, 0.8],
            [0.0, -0.8],
            [0.0, -1.0],
            [0.0, 0.6],
            [0.0, -0.6],
        ],
        dtype=np.float64,
    )
    cov_local, normal_var, tangent_var, nlr, tcr = empirical_covariance_local(
        actions,
        pos=np.array([1.0, 0.0]),
        center=np.zeros(2),
    )
    assert cov_local.shape == (2, 2)
    assert tangent_var > normal_var
    assert nlr < 0.5
    assert tcr > 0.5

import numpy as np

from relax.utils.pullback_geometry import (
    local_normal_tangent,
    multi_obstacle_curvature_covariances_double_integrator,
)


def test_multi_obstacle_curvature_is_spd_and_has_valid_frame():
    centers = np.array(
        [
            [-0.70, 0.00],
            [0.70, 0.80],
            [0.70, -0.80],
        ],
        dtype=np.float64,
    )
    radii = np.array([0.45, 0.42, 0.42], dtype=np.float64)
    pos = np.array([0.55, 0.95], dtype=np.float64)
    vel = np.array([0.2, -0.1], dtype=np.float64)

    stats = multi_obstacle_curvature_covariances_double_integrator(
        pos=pos,
        vel=vel,
        obstacle_centers=centers,
        obstacle_radii=radii,
        dt=0.08,
        a_max=3.5,
    )

    eigvals = np.linalg.eigvalsh(stats["cov_safety"])
    assert np.all(eigvals > 0.0)
    assert stats["nearest_obstacle_id"] in {0, 1, 2}
    assert np.isclose(np.linalg.norm(stats["normal"]), 1.0)
    assert np.isclose(np.linalg.norm(stats["tangent"]), 1.0)
    assert np.isclose(float(stats["normal"] @ stats["tangent"]), 0.0, atol=1e-8)


def test_multi_obstacle_safety_curvature_reduces_nearest_normal_variance():
    centers = np.array(
        [
            [-0.70, 0.00],
            [0.70, 0.80],
            [0.70, -0.80],
        ],
        dtype=np.float64,
    )
    radii = np.array([0.45, 0.42, 0.42], dtype=np.float64)
    pos = np.array([0.50, -0.92], dtype=np.float64)
    vel = np.array([0.1, 0.0], dtype=np.float64)

    stats = multi_obstacle_curvature_covariances_double_integrator(
        pos=pos,
        vel=vel,
        obstacle_centers=centers,
        obstacle_radii=radii,
        dt=0.08,
        a_max=3.5,
        lambda_scale=5.0,
        lambda_robust=30.0,
        robust_iso=0.25,
    )

    assert stats["safety_normal_var"] < stats["nominal_normal_var"]
    assert stats["robust_normal_var"] < stats["safety_normal_var"]


def test_nearest_obstacle_local_frame_matches_geometry():
    centers = np.array(
        [
            [-0.70, 0.00],
            [0.70, 0.80],
            [0.70, -0.80],
        ],
        dtype=np.float64,
    )
    radii = np.array([0.45, 0.42, 0.42], dtype=np.float64)
    pos = np.array([-0.55, 0.15], dtype=np.float64)
    vel = np.zeros(2, dtype=np.float64)
    stats = multi_obstacle_curvature_covariances_double_integrator(
        pos=pos,
        vel=vel,
        obstacle_centers=centers,
        obstacle_radii=radii,
        dt=0.08,
        a_max=3.5,
    )
    nearest = int(stats["nearest_obstacle_id"])
    normal_ref, tangent_ref = local_normal_tangent(pos, centers[nearest])
    assert np.allclose(stats["normal"], normal_ref, atol=1e-8)
    assert np.allclose(stats["tangent"], tangent_ref, atol=1e-8)

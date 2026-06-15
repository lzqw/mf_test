import sys
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from relax.utils.curvature import (  # noqa: E402
    covariance_from_curvature,
    covariance_variances,
    default_double_integrator,
    default_lqr_config,
    full_robust_curvature,
    nlr_tcr,
    nominal_curvature,
    pbar_tau,
    safety_curvature,
    solve_discounted_riccati,
    tangent_projector,
    action_normal_matrix,
    choose_tau,
)


def test_solve_discounted_riccati_correctness():
    cfg = default_lqr_config()
    A, B = default_double_integrator(dt=cfg["dt"])

    P, K = solve_discounted_riccati(A, B, cfg["Q"], cfg["R"], rho=cfg["rho"])

    assert np.allclose(P, P.T, atol=1e-8)
    assert np.min(np.linalg.eigvalsh(P)) > 1e-8

    S = cfg["R"] + cfg["rho"] * B.T @ P @ B
    K_expected = np.linalg.solve(S, cfg["rho"] * B.T @ P @ A)
    assert np.allclose(K, K_expected, atol=1e-8)

    rhs = cfg["Q"] + cfg["rho"] * (A.T @ P @ A) - cfg["rho"] * (A.T @ P @ B @ K)
    residual = P - rhs
    assert np.max(np.abs(residual)) < 1e-6


def test_tangent_projector_properties():
    cfg = default_lqr_config()
    _, B = default_double_integrator(dt=cfg["dt"])
    C_active = cfg["C"][[0], :]
    D = action_normal_matrix(B, C_active)

    Pi_T = tangent_projector(D)
    I = np.eye(2)

    assert np.allclose(Pi_T, Pi_T.T, atol=1e-9)
    assert np.allclose(Pi_T @ Pi_T, Pi_T, atol=1e-7)
    assert np.linalg.norm(D.T @ Pi_T) < 1e-8


def test_curvature_spd():
    cfg = default_lqr_config()
    A, B = default_double_integrator(dt=cfg["dt"])
    C_active = cfg["C"][[0], :]
    Lambda_b = np.array([[1.0]], dtype=np.float64)
    P, _ = solve_discounted_riccati(A, B, cfg["Q"], cfg["R"], rho=cfg["rho"])

    M0 = nominal_curvature(P, B, cfg["R"], cfg["rho"])
    Sigma0 = covariance_from_curvature(M0)
    M_safe = M0 + safety_curvature(B, C_active, Lambda_b)
    Sigma_safe = covariance_from_curvature(M_safe)

    E = B
    Phi_u = np.eye(2)
    W_P = E.T @ P @ E
    tau = choose_tau(P, E, cfg["rho"])
    M_robust, _, _ = full_robust_curvature(
        P,
        B,
        cfg["R"],
        C_active,
        Lambda_b,
        cfg["rho"],
        E,
        Phi_u,
        W_P,
        tau,
        eps=0.0,
    )
    Sigma_rob = covariance_from_curvature(M_robust)

    assert np.min(np.linalg.eigvalsh(M0)) > 0
    assert np.min(np.linalg.eigvalsh(M_safe)) > 0
    assert np.min(np.linalg.eigvalsh(M_robust)) > 0

    for Sigma in (Sigma0, Sigma_safe, Sigma_rob):
        assert np.all(np.isfinite(Sigma))
        assert np.min(np.linalg.eigvalsh(0.5 * (Sigma + Sigma.T))) > 0


def test_normal_variance_collapse():
    cfg = default_lqr_config()
    A, B = default_double_integrator(dt=cfg["dt"])
    C_active = cfg["C"][[0], :]
    P, _ = solve_discounted_riccati(A, B, cfg["Q"], cfg["R"], rho=cfg["rho"])
    M0 = nominal_curvature(P, B, cfg["R"], cfg["rho"])
    D = action_normal_matrix(B, C_active)

    normal_vars = []
    for lam in cfg["lambda_sweep"]:
        Lambda_b = np.array([[float(lam)]], dtype=np.float64)
        M_safe = M0 + safety_curvature(B, C_active, Lambda_b)
        Sigma_safe = covariance_from_curvature(M_safe)
        normal_var, _ = covariance_variances(Sigma_safe, D)
        normal_vars.append(normal_var)

    for i in range(len(normal_vars) - 1):
        assert normal_vars[i + 1] <= normal_vars[i] + 1e-8
    assert normal_vars[-1] < normal_vars[0]


def test_robust_tau_validity():
    cfg = default_lqr_config()
    A, B = default_double_integrator(dt=cfg["dt"])
    P, _ = solve_discounted_riccati(A, B, cfg["Q"], cfg["R"], rho=cfg["rho"])

    E = B
    rho = cfg["rho"]
    tau = choose_tau(P, E, rho)
    _, D_tau = pbar_tau(P, E, rho, tau)
    assert np.min(np.linalg.eigvalsh(D_tau)) > 0

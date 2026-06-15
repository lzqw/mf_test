import numpy as np


def state_action_normal(state, dt, a_max, obstacle_center=None):
    """Return unit normal direction d_unit in action space for a state.

    We use a simple geometric approximation with
    d = dt^2 * a_max * (p - p_obs), normalized in the workspace.
    """
    state = np.asarray(state, dtype=np.float64)
    if state.shape != (10,):
        raise ValueError("double-integrator observation must be shape (10,)")
    if obstacle_center is None:
        obstacle_center = np.array([0.0, 0.0], dtype=np.float64)
    else:
        obstacle_center = np.asarray(obstacle_center, dtype=np.float64)

    p = state[:2]
    clearance_vec = p - obstacle_center
    d = dt ** 2 * a_max * clearance_vec
    nrm = np.linalg.norm(d)
    if nrm <= 1e-12:
        d_unit = np.array([1.0, 0.0], dtype=np.float64)
    else:
        d_unit = d / nrm
    return d_unit


def action_curvature_matrix(
    state,
    dt,
    a_max,
    B,
    R=None,
    m0=1.0,
    lambda_max=1.0,
    lambda_clip=80.0,
    robust_iso=0.2,
    lambda_eps=0.08,
    obstacle_center=None,
):
    """Construct a local 2x2 obstacle-curvature matrix.

    M(x) = m0*I + lambda_safe(x) d_unit d_unit^T + robust_iso I,
    with
        lambda_safe(x) = clip(lambda_max / (clearance + lambda_eps)^2, 0, lambda_clip)
    """
    state = np.asarray(state, dtype=np.float64)
    if state.shape != (10,):
        raise ValueError("double-integrator observation must be shape (10,)")
    if obstacle_center is None:
        obstacle_center = np.array([0.0, 0.0], dtype=np.float64)
    else:
        obstacle_center = np.asarray(obstacle_center, dtype=np.float64)

    p = state[:2]
    clearance = float(np.linalg.norm(p - obstacle_center) - 0.8)
    lambda_safe = float(lambda_max / (clearance + float(lambda_eps)) ** 2)
    lambda_safe = float(np.clip(lambda_safe, 0.0, float(lambda_clip)))

    d_unit = state_action_normal(state, dt=dt, a_max=a_max, obstacle_center=obstacle_center)
    C_active = np.zeros((1, 4), dtype=np.float64)
    C_active[0, :2] = d_unit
    D = B.T @ C_active.T
    safety_term = lambda_safe * (D @ D.T) if D.size == 2 else 0.0 * np.eye(2)
    M = m0 * np.eye(2, dtype=np.float64) + robust_iso * np.eye(2, dtype=np.float64) + safety_term

    if R is not None:
        R = 0.5 * (np.asarray(R, dtype=np.float64) + np.asarray(R, dtype=np.float64).T)
        M = M + R
    M = 0.5 * (M + M.T)
    return M, lambda_safe, clearance, d_unit


def local_covariance_from_state(state, *args, eps=1e-4, alpha=1.0, **kwargs):
    M, lambda_safe, clearance, d_unit = action_curvature_matrix(state, *args, **kwargs)
    Sigma = alpha / 2.0 * np.linalg.solve(M + eps * np.eye(2, dtype=np.float64), np.eye(2, dtype=np.float64))
    return Sigma, M, lambda_safe, clearance, d_unit

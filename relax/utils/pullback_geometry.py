import numpy as np


def obstacle_margin_sq(pos, center, radius):
    """h(p)=||p-center||^2-radius^2."""
    pos = np.asarray(pos, dtype=np.float64)
    center = np.asarray(center, dtype=np.float64)
    radius = float(radius)
    delta = pos - center
    return float(delta @ delta - radius ** 2)


def obstacle_clearance(pos, center, radius):
    """clearance=||p-center||-radius."""
    pos = np.asarray(pos, dtype=np.float64)
    center = np.asarray(center, dtype=np.float64)
    radius = float(radius)
    return float(np.linalg.norm(pos - center) - radius)


def local_normal_tangent(pos, center, eps=1e-8):
    """
    n=(pos-center)/||pos-center||
    t=[-n_y,n_x]
    return n,t
    """
    pos = np.asarray(pos, dtype=np.float64)
    center = np.asarray(center, dtype=np.float64)
    delta = pos - center
    norm = float(np.linalg.norm(delta))
    if norm <= float(eps):
        normal = np.array([1.0, 0.0], dtype=np.float64)
    else:
        normal = delta / norm
    tangent = np.array([-normal[1], normal[0]], dtype=np.float64)
    return normal, tangent


def action_normal_double_integrator(pos, vel, center, dt, a_max, action=None, action_gain=1.0):
    """
    For p_next = p + dt*v + 0.5*dt^2*action_gain*a_max*u.
    g(u)=||p_next-center||^2-r^2.
    grad_u g = 2*(0.5*dt^2*action_gain*a_max)*(p_next-center).
    Return action-space normal d.
    """
    pos = np.asarray(pos, dtype=np.float64)
    vel = np.asarray(vel, dtype=np.float64)
    center = np.asarray(center, dtype=np.float64)
    if action is None:
        action = np.zeros(2, dtype=np.float64)
    action = np.asarray(action, dtype=np.float64)
    scale = 0.5 * float(dt) ** 2 * float(action_gain) * float(a_max)
    p_next = pos + float(dt) * vel + scale * action
    d = 2.0 * scale * (p_next - center)
    if np.linalg.norm(d) <= 1e-12:
        d = pos - center
    if np.linalg.norm(d) <= 1e-12:
        d = np.array([1.0, 0.0], dtype=np.float64)
    return np.asarray(d, dtype=np.float64)


def to_normal_tangent_frame(actions, normal, tangent):
    """
    actions shape [N,2].
    return local coordinates [normal_component,tangent_component].
    """
    actions = np.asarray(actions, dtype=np.float64)
    normal = np.asarray(normal, dtype=np.float64)
    tangent = np.asarray(tangent, dtype=np.float64)
    basis = np.stack([normal, tangent], axis=1)
    if actions.ndim == 1:
        return actions @ basis
    return actions @ basis


def _covariance_and_stats_from_local(local_actions):
    local_actions = np.asarray(local_actions, dtype=np.float64)
    if local_actions.ndim != 2 or local_actions.shape[1] != 2:
        raise ValueError("expected actions with shape [N, 2]")
    if local_actions.shape[0] <= 1:
        cov_local = np.eye(2, dtype=np.float64)
    else:
        cov_local = np.cov(local_actions.T, bias=False)
    cov_local = 0.5 * (cov_local + cov_local.T)
    normal_var = float(cov_local[0, 0])
    tangent_var = float(cov_local[1, 1])
    denom = max(normal_var + tangent_var, 1e-12)
    nlr = normal_var / denom
    tcr = tangent_var / denom
    return cov_local, normal_var, tangent_var, nlr, tcr


def empirical_covariance_local(actions, pos, center):
    """
    Return cov_local, normal_var, tangent_var, nlr, tcr.
    """
    normal, tangent = local_normal_tangent(pos, center)
    local_actions = to_normal_tangent_frame(actions, normal, tangent)
    cov_local, normal_var, tangent_var, nlr, tcr = _covariance_and_stats_from_local(local_actions)
    return cov_local, normal_var, tangent_var, nlr, tcr


def local_curvature_covariances_double_integrator(
    pos,
    vel,
    center,
    dt,
    a_max,
    alpha=1.0,
    m0_scale=1.0,
    lambda_safe=80.0,
    lambda_robust=40.0,
    robust_iso=0.25,
    eps=1e-5,
):
    """
    Construct local nominal/safety/robust covariance matrices.
    M0 = m0_scale * I
    d = action_normal_double_integrator(...)
    d_unit = d / ||d||
    M_safe = M0 + lambda_safe * d_unit d_unit^T
    M_robust = M_safe + lambda_robust * d_unit d_unit^T + robust_iso * I
    Sigma = alpha/2 * inv(M + eps I)
    Return dict with covariances and local frame basis.
    """
    pos = np.asarray(pos, dtype=np.float64)
    vel = np.asarray(vel, dtype=np.float64)
    center = np.asarray(center, dtype=np.float64)
    alpha = float(alpha)
    eps = float(eps)

    normal, tangent = local_normal_tangent(pos, center)
    d = action_normal_double_integrator(pos, vel, center, dt=dt, a_max=a_max)
    d_norm = float(np.linalg.norm(d))
    if d_norm <= 1e-12:
        d_unit = normal
    else:
        d_unit = d / d_norm

    ident = np.eye(2, dtype=np.float64)
    outer = np.outer(d_unit, d_unit)
    m0 = float(m0_scale) * ident
    m_safe = m0 + float(lambda_safe) * outer
    m_robust = m_safe + float(lambda_robust) * outer + float(robust_iso) * ident

    def _sigma_from_metric(metric):
        metric = 0.5 * (metric + metric.T)
        sigma = alpha / 2.0 * np.linalg.inv(metric + eps * ident)
        sigma = 0.5 * (sigma + sigma.T)
        return sigma

    sigma_nominal = _sigma_from_metric(m0)
    sigma_safety = _sigma_from_metric(m_safe)
    sigma_robust = _sigma_from_metric(m_robust)

    basis = np.stack([normal, tangent], axis=1)

    def _local_stats(sigma):
        sigma_local = basis.T @ sigma @ basis
        cov_local = 0.5 * (sigma_local + sigma_local.T)
        normal_var = float(cov_local[0, 0])
        tangent_var = float(cov_local[1, 1])
        denom = max(normal_var + tangent_var, 1e-12)
        nlr = normal_var / denom
        tcr = tangent_var / denom
        return cov_local, normal_var, tangent_var, nlr, tcr

    nom_local, nom_nv, nom_tv, nom_nlr, nom_tcr = _local_stats(sigma_nominal)
    safe_local, safe_nv, safe_tv, safe_nlr, safe_tcr = _local_stats(sigma_safety)
    robust_local, robust_nv, robust_tv, robust_nlr, robust_tcr = _local_stats(sigma_robust)

    return {
        "normal": normal,
        "tangent": tangent,
        "action_normal": d,
        "action_normal_unit": d_unit,
        "metric_nominal": m0,
        "metric_safety": m_safe,
        "metric_robust": m_robust,
        "cov_nominal": sigma_nominal,
        "cov_safety": sigma_safety,
        "cov_robust": sigma_robust,
        "cov_nominal_local": nom_local,
        "cov_safety_local": safe_local,
        "cov_robust_local": robust_local,
        "nominal_normal_var": nom_nv,
        "nominal_tangent_var": nom_tv,
        "nominal_nlr": nom_nlr,
        "nominal_tcr": nom_tcr,
        "safety_normal_var": safe_nv,
        "safety_tangent_var": safe_tv,
        "safety_nlr": safe_nlr,
        "safety_tcr": safe_tcr,
        "robust_normal_var": robust_nv,
        "robust_tangent_var": robust_tv,
        "robust_nlr": robust_nlr,
        "robust_tcr": robust_tcr,
    }


def multi_obstacle_curvature_covariances_double_integrator(
    pos,
    vel,
    obstacle_centers,
    obstacle_radii,
    dt,
    a_max,
    alpha=1.0,
    m0_scale=1.0,
    lambda_scale=4.0,
    lambda_eps=0.05,
    lambda_clip=120.0,
    lambda_robust=40.0,
    robust_iso=0.25,
    eps=1e-5,
):
    pos = np.asarray(pos, dtype=np.float64)
    vel = np.asarray(vel, dtype=np.float64)
    obstacle_centers = np.asarray(obstacle_centers, dtype=np.float64).reshape(-1, 2)
    obstacle_radii = np.asarray(obstacle_radii, dtype=np.float64).reshape(-1)
    if obstacle_centers.shape[0] != obstacle_radii.shape[0]:
        raise ValueError("obstacle_centers and obstacle_radii must match in length")

    deltas = pos[None, :] - obstacle_centers
    distances = np.linalg.norm(deltas, axis=1)
    clearances = distances - obstacle_radii
    nearest_obstacle_id = int(np.argmin(clearances))
    normal, tangent = local_normal_tangent(pos, obstacle_centers[nearest_obstacle_id])

    ident = np.eye(2, dtype=np.float64)
    m0 = float(m0_scale) * ident
    m_safe = m0.copy()
    robust_outer = np.zeros((2, 2), dtype=np.float64)
    action_normals = []
    lambdas = []

    for center, clearance in zip(obstacle_centers, clearances):
        d = action_normal_double_integrator(pos, vel, center, dt=dt, a_max=a_max)
        d_norm = float(np.linalg.norm(d))
        if d_norm <= 1e-12:
            d_unit = np.array([1.0, 0.0], dtype=np.float64)
        else:
            d_unit = d / d_norm
        lam = float(lambda_scale / max((clearance + lambda_eps) ** 2, 1e-8))
        lam = float(np.clip(lam, 0.0, lambda_clip))
        outer = np.outer(d_unit, d_unit)
        m_safe = m_safe + lam * outer
        robust_outer = robust_outer + outer
        action_normals.append(d_unit)
        lambdas.append(lam)

    m_robust = m_safe + float(lambda_robust) * robust_outer + float(robust_iso) * ident

    def _sigma_from_metric(metric):
        metric = 0.5 * (metric + metric.T)
        sigma = float(alpha) / 2.0 * np.linalg.inv(metric + float(eps) * ident)
        return 0.5 * (sigma + sigma.T)

    sigma_nominal = _sigma_from_metric(m0)
    sigma_safety = _sigma_from_metric(m_safe)
    sigma_robust = _sigma_from_metric(m_robust)

    basis = np.stack([normal, tangent], axis=1)

    def _local_stats(sigma):
        sigma_local = basis.T @ sigma @ basis
        cov_local = 0.5 * (sigma_local + sigma_local.T)
        normal_var = float(cov_local[0, 0])
        tangent_var = float(cov_local[1, 1])
        denom = max(normal_var + tangent_var, 1e-12)
        nlr = normal_var / denom
        tcr = tangent_var / denom
        return cov_local, normal_var, tangent_var, nlr, tcr

    nom_local, nom_nv, nom_tv, nom_nlr, nom_tcr = _local_stats(sigma_nominal)
    safe_local, safe_nv, safe_tv, safe_nlr, safe_tcr = _local_stats(sigma_safety)
    robust_local, robust_nv, robust_tv, robust_nlr, robust_tcr = _local_stats(sigma_robust)

    return {
        "nearest_obstacle_id": nearest_obstacle_id,
        "clearances": clearances,
        "lambdas": np.asarray(lambdas, dtype=np.float64),
        "normal": normal,
        "tangent": tangent,
        "action_normals": np.asarray(action_normals, dtype=np.float64),
        "metric_nominal": m0,
        "metric_safety": m_safe,
        "metric_robust": m_robust,
        "cov_nominal": sigma_nominal,
        "cov_safety": sigma_safety,
        "cov_robust": sigma_robust,
        "cov_nominal_local": nom_local,
        "cov_safety_local": safe_local,
        "cov_robust_local": robust_local,
        "nominal_normal_var": nom_nv,
        "nominal_tangent_var": nom_tv,
        "nominal_nlr": nom_nlr,
        "nominal_tcr": nom_tcr,
        "safety_normal_var": safe_nv,
        "safety_tangent_var": safe_tv,
        "safety_nlr": safe_nlr,
        "safety_tcr": safe_tcr,
        "robust_normal_var": robust_nv,
        "robust_tangent_var": robust_tv,
        "robust_nlr": robust_nlr,
        "robust_tcr": robust_tcr,
    }

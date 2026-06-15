import numpy as np


def default_double_integrator(dt: float = 0.5):
    A = np.array(
        [
            [1.0, dt, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, dt],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )

    dt2 = 0.5 * dt * dt
    B = np.array(
        [
            [dt2, 0.0],
            [dt, 0.0],
            [0.0, dt2],
            [0.0, dt],
        ],
        dtype=np.float64,
    )
    return A, B


def default_lqr_config():
    return {
        "dt": 0.5,
        "rho": 0.95,
        "alpha": 1.0,
        "eps": 1e-4,
        "Q": np.diag([1.0, 0.1, 1.0, 0.1]),
        "R": 0.1 * np.eye(2),
        "Sigma_w": 1e-4 * np.eye(4),
        "b": np.array([1.5, 1.5], dtype=np.float64),
        "C": np.array(
            [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
            ],
            dtype=np.float64,
        ),
        "lambda_sweep": [0, 1, 10, 100, 1000],
        "delta_sweep": [0, 0.1, 0.2, 0.3, 0.4],
        "boundary_states": [
            np.array([1.20, 0.0, 0.0, 0.0], dtype=np.float64),
            np.array([1.45, 0.0, 0.0, 0.0], dtype=np.float64),
            np.array([1.49, 0.0, 0.0, 0.0], dtype=np.float64),
        ],
    }


def solve_discounted_riccati(A, B, Q, R, rho=0.95, max_iter=10000, tol=1e-10):
    A = np.asarray(A, dtype=np.float64)
    B = np.asarray(B, dtype=np.float64)
    Q = 0.5 * (np.asarray(Q, dtype=np.float64) + np.asarray(Q, dtype=np.float64).T)
    R = 0.5 * (np.asarray(R, dtype=np.float64) + np.asarray(R, dtype=np.float64).T)

    n = A.shape[0]
    P = np.eye(n)
    I_m = np.eye(B.shape[1], dtype=np.float64)

    for _ in range(max_iter):
        S = R + rho * B.T @ P @ B
        if np.linalg.cond(S) > 1e18:
            raise RuntimeError("Discounted Riccati solve singular/ill-conditioned S_k matrix.")

        K = np.linalg.solve(S, rho * B.T @ P @ A)
        P_next = Q + rho * A.T @ P @ A - rho * A.T @ P @ B @ K
        P_next = 0.5 * (P_next + P_next.T)

        if np.linalg.norm(P_next - P, ord="fro") <= tol * (np.linalg.norm(P, ord="fro") + 1e-12):
            return P_next, K

        P = P_next

    raise RuntimeError("Discounted Riccati iteration did not converge.")


def action_normal_matrix(B, C_active):
    B = np.asarray(B, dtype=np.float64)
    C_active = np.asarray(C_active, dtype=np.float64)
    if C_active.ndim != 2:
        raise ValueError("C_active must be a 2-D array with shape [k, n].")
    return B.T @ C_active.T


def safety_curvature(B, C_active, Lambda_b):
    B = np.asarray(B, dtype=np.float64)
    C_active = np.asarray(C_active, dtype=np.float64)
    Lambda_b = np.asarray(Lambda_b, dtype=np.float64)
    D = C_active.T @ Lambda_b @ C_active
    return B.T @ D @ B


def tangent_projector(D):
    D = np.asarray(D, dtype=np.float64)
    if D.ndim == 1:
        D = D[:, None]
    if D.size == 0 or D.shape[1] == 0:
        return np.eye(D.shape[0], dtype=np.float64)

    gram = D.T @ D
    projector = D @ np.linalg.pinv(gram) @ D.T
    Pi_T = np.eye(D.shape[0], dtype=np.float64) - projector
    return 0.5 * (Pi_T + Pi_T.T)


def pbar_tau(P, E, rho, tau):
    P = 0.5 * (np.asarray(P, dtype=np.float64) + np.asarray(P, dtype=np.float64).T)
    E = np.asarray(E, dtype=np.float64)
    D_tau = tau * np.eye(E.shape[1], dtype=np.float64) - rho * E.T @ P @ E
    D_tau = 0.5 * (D_tau + D_tau.T)

    min_eig = np.min(np.linalg.eigvalsh(D_tau))
    if min_eig <= 1e-10:
        raise ValueError(
            f"D_tau is not positive definite: min eigenvalue = {min_eig:.6e}"
        )

    right = np.linalg.solve(D_tau, E.T @ P)
    Pbar = P + rho * P @ E @ right
    Pbar = 0.5 * (Pbar + Pbar.T)
    return Pbar, D_tau


def choose_tau(P, E, rho, margin=1.1, eps=1e-3):
    P = 0.5 * (np.asarray(P, dtype=np.float64) + np.asarray(P, dtype=np.float64).T)
    E = np.asarray(E, dtype=np.float64)
    lambda_max = np.max(np.linalg.eigvalsh(E.T @ P @ E))
    return margin * rho * lambda_max + eps


def nominal_curvature(P, B, R, rho):
    P = 0.5 * (np.asarray(P, dtype=np.float64) + np.asarray(P, dtype=np.float64).T)
    return np.asarray(R, dtype=np.float64) + rho * np.asarray(B, dtype=np.float64).T @ P @ np.asarray(B, dtype=np.float64)


def full_robust_curvature(
    P,
    B,
    R,
    C_active,
    Lambda_b,
    rho,
    E,
    Phi_u,
    W_P,
    tau,
    eps=0.0,
):
    P = 0.5 * (np.asarray(P, dtype=np.float64) + np.asarray(P, dtype=np.float64).T)
    B = np.asarray(B, dtype=np.float64)
    R = np.asarray(R, dtype=np.float64)
    C_active = np.asarray(C_active, dtype=np.float64)
    Lambda_b = np.asarray(Lambda_b, dtype=np.float64)
    E = np.asarray(E, dtype=np.float64)
    Phi_u = np.asarray(Phi_u, dtype=np.float64)
    W_P = np.asarray(W_P, dtype=np.float64)

    Pbar, D_tau = pbar_tau(P, E, rho, tau)
    m = B.shape[1]

    M = (
        R
        + B.T @ C_active.T @ Lambda_b @ C_active @ B
        + rho * B.T @ Pbar @ B
        + rho * Phi_u.T @ W_P @ Phi_u
        + tau * Phi_u.T @ Phi_u
        + eps * np.eye(m, dtype=np.float64)
    )
    M = 0.5 * (M + M.T)

    min_eig = np.min(np.linalg.eigvalsh(M))
    if min_eig <= 1e-10:
        raise ValueError(
            f"Curvature matrix M is not positive definite: min eigenvalue = {min_eig:.6e}"
        )

    return M, Pbar, D_tau


def covariance_from_curvature(M, alpha=1.0):
    M = np.asarray(M, dtype=np.float64)
    M = 0.5 * (M + M.T)
    min_eig = np.min(np.linalg.eigvalsh(M))
    if min_eig <= 1e-10:
        raise ValueError(
            f"Curvature matrix M is not positive definite: min eigenvalue = {min_eig:.6e}"
        )

    Sigma = alpha / 2.0 * np.linalg.solve(M, np.eye(M.shape[0], dtype=np.float64))
    Sigma = 0.5 * (Sigma + Sigma.T)
    return Sigma


def covariance_variances(Sigma, D):
    Sigma = 0.5 * (np.asarray(Sigma, dtype=np.float64) + np.asarray(Sigma, dtype=np.float64).T)
    D = np.asarray(D, dtype=np.float64)
    if Sigma.shape != (2, 2):
        raise ValueError("Sigma must be 2x2.")
    if D.ndim == 1:
        D = D[:, None]
    if D.size == 0 or D.shape[1] == 0:
        raise ValueError("D must contain at least one active normal.")
    if D.shape[0] != 2:
        raise ValueError("Action dimension must be 2.")

    d = D[:, 0]
    dnorm = np.linalg.norm(d)
    if dnorm <= 0.0:
        raise ValueError("First active normal has zero norm.")
    e_n = d / dnorm
    e_t = np.array([-e_n[1], e_n[0]], dtype=np.float64)

    normal_var = float(e_n.T @ Sigma @ e_n)
    tangent_var = float(e_t.T @ Sigma @ e_t)
    return normal_var, tangent_var


def nlr_tcr(Sigma, D, eps=1e-8):
    Sigma = 0.5 * (np.asarray(Sigma, dtype=np.float64) + np.asarray(Sigma, dtype=np.float64).T)
    D = np.asarray(D, dtype=np.float64)
    if D.ndim == 1:
        D = D[:, None]
    Pi_T = tangent_projector(D)
    I = np.eye(Sigma.shape[0], dtype=np.float64)
    N = I - Pi_T

    num_n = np.trace(N @ Sigma @ N.T)
    num_t = np.trace(Pi_T @ Sigma @ Pi_T.T)
    tr = np.trace(Sigma)
    if abs(tr) < eps:
        return 0.0, 0.0
    return float(num_n / (tr + eps)), float(num_t / (tr + eps))


def covariance_ellipse_points(mean, Sigma, nsig=2.0, num=200):
    mean = np.asarray(mean, dtype=np.float64)
    if mean.shape != (2,):
        raise ValueError("mean must be shape (2,).")
    Sigma = np.asarray(Sigma, dtype=np.float64)
    if Sigma.shape != (2, 2):
        raise ValueError("Sigma must be 2x2.")

    vals, vecs = np.linalg.eigh(Sigma)
    vals = np.clip(vals, 1e-18, None)
    angles = np.linspace(0.0, 2.0 * np.pi, num)
    unit_circle = np.column_stack([np.cos(angles), np.sin(angles)])
    scale = vecs @ np.diag(np.sqrt(vals))
    points = (nsig * unit_circle @ scale.T) + mean[None, :]
    return points

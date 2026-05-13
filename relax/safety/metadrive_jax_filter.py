from typing import Optional

import jax.numpy as jnp


def project_action_jax_metadrive(
    raw_action: jnp.ndarray,
    prior_action: Optional[jnp.ndarray] = None,
    steer_limit: float = 0.7,
    throttle_limit: float = 0.8,
    brake_limit: float = -0.8,
    max_dsteer: float = 0.12,
    max_daccel: float = 0.20,
    residual_radius: float = 0.35,
    action_limit: float = 1.0,
    eps: float = 1e-6,
):
    del max_dsteer, max_daccel
    a = jnp.asarray(raw_action)
    p = jnp.zeros_like(a) if prior_action is None else jnp.asarray(prior_action)

    steer = jnp.clip(a[..., 0], -steer_limit, steer_limit)
    accel = jnp.clip(a[..., 1], brake_limit, throttle_limit)
    clipped = jnp.stack([steer, accel], axis=-1)

    delta = clipped - p
    delta_norm = jnp.linalg.norm(delta, axis=-1, keepdims=True)
    scale = jnp.minimum(1.0, residual_radius / (delta_norm + eps))
    exec_action = p + delta * scale
    exec_action = jnp.clip(exec_action, -action_limit, action_limit)

    diff = exec_action - a
    projection_cost = jnp.sum(diff**2, axis=-1)
    projection_residual = jnp.linalg.norm(diff, axis=-1)
    return exec_action, projection_cost, projection_residual

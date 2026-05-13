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
):
    if prior_action is None:
        prior_action = jnp.zeros_like(raw_action)

    a = jnp.asarray(raw_action)
    p = jnp.asarray(prior_action)

    steer = jnp.clip(a[..., 0], -steer_limit, steer_limit)
    accel = jnp.clip(a[..., 1], brake_limit, throttle_limit)
    clipped = jnp.stack([steer, accel], axis=-1)

    d = clipped - p
    dsteer = jnp.clip(d[..., 0], -max_dsteer, max_dsteer)
    daccel = jnp.clip(d[..., 1], -max_daccel, max_daccel)
    exec_action = p + jnp.stack([dsteer, daccel], axis=-1)

    exec_action = jnp.stack(
        [
            jnp.clip(exec_action[..., 0], -1.0, 1.0),
            jnp.clip(exec_action[..., 1], -1.0, 1.0),
        ],
        axis=-1,
    )
    diff = exec_action - a
    projection_cost = jnp.sum(diff ** 2, axis=-1)
    projection_residual = jnp.linalg.norm(diff, axis=-1)
    return exec_action, projection_cost, projection_residual

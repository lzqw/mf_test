from typing import NamedTuple
import jax
import numpy as np


class SafePullbackExperience(NamedTuple):
    obs: jax.Array
    raw_action: jax.Array
    action: jax.Array
    reward: jax.Array
    done: jax.Array
    next_obs: jax.Array
    projection_residual: jax.Array
    projection_cost: jax.Array
    safe_violation: jax.Array
    filter_active: jax.Array
    state_violation: jax.Array
    is_success: jax.Array
    cost: jax.Array

    @staticmethod
    def create(obs, raw_action, exec_action, reward, terminated, truncated, next_obs, info):
        done = np.logical_or(terminated, truncated)
        projection_residual = info["projection_residual"]
        projection_cost = projection_residual ** 2
        return SafePullbackExperience(
            obs=obs,
            raw_action=info["raw_action"],
            action=info["exec_action"],
            reward=reward,
            done=done,
            next_obs=next_obs,
            projection_residual=projection_residual,
            projection_cost=projection_cost,
            safe_violation=info["safe_violation"],
            filter_active=info["filter_active"],
            state_violation=info["state_violation"],
            is_success=info.get("is_success", info.get("success", info.get("goal_met", info.get("task_success", 0.0)))),
            cost=info.get("cost", 0.0),
        )

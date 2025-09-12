from typing import Protocol, Tuple
from dataclasses import dataclass
from jax.lax import scan
import numpy as np
import jax, jax.numpy as jnp
import optax

class FlowModel(Protocol):
    def __call__(self, t: jax.Array, x: jax.Array) -> jax.Array:
        ...

class MeanFlowModel(Protocol):
    def __call__(self, x: jax.Array, r: jax.Array, t: jax.Array) -> jax.Array:
        ...

@dataclass(frozen=True)
class OTFlow:
    num_timesteps: int

    def p_sample(self, key: jax.Array, model: FlowModel, shape: Tuple[int, ...]) -> jax.Array:
        x = 0.5 * jax.random.normal(key, shape)
        #x = jax.random.normal(key, shape)
        dt = 1.0 / self.num_timesteps
        def body_fn(x, t):
            tau = t * dt
            drift = model(tau, x)
            x_next = x + drift * dt
            return x_next, None
        t_seq = jnp.arange(self.num_timesteps)
        x, _ = jax.lax.scan(body_fn, x, t_seq)
        return x

    def p_sample_traj(self, key: jax.Array, model: FlowModel, shape: Tuple[int, ...]) -> jax.Array:
        x = 0.5 * jax.random.normal(key, shape)
        dt = 1.0 / self.num_timesteps
        def body_fn(x, t):
            tau = t * dt
            drift = model(tau, x)
            x_next = x + drift * dt
            return x_next, x_next
        t_seq = jnp.arange(self.num_timesteps)
        _, x = jax.lax.scan(body_fn, x, t_seq)
        return x

    def p_sample_fast(self, model: FlowModel, shape: Tuple[int, ...]) -> jax.Array:
        x = jnp.zeros(shape)
        drift = model(0, x)
        return drift

    def q_sample(self, t: int, x_start: jax.Array, noise: jax.Array):
        return t * x_start + (1 - t) * noise

    def weighted_p_loss(self, key: jax.Array, weights: jax.Array, model: FlowModel, t: jax.Array,
                        x_start: jax.Array):
        if len(weights.shape) == 1:
            weights = weights.reshape(-1, 1)
        assert t.ndim == 1 and t.shape[0] == x_start.shape[0]
        noise = jax.random.normal(key, x_start.shape)
        # noise = 0.3 * jax.random.normal(key, x_start.shape)
        x_t = jax.vmap(self.q_sample)(t, x_start, noise)
        v_pred = model(t, x_t)
        loss = weights * optax.squared_error(v_pred, (x_start - noise))
        return loss.mean()

    def reverse_weighted_p_loss(self,  model: FlowModel, t: jax.Array,
                        x_t: jax.Array, u_estimation:jax.Array):
        t_squeezed = jnp.squeeze(t)
        v_pred = model(t_squeezed, x_t)
        loss = optax.squared_error(v_pred, u_estimation)
        return loss.mean()

    def weighted_p_loss_coupled(self, noise: jax.Array, weights: jax.Array, model: FlowModel, t: jax.Array,
                        x_start: jax.Array):
        if len(weights.shape) == 1:
            weights = weights.reshape(-1, 1)
        assert t.ndim == 1 and t.shape[0] == x_start.shape[0]
        x_t = jax.vmap(self.q_sample)(t, x_start, noise)
        v_pred = model(t, x_t)
        loss = weights * optax.squared_error(v_pred, (x_start - noise))
        return loss.mean()

    def recon_sample(self, t: jax.Array, x_t: jax.Array, noise: jax.Array):
        return (1 / t[:, jnp.newaxis]) * x_t - (1-t[:, jnp.newaxis])/t[:, jnp.newaxis] * noise

#t_final_unique,at,ut
    def recon_weighted_p_loss(self,  model, t_final_unique:jax.Array,at:jax.Array,ut:jax.Array):
        at=jnp.mean(at,axis=1)
        v_pred = model(t_final_unique, at)
        loss = optax.squared_error(v_pred, ut)
        return loss.mean()


@dataclass(frozen=True)
class MeanFlow:
    num_timesteps: int

    def p_sample(self, key: jax.Array, model: MeanFlowModel, shape: Tuple[int, ...]) -> jax.Array:
        x = 0.5 * jax.random.normal(key, shape)
        dt = 1.0 / self.num_timesteps

        def body_fn(x, t):
            tau = (self.num_timesteps - t) * dt
            # drift = model(x, tau, tau)
            drift = model(x, tau - dt, tau)
            x_next = x - drift
            # x_next = x - drift * dt
            return x_next, None

        t_seq = jnp.arange(self.num_timesteps)
        x, _ = jax.lax.scan(body_fn, x, t_seq)
        return x

    def p_sample_traj(self, key: jax.Array, model: MeanFlowModel, shape: Tuple[int, ...]) -> jax.Array:
        x = 0.5 * jax.random.normal(key, shape)
        dt = 1.0 / self.num_timesteps

        def body_fn(x, t):
            tau = t * dt
            drift = model(x, tau, tau + dt)
            x_next = x - drift
            return x_next, x_next

        t_seq = jnp.arange(self.num_timesteps)
        _, x = jax.lax.scan(body_fn, x, t_seq)
        return x

    def p_sample_fast(self, model: FlowModel, shape: Tuple[int, ...]) -> jax.Array:
        x = jnp.zeros(shape)
        drift = model(x, 0, 1)
        return -drift

    def q_sample(self, t: int, x_start: jax.Array, noise: jax.Array):
        return (1 - t) * x_start + t * noise

    def weighted_p_loss(self, key: jax.Array, weights: jax.Array, model: MeanFlowModel, r: jax.Array, t: jax.Array,
                        x_start: jax.Array):
        if len(weights.shape) == 1:
            weights = weights.reshape(-1, 1)
        assert r.ndim == 1 and t.ndim == 1 and t.shape[0] == x_start.shape[0]
        noise = jax.random.normal(key, x_start.shape)
        x_t = jax.vmap(self.q_sample)(t, x_start, noise)
        v = noise - x_start
        zero_r = jnp.zeros_like(r, dtype=jnp.float32)
        one_t = jnp.ones_like(t, dtype=jnp.float32)
        u_pred, dudt = jax.jvp(model, (x_t, r, t), (v, zero_r, one_t))
        u_tgt = jax.lax.stop_gradient(v - (t - r)[:, None] * dudt)
        loss = weights * optax.squared_error(u_pred, u_tgt)
        return loss.mean()

    # def reverse_weighted_p_loss(self, weights: jax.Array, model: MeanFlowModel, r: jax.Array, t: jax.Array,
    #                     x_start: jax.Array, noise: jax.Array,x_t: jax.Array,v_estimation:jax.Array):
    #     u = x_start - noise
    #     K = weights.shape[1]
    #
    #     x_t_expanded = jnp.repeat(jnp.expand_dims(x_t, axis=1), repeats=K, axis=1)
    #     r_expanded = jnp.repeat(jnp.expand_dims(r, axis=1), repeats=K, axis=1)
    #     t_expanded = jnp.repeat(jnp.expand_dims(t, axis=1), repeats=K, axis=1)
    #
    #     zero_r_tangent = jnp.zeros_like(r_expanded)  # Shape: (B, K, 1)
    #     one_t_tangent = jnp.ones_like(t_expanded)  # Shape: (B, K, 1)
    #
    #     u_pred, dudt = jax.jvp(
    #         model,
    #         (x_t_expanded, r_expanded, t_expanded),
    #         (u, zero_r_tangent, one_t_tangent)
    #     )
    #
    #     u_tgt = jax.lax.stop_gradient(u - (t - r)[:, None] * dudt)
    #     u_tgt_estimation = jax.lax.stop_gradient(jnp.sum(weights[:, :, None] * u_tgt, axis=1))
    #     loss = optax.squared_error(u_pred, u_tgt_estimation)
    #     return loss.mean()
    def reverse_weighted_p_loss(self, weights: jax.Array, model: MeanFlowModel, r: jax.Array, t: jax.Array,
                                x_start: jax.Array, noise: jax.Array, x_t: jax.Array):
        """
        计算加权的 flow-matching 损失 (内存优化版)。

        使用 jax.lax.scan 逐个处理 K 个样本，避免 OOM。
        """
        # u 是我们需要 jvp 计算其导数的方向向量
        u = x_start - noise  # Shape: (B, K, D)
        B, K, D = x_start.shape

        # --- 核心修改：使用 jax.lax.scan 代替展平操作 ---

        # 1. 准备 jax.lax.scan 的输入数据
        # scan 会沿着数组的第一个维度进行迭代。
        # 我们要迭代 K 个样本，所以需要将 u 的维度从 (B, K, D) 转置为 (K, B, D)
        u_transposed = jnp.transpose(u, (1, 0, 2))  # Shape: (K, B, D)

        # 准备 jvp 的不变参数 (primals 和部分 tangents)
        # 这些张量的批次大小都是 B，是内存友好的
        zero_tangent_r = jnp.zeros_like(r.squeeze(axis=-1))  # Shape: (B,)
        one_tangent_t = jnp.ones_like(t.squeeze(axis=-1))  # Shape: (B,)
        r_squeezed = r.squeeze(axis=-1)  # Shape: (B,)
        t_squeezed = t.squeeze(axis=-1)  # Shape: (B,)

        # 2. 定义 scan 的循环体函数 (scan_body)
        # 这个函数将在每个 K 上执行一次
        def scan_body(carry, u_k):
            # carry: 在本例中我们不需要在迭代间传递状态，所以忽略它
            # u_k: 这是 u_transposed 的一个切片，shape 为 (B, D)

            # 在单次循环内部，所有张量的批次大小都是 B
            u_pred_k, dudt_k = jax.jvp(
                model,
                # Primals: x_t, r, t 的批次大小都是 B
                (x_t, r_squeezed, t_squeezed),
                # Tangents: u_k 的批次大小是 B
                (u_k, zero_tangent_r, one_tangent_t)
            )
            # 返回 carry 和本次迭代的结果
            return carry, (u_pred_k, dudt_k)

        # 3. 执行 scan
        # scan 会自动为我们循环 K 次，并收集每次的结果
        # scan 的输入是 u_transposed，它有 K 个 (B, D) 的切片
        _, (u_pred_stacked, dudt_stacked) = scan(scan_body, None, u_transposed)

        # 4. 处理 scan 的输出
        # scan 的输出结果是堆叠起来的，shape 为 (K, B, D)
        # 我们需要将它们转置回我们习惯的 (B, K, D) 格式
        u_pred = jnp.transpose(u_pred_stacked, (1, 0, 2))  # Shape: (B, K, D)
        dudt = jnp.transpose(dudt_stacked, (1, 0, 2))  # Shape: (B, K, D)

        # --- 修改结束 ---

        # 5. 后续的损失计算逻辑和之前完全一样
        u_tgt = u - (t - r)[:, None] * dudt
        u_tgt = jax.lax.stop_gradient(u_tgt)

        u_tgt_estimation = jnp.sum(weights[:, :, None] * u_tgt, axis=1)

        squared_error = optax.squared_error(u_pred, jax.lax.stop_gradient(u_tgt_estimation))

        loss = jnp.mean(weights[:, :, None] * squared_error)
        return loss

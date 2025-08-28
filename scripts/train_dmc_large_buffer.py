import argparse
import os
from pathlib import Path
import time
from functools import partial
import yaml
import gc

import jax, jax.numpy as jnp
import numpy as np
from dm_env import specs

from relax.algorithm.sac_v import SAC_V
from relax.algorithm.rf_v import RF_V
from relax.algorithm.mf_v import MF_V
from relax.algorithm.mf2_v import MF2_V
from relax.algorithm.dpmd_v import DPMD_V
from relax.buffer.large import ReplayBufferStorage, ReplayBuffer
from relax.buffer.large import make_replay_loader
from relax.network.sac_v import create_sac_net_visual
from relax.network.rf_v import create_rf_net_visual
from relax.network.mf_v import create_mf_net_visual
from relax.network.mf2_v import create_mf2_net_visual
from relax.network.dpmd_v import create_dpmd_net_visual
from relax.trainer.off_policy_visual import OffPolicyTrainer
from relax.env import create_env, create_vector_env
from relax.utils.experience import Experience, ObsActionPair
from relax.utils.fs import PROJECT_ROOT
from relax.utils.random_utils import seeding
from relax.utils.log_diff import log_git_details

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--alg", type=str, default="rf_v")
    parser.add_argument("--env", type=str, default="Ant-v4")
    parser.add_argument("--suffix", type=str, default="test_use_atp1")
    parser.add_argument("--num_vec_envs", type=int, default=5)
    parser.add_argument("--hidden_num", type=int, default=3)
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--diffusion_steps", type=int, default=20)
    parser.add_argument("--diffusion_steps_test", type=int, default=20)
    parser.add_argument("--diffusion_hidden_dim", type=int, default=256)
    parser.add_argument("--start_step", type=int, default=int(3e4)) # other envs 3e4
    parser.add_argument("--total_step", type=int, default=int(5e5)) #1e6
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--lr_schedule_end", type=float, default=3e-5)
    parser.add_argument("--alpha_lr", type=float, default=7e-3)
    parser.add_argument("--delay_alpha_update", type=float, default=250)
    parser.add_argument("--seed", type=int, default=100)
    parser.add_argument("--num_particles", type=int, default=32)
    parser.add_argument("--noise_scale", type=float, default=0.1)
    parser.add_argument("--target_entropy_scale", type=float, default=1.5)
    parser.add_argument("--replay_buffer_size", type=int, default=int(1e6))
    parser.add_argument("--debug", action='store_true', default=False)
    parser.add_argument("--use_ema_policy", default=True, action="store_true")
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--n_step", type=int, default=1)
    parser.add_argument("--reward_scale", type=float, default=1.0)
    parser.add_argument("--sample_per_iteration", type=int, default=1)
    parser.add_argument("--update_per_iteration", type=int, default=1)
    parser.add_argument("--temperature", type=float, default=1.0)
    args = parser.parse_args()

    if args.debug:
        from jax import config
        config.update("jax_disable_jit", True)

    import torch.multiprocessing as mp
    mp.set_start_method('spawn')

    master_seed = args.seed
    master_rng, _ = seeding(master_seed)
    env_seed, env_action_seed, eval_env_seed, buffer_seed, init_network_seed, train_seed = map(
        int, master_rng.integers(0, 2**32 - 1, 6)
    )
    init_network_key = jax.random.key(init_network_seed)
    train_key = jax.random.key(train_seed)
    del init_network_seed, train_seed

    if args.num_vec_envs > 0:
        env, obs_dim, act_dim = create_vector_env(args.env, args.num_vec_envs, env_seed, env_action_seed, mode="futex")
    else:
        env, obs_dim, act_dim = create_env(args.env, env_seed, env_action_seed)
    latent_obs_dim = 32 * 35 * 35
    eval_env = None

    hidden_sizes = [args.hidden_dim] * args.hidden_num
    diffusion_hidden_sizes = [args.diffusion_hidden_dim] * args.hidden_num

    data_specs = (specs.Array((obs_dim,), np.uint8, 'observation'),
                  specs.Array((act_dim,), np.float32, 'action'),
                  specs.Array((1,), np.float32, 'reward'))
    exp_dir = PROJECT_ROOT / "logs" / args.env / (args.alg + '_' + time.strftime("%Y-%m-%d_%H-%M-%S") + f'_s{args.seed}_{args.suffix}')
    os.makedirs(exp_dir / 'buffer', exist_ok=True)
    replay_storage = ReplayBufferStorage(data_specs, exp_dir / 'buffer', args.num_vec_envs)
    buffer = make_replay_loader(
        exp_dir / 'buffer', args.replay_buffer_size,
        args.batch_size, num_workers=4, save_snapshot=False, nstep=args.n_step, discount=args.gamma)

    gelu = partial(jax.nn.gelu, approximate=False)

    if args.alg == 'rf_v':
        def mish(x: jax.Array):
            return x * jnp.tanh(jax.nn.softplus(x))
        agent, params = create_rf_net_visual(init_network_key, obs_dim, latent_obs_dim, act_dim, hidden_sizes, diffusion_hidden_sizes, mish,
                                          num_timesteps=args.diffusion_steps, 
                                          num_timesteps_test=args.diffusion_steps_test,
                                          num_particles=args.num_particles, 
                                          noise_scale=args.noise_scale,
                                          target_entropy_scale=args.target_entropy_scale)
        algorithm = RF_V(agent, params, gamma=args.gamma, lr=args.lr, alpha_lr=args.alpha_lr, 
                            delay_alpha_update=args.delay_alpha_update,
                            lr_schedule_end=args.lr_schedule_end,
                            use_ema=args.use_ema_policy,
                            reward_scale=args.reward_scale,
                            temperature=args.temperature,
                            total_step=args.total_step)
    elif args.alg == 'mf_v':
        def mish(x: jax.Array):
            return x * jnp.tanh(jax.nn.softplus(x))
        agent, params = create_mf_net_visual(init_network_key, obs_dim, latent_obs_dim, act_dim, hidden_sizes, diffusion_hidden_sizes, mish,
                                          num_timesteps=args.diffusion_steps, 
                                          num_timesteps_test=args.diffusion_steps_test,
                                          num_particles=args.num_particles, 
                                          noise_scale=args.noise_scale,
                                          target_entropy_scale=args.target_entropy_scale)
        algorithm = MF_V(agent, params, gamma=args.gamma, lr=args.lr, alpha_lr=args.alpha_lr, 
                           delay_alpha_update=args.delay_alpha_update,
                             lr_schedule_end=args.lr_schedule_end,
                             use_ema=args.use_ema_policy,
                             reward_scale=args.reward_scale,
                             temperature=args.temperature)
    elif args.alg == 'mf2_v':
        def mish(x: jax.Array):
            return x * jnp.tanh(jax.nn.softplus(x))
        agent, params = create_mf2_net_visual(init_network_key, obs_dim, latent_obs_dim, act_dim, hidden_sizes, diffusion_hidden_sizes, mish,
                                          num_timesteps=args.diffusion_steps, 
                                          num_timesteps_test=args.diffusion_steps_test,
                                          num_particles=args.num_particles, 
                                          noise_scale=args.noise_scale,
                                          target_entropy_scale=args.target_entropy_scale)
        algorithm = MF2_V(agent, params, gamma=args.gamma, lr=args.lr, alpha_lr=args.alpha_lr, 
                           delay_alpha_update=args.delay_alpha_update,
                             lr_schedule_end=args.lr_schedule_end,
                             use_ema=args.use_ema_policy,
                             reward_scale=args.reward_scale,
                             )
    elif args.alg == "sac_v":
        agent, params = create_sac_net_visual(init_network_key, obs_dim, latent_obs_dim, act_dim, hidden_sizes, gelu)
        algorithm = SAC_V(agent, params, lr=args.lr)
    elif args.alg == 'dpmd_v':
        def mish(x: jax.Array):
            return x * jnp.tanh(jax.nn.softplus(x))
        agent, params = create_dpmd_net_visual(init_network_key, obs_dim, latent_obs_dim, act_dim, hidden_sizes, diffusion_hidden_sizes, mish,
                                          num_timesteps=args.diffusion_steps, 
                                          num_particles=args.num_particles, 
                                          noise_scale=args.noise_scale,
                                          target_entropy_scale=args.target_entropy_scale)
        algorithm = DPMD_V(agent, params, gamma=args.gamma, lr=args.lr, alpha_lr=args.alpha_lr, 
                           delay_alpha_update=args.delay_alpha_update,
                             lr_schedule_end=args.lr_schedule_end,
                             use_ema=args.use_ema_policy,
                             reward_scale=args.reward_scale,)
    else:
        raise ValueError(f"Invalid algorithm {args.alg}!")

    trainer = OffPolicyTrainer(
        env=env,
        algorithm=algorithm,
        storage=replay_storage,
        buffer=buffer,
        batch_size=args.batch_size,
        start_step=args.start_step,
        total_step=args.total_step,
        sample_per_iteration=args.sample_per_iteration,
        update_per_iteration=args.update_per_iteration,
        evaluate_env=eval_env,
        save_policy_every=int(args.total_step / 40),
        warmup_with="random",
        log_path=exp_dir,
    )

    example_data = (np.zeros((args.batch_size, obs_dim), dtype=np.uint8),
                    np.zeros((args.batch_size, act_dim), dtype=np.float32),
                    np.zeros((args.batch_size,), dtype=np.float32),
                    np.zeros((args.batch_size, obs_dim), dtype=np.uint8),
                    np.ones((args.batch_size,), dtype=np.float32),)
    trainer.setup(example_data)
    log_git_details(log_file=os.path.join(exp_dir, 'git.diff'))
    
    # Save the arguments to a YAML file
    args_dict = vars(args)
    with open(os.path.join(exp_dir, 'config.yaml'), 'w') as yaml_file:
        yaml.dump(args_dict, yaml_file)
    trainer.run(train_key)

    del buffer
    gc.collect()

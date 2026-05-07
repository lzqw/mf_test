import argparse
import os.path
import time
from functools import partial
import yaml

import jax
import jax.numpy as jnp

# --- 算法 Imports ---
from relax.algorithm.sac import SAC
from relax.algorithm.dacer import DACER
from relax.algorithm.qsm import QSM
from relax.algorithm.dipo import DIPO
from relax.algorithm.qvpo import QVPO
from relax.algorithm.sdac import SDAC
from relax.algorithm.rf import RF
from relax.algorithm.rf2 import RF2
from relax.algorithm.mf import MF
from relax.algorithm.mf2 import MF2

# mf_r: basic reweighting method for Mean Flow SAC
from relax.algorithm.rf_sac import RFSAC
from relax.algorithm.rf_sac_b import RFSACB
from relax.algorithm.rf_sac_estient import RFSACESTIENT
from relax.algorithm.rf_sac_ent import RFSACENT
from relax.algorithm.rf2_sac_ent import RF2SACENT

from relax.algorithm.mf_sac import MFSAC
from relax.algorithm.mf_sac2_ent import MFSAC2ENT
from relax.algorithm.mf_sac_ent import MFSACENT
from relax.algorithm.mf2_sac_ent import MF2SACENT
from relax.algorithm.mf2_sac_ent2 import MF2SACENT2

from relax.buffer import TreeBuffer

# --- 网络 Imports ---
from relax.network.sac import create_sac_net
from relax.network.dacer import create_dacer_net
from relax.network.qsm import create_qsm_net
from relax.network.dipo import create_dipo_net
from relax.network.sdac import create_sdac_net
from relax.network.rf import create_rf_net
from relax.network.rf2 import create_rf2_net
from relax.network.mf import create_mf_net
from relax.network.mf2 import create_mf2_net
from relax.network.qvpo import create_qvpo_net
from relax.network.mf_sac import create_mf_sac_net
from relax.network.mf_sac_ent import create_mf_sac_ent_net
from relax.network.mf_sac2_ent import create_mf_sac2_ent_net
from relax.network.mf2_sac_ent import create_mf2_sac_ent_net
from relax.network.mf2_sac_ent2 import create_mf2_sac_ent2_net

from relax.network.rf_sac import create_rf_sac_net
from relax.network.rf_sac_b import create_rf_sac_b_net
from relax.network.rf_sac_estient import create_rf_sac_estient_net
from relax.network.rf_sac_ent import create_rf_sac_ent_net
from relax.network.rf2_sac_ent import create_rf2_sac_ent_net

# --- Trainer & Utils ---
from relax.trainer.off_policy import OffPolicyTrainer
from relax.env import create_env, create_vector_env
from scripts.experience import Experience, ObsActionPair
from relax.utils.fs import PROJECT_ROOT
from relax.utils.random_utils import seeding
from relax.utils.log_diff import log_git_details


# 引入 Mish 激活函数 (全局定义)
def mish(x: jax.Array):
    return x * jnp.tanh(jax.nn.softplus(x))


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # 算法选择
    parser.add_argument("--alg", type=str, default="mf2")

    # 环境设置 (请使用 register.py 中注册的向量环境 ID，如 dm_control_vector_walker_walk-v0)
    parser.add_argument("--env", type=str, default="dm_control_vector_dog_trot-v0")
    #dm_control_vector_dog_walk-v0
    #dm_control_vector_humanoid_walk-v0
    #dm_control_vector_manipulator_insert_ball-v0
    parser.add_argument("--suffix", type=str, default="vector_test")
    parser.add_argument("--num_vec_envs", type=int, default=2)  # 默认开启5个并行环境

    # 网络参数
    parser.add_argument("--hidden_num", type=int, default=3)
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--diffusion_hidden_dim", type=int, default=256)

    # Diffusion/Flow 参数
    parser.add_argument("--diffusion_steps", type=int, default=10)  # SET 1 FOR MF BASED ALGORITHM
    parser.add_argument("--diffusion_steps_test", type=int, default=1)
    parser.add_argument("--num_ent_timesteps", type=int,
                        default=10)  # The same as diffusion steps for rf. For mf based algorithms, set 5 or 4
    parser.add_argument("--num_particles", type=int, default=32)
    parser.add_argument("--noise_scale", type=float, default=0.001)

    # 训练超参数
    parser.add_argument("--start_step", type=int, default=int(3e4))
    parser.add_argument("--total_step", type=int, default=int(3e6))
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--lr_schedule_end", type=float, default=3e-5)
    parser.add_argument("--alpha_lr", type=float, default=0.005)  # 注意：这里根据参考代码设为了 0.005
    parser.add_argument("--delay_alpha_update", type=float, default=50)
    parser.add_argument("--seed", type=int, default=2)
    parser.add_argument("--replay_buffer_size", type=int, default=int(1e6))
    parser.add_argument("--use_ema_policy", default=True, action="store_true")

    # Entropy / Reweighting 参数
    parser.add_argument("--target_entropy_scale", type=float, default=2.0)
    parser.add_argument("--sample_k", type=int, default=200)
    parser.add_argument("--fix_alpha", type=str2bool, default=True)
    parser.add_argument("--alpha", type=float, default=0.01)
    parser.add_argument("--init_alpha", type=float, default=2.0)

    parser.add_argument("--debug", default=False, action='store_true')  # 修正了 debug 参数定义
    args = parser.parse_args()

    if args.debug:
        from jax import config

        config.update("jax_disable_jit", True)

    master_seed = args.seed
    master_rng, _ = seeding(master_seed)
    env_seed, env_action_seed, eval_env_seed, buffer_seed, init_network_seed, train_seed = map(
        int, master_rng.integers(0, 2 ** 32 - 1, 6)
    )
    init_network_key = jax.random.key(init_network_seed)
    train_key = jax.random.key(train_seed)
    del init_network_seed, train_seed

    # --- 环境创建逻辑 (保持 DMC 向量环境并行化) ---
    print(f"Loading environment: {args.env}")
    if args.num_vec_envs > 0:
        # 如果是 DMC 环境且需要并行，必须确保已注册
        if "dm_control" in args.env:
            from relax.env.dmc.register import register_dm_control_envs

            register_dm_control_envs()

        # 使用 "futex" 模式进行高效并行
        env, obs_dim, act_dim = create_vector_env(
            args.env, args.num_vec_envs, env_seed, env_action_seed, mode="futex"
        )
        print(f"Created {args.num_vec_envs} parallel vector environments (mode='futex').")
    else:
        env, obs_dim, act_dim = create_env(args.env, env_seed, env_action_seed)
        print("Created single environment.")

    eval_env = None

    print(f"Observation Dim: {obs_dim}, Action Dim: {act_dim}")

    hidden_sizes = [args.hidden_dim] * args.hidden_num
    diffusion_hidden_sizes = [args.diffusion_hidden_dim] * args.hidden_num

    buffer = TreeBuffer.from_experience(obs_dim, act_dim, size=args.replay_buffer_size, seed=buffer_seed)

    gelu = partial(jax.nn.gelu, approximate=False)

    # --- 算法实例化 (包含所有参考算法) ---
    if args.alg == 'sdac':
        agent, params = create_sdac_net(init_network_key, obs_dim, act_dim, hidden_sizes, diffusion_hidden_sizes, mish,
                                        num_timesteps=args.diffusion_steps,
                                        num_particles=args.num_particles,
                                        noise_scale=args.noise_scale,
                                        target_entropy_scale=args.target_entropy_scale)
        algorithm = SDAC(agent, params, lr=args.lr, alpha_lr=args.alpha_lr,
                         delay_alpha_update=args.delay_alpha_update,
                         lr_schedule_end=args.lr_schedule_end,
                         use_ema=args.use_ema_policy)
    elif args.alg == 'rf':
        agent, params = create_rf_net(init_network_key, obs_dim, act_dim, hidden_sizes, diffusion_hidden_sizes, mish,
                                      num_timesteps=args.diffusion_steps,
                                      num_timesteps_test=args.diffusion_steps_test,
                                      num_particles=args.num_particles,
                                      noise_scale=args.noise_scale,
                                      target_entropy_scale=args.target_entropy_scale)
        algorithm = RF(agent, params, lr=args.lr, alpha_lr=args.alpha_lr,
                       delay_alpha_update=args.delay_alpha_update,
                       lr_schedule_end=args.lr_schedule_end,
                       use_ema=args.use_ema_policy)
    elif args.alg == 'rf2':
        agent, params = create_rf2_net(init_network_key, obs_dim, act_dim, hidden_sizes, diffusion_hidden_sizes, mish,
                                       num_timesteps=args.diffusion_steps,
                                       num_timesteps_test=args.diffusion_steps_test,
                                       num_particles=args.num_particles,
                                       noise_scale=args.noise_scale,
                                       target_entropy_scale=args.target_entropy_scale)
        algorithm = RF2(agent, params, lr=args.lr, alpha_lr=args.alpha_lr,
                        delay_alpha_update=args.delay_alpha_update,
                        lr_schedule_end=args.lr_schedule_end,
                        use_ema=args.use_ema_policy)
    elif args.alg == 'mf':
        agent, params = create_mf_net(init_network_key, obs_dim, act_dim, hidden_sizes, diffusion_hidden_sizes, mish,
                                      num_timesteps=args.diffusion_steps,
                                      num_timesteps_test=args.diffusion_steps_test,
                                      num_particles=args.num_particles,
                                      noise_scale=args.noise_scale,
                                      target_entropy_scale=args.target_entropy_scale)
        algorithm = MF(agent, params, lr=args.lr, alpha_lr=args.alpha_lr,
                       delay_alpha_update=args.delay_alpha_update,
                       lr_schedule_end=args.lr_schedule_end,
                       use_ema=args.use_ema_policy)
    elif args.alg == 'mf2':
        agent, params = create_mf2_net(init_network_key, obs_dim, act_dim, hidden_sizes, diffusion_hidden_sizes, mish,
                                       num_timesteps=args.diffusion_steps,
                                       num_timesteps_test=args.diffusion_steps_test,
                                       num_particles=args.num_particles,
                                       noise_scale=args.noise_scale,
                                       target_entropy_scale=args.target_entropy_scale)
        algorithm = MF2(agent, params, lr=args.lr, alpha_lr=args.alpha_lr,
                        delay_alpha_update=args.delay_alpha_update,
                        lr_schedule_end=args.lr_schedule_end,
                        use_ema=args.use_ema_policy)
    elif args.alg == "qsm":
        agent, params = create_qsm_net(init_network_key, obs_dim, act_dim, hidden_sizes, num_timesteps=20,
                                       num_particles=args.num_particles)
        algorithm = QSM(agent, params, lr=args.lr, lr_schedule_end=args.lr_schedule_end)
    elif args.alg == "sac":
        agent, params = create_sac_net(init_network_key, obs_dim, act_dim, hidden_sizes, gelu)
        algorithm = SAC(agent, params, lr=args.lr)
    # DSACT and DACERDoubleQ omitted as their imports are missing in provided context
    elif args.alg == "dacer":
        agent, params = create_dacer_net(init_network_key, obs_dim, act_dim, hidden_sizes, diffusion_hidden_sizes, mish,
                                         num_timesteps=args.diffusion_steps)
        algorithm = DACER(agent, params, lr=args.lr, lr_schedule_end=args.lr_schedule_end)
    elif args.alg == "dipo":
        diffusion_buffer = TreeBuffer.from_example(
            ObsActionPair.create_example(obs_dim, act_dim),
            args.total_step,
            int(master_rng.integers(0, 2 ** 32 - 1)),
            remove_batch_dim=False
        )
        TreeBuffer.connect(buffer, diffusion_buffer, lambda exp: ObsActionPair(exp.obs, exp.action))

        agent, params = create_dipo_net(init_network_key, obs_dim, act_dim, hidden_sizes, num_timesteps=100)
        algorithm = DIPO(agent, params, diffusion_buffer, lr=args.lr, action_gradient_steps=30, policy_target_delay=2,
                         action_grad_norm=0.16)
    elif args.alg == "qvpo":
        agent, params = create_qvpo_net(init_network_key, obs_dim, act_dim, hidden_sizes, diffusion_hidden_sizes, mish,
                                        num_timesteps=args.diffusion_steps,
                                        num_particles=args.num_particles,
                                        noise_scale=args.noise_scale)
        algorithm = QVPO(agent, params, lr=args.lr, alpha_lr=args.alpha_lr, delay_alpha_update=args.delay_alpha_update)

    elif args.alg == 'rf_sac':
        agent, params = create_rf_sac_net(init_network_key, obs_dim, act_dim, hidden_sizes, diffusion_hidden_sizes,
                                          mish,
                                          num_timesteps=args.diffusion_steps,
                                          num_timesteps_test=args.diffusion_steps_test,
                                          num_particles=args.num_particles,
                                          noise_scale=args.noise_scale,
                                          target_entropy_scale=args.target_entropy_scale)
        algorithm = RFSAC(agent, params, lr=args.lr, alpha_lr=args.alpha_lr,
                          delay_alpha_update=args.delay_alpha_update,
                          lr_schedule_end=args.lr_schedule_end,
                          use_ema=args.use_ema_policy,
                          sample_k=args.sample_k)
    elif args.alg == 'rf_sac_b':
        agent, params = create_rf_sac_b_net(init_network_key, obs_dim, act_dim, hidden_sizes, diffusion_hidden_sizes,
                                            mish,
                                            num_timesteps=args.diffusion_steps,
                                            num_timesteps_test=args.diffusion_steps_test,
                                            num_particles=args.num_particles,
                                            noise_scale=args.noise_scale,
                                            target_entropy_scale=args.target_entropy_scale)
        algorithm = RFSACB(agent, params, lr=args.lr, alpha_lr=args.alpha_lr,
                           delay_alpha_update=args.delay_alpha_update,
                           lr_schedule_end=args.lr_schedule_end,
                           use_ema=args.use_ema_policy,
                           sample_k=args.sample_k)

    elif args.alg == 'rf_sac_estient':
        agent, params = create_rf_sac_estient_net(init_network_key, obs_dim, act_dim, hidden_sizes,
                                                  diffusion_hidden_sizes, mish,
                                                  num_timesteps=args.diffusion_steps,
                                                  num_timesteps_test=args.diffusion_steps_test,
                                                  num_particles=args.num_particles,
                                                  noise_scale=args.noise_scale,
                                                  target_entropy_scale=args.target_entropy_scale)
        algorithm = RFSACESTIENT(agent, params, lr=args.lr, alpha_lr=args.alpha_lr,
                                 delay_alpha_update=args.delay_alpha_update,
                                 lr_schedule_end=args.lr_schedule_end,
                                 use_ema=args.use_ema_policy,
                                 sample_k=args.sample_k)

    elif args.alg == 'rf_sac_ent':
        agent, params = create_rf_sac_ent_net(init_network_key, obs_dim, act_dim, hidden_sizes, diffusion_hidden_sizes,
                                              mish,
                                              num_timesteps=args.diffusion_steps,
                                              num_timesteps_test=args.diffusion_steps_test,
                                              num_particles=args.num_particles,
                                              noise_scale=args.noise_scale,
                                              target_entropy_scale=args.target_entropy_scale,
                                              alpha_value=args.alpha)
        algorithm = RFSACENT(agent, params, lr=args.lr, alpha_lr=args.alpha_lr,
                             delay_alpha_update=args.delay_alpha_update,
                             lr_schedule_end=args.lr_schedule_end,
                             use_ema=args.use_ema_policy,
                             sample_k=args.sample_k,
                             fixed_alpha=args.fix_alpha)

    elif args.alg == 'rf2_sac_ent':
        agent, params = create_rf2_sac_ent_net(init_network_key, obs_dim, act_dim, hidden_sizes, diffusion_hidden_sizes,
                                               mish,
                                               num_timesteps=args.diffusion_steps,
                                               num_timesteps_test=args.diffusion_steps_test,
                                               num_ent_timesteps=args.num_ent_timesteps,
                                               num_particles=args.num_particles,
                                               noise_scale=args.noise_scale,
                                               target_entropy_scale=args.target_entropy_scale,
                                               alpha_value=args.alpha,
                                               fixed_alpha=args.fix_alpha,
                                               init_alpha=args.init_alpha)

        algorithm = RF2SACENT(agent, params, lr=args.lr, alpha_lr=args.alpha_lr,
                              delay_alpha_update=args.delay_alpha_update,
                              lr_schedule_end=args.lr_schedule_end,
                              use_ema=args.use_ema_policy,
                              sample_k=args.sample_k,
                              alpha_value=args.alpha,
                              reward_scale=1.0,
                              fixed_alpha=args.fix_alpha)

    elif args.alg == 'mf_sac_ent':
        agent, params = create_mf_sac_ent_net(init_network_key, obs_dim, act_dim, hidden_sizes, diffusion_hidden_sizes,
                                              mish,
                                              num_timesteps=args.diffusion_steps,
                                              num_timesteps_test=args.diffusion_steps_test,
                                              num_particles=args.num_particles,
                                              noise_scale=args.noise_scale,
                                              target_entropy_scale=args.target_entropy_scale,
                                              alpha_value=args.alpha)
        algorithm = MFSACENT(agent, params, lr=args.lr, alpha_lr=args.alpha_lr,
                             delay_alpha_update=args.delay_alpha_update,
                             lr_schedule_end=args.lr_schedule_end,
                             use_ema=args.use_ema_policy,
                             sample_k=args.sample_k)

    elif args.alg == 'mf_sac2_ent':
        agent, params = create_mf_sac2_ent_net(init_network_key, obs_dim, act_dim, hidden_sizes, diffusion_hidden_sizes,
                                               mish,
                                               num_timesteps=args.diffusion_steps,
                                               num_timesteps_test=args.diffusion_steps_test,
                                               num_particles=args.num_particles,
                                               noise_scale=args.noise_scale,
                                               target_entropy_scale=args.target_entropy_scale,
                                               fixed_alpha=args.alpha)
        algorithm = MFSAC2ENT(agent, params, lr=args.lr, alpha_lr=args.alpha_lr,
                              delay_alpha_update=args.delay_alpha_update,
                              lr_schedule_end=args.lr_schedule_end,
                              use_ema=args.use_ema_policy,
                              sample_k=args.sample_k)

    elif args.alg == 'mf2_sac_ent':
        agent, params = create_mf2_sac_ent_net(init_network_key, obs_dim, act_dim, hidden_sizes, diffusion_hidden_sizes,
                                               mish,
                                               num_timesteps=args.diffusion_steps,
                                               num_timesteps_test=args.diffusion_steps_test,
                                               num_particles=args.num_particles,
                                               noise_scale=args.noise_scale,
                                               target_entropy_scale=args.target_entropy_scale,
                                               alpha_value=args.alpha)
        algorithm = MF2SACENT(agent, params, lr=args.lr, alpha_lr=args.alpha_lr,
                              delay_alpha_update=args.delay_alpha_update,
                              lr_schedule_end=args.lr_schedule_end,
                              use_ema=args.use_ema_policy,
                              sample_k=args.sample_k)

    elif args.alg == 'mf2_sac_ent2':
        agent, params = create_mf2_sac_ent2_net(init_network_key, obs_dim, act_dim, hidden_sizes,
                                                diffusion_hidden_sizes, mish,
                                                num_timesteps=args.diffusion_steps,
                                                num_ent_timesteps=args.num_ent_timesteps,
                                                num_timesteps_test=args.diffusion_steps_test,
                                                num_particles=args.num_particles,
                                                noise_scale=args.noise_scale,
                                                target_entropy_scale=args.target_entropy_scale,
                                                alpha_value=args.alpha,
                                                fixed_alpha=args.fix_alpha,
                                                init_alpha=args.init_alpha)
        algorithm = MF2SACENT2(agent, params, lr=args.lr, alpha_lr=args.alpha_lr,
                               delay_alpha_update=args.delay_alpha_update,
                               lr_schedule_end=args.lr_schedule_end,
                               use_ema=args.use_ema_policy,
                               sample_k=args.sample_k,
                               alpha_value=args.alpha,
                               fixed_alpha=args.fix_alpha
                               )

    elif args.alg == 'mf_sac':
        agent, params = create_mf_sac_net(init_network_key, obs_dim, act_dim, hidden_sizes, diffusion_hidden_sizes,
                                          mish,
                                          num_timesteps=args.diffusion_steps,
                                          num_timesteps_test=args.diffusion_steps_test,
                                          num_particles=args.num_particles,
                                          noise_scale=args.noise_scale,
                                          target_entropy_scale=args.target_entropy_scale)
        algorithm = MFSAC(agent, params, lr=args.lr, alpha_lr=args.alpha_lr,
                          delay_alpha_update=args.delay_alpha_update,
                          lr_schedule_end=args.lr_schedule_end,
                          use_ema=args.use_ema_policy,
                          sample_k=args.sample_k)

    else:
        raise ValueError(f"Invalid algorithm {args.alg}!")

    # --- 训练器初始化与运行 ---
    exp_dir = PROJECT_ROOT / "logs" / args.env / (
            args.alg + '_' + time.strftime("%Y-%m-%d_%H-%M-%S") + f'_s{args.seed}_{args.suffix}')

    trainer = OffPolicyTrainer(
        env=env,
        algorithm=algorithm,
        buffer=buffer,
        start_step=args.start_step,
        total_step=args.total_step,
        sample_per_iteration=1,
        evaluate_env=eval_env,
        save_policy_every=int(args.total_step / 40),
        warmup_with="random",
        log_path=exp_dir,
    )

    trainer.setup(Experience.create_example(obs_dim, act_dim, trainer.batch_size))
    log_git_details(log_file=os.path.join(exp_dir, 'git.diff'))

    # 保存配置
    os.makedirs(exp_dir, exist_ok=True)
    args_dict = vars(args)
    with open(os.path.join(exp_dir, 'config.yaml'), 'w') as yaml_file:
        yaml.dump(args_dict, yaml_file)

    print(f"Starting training for {args.alg} on {args.env}...")
    trainer.run(train_key)

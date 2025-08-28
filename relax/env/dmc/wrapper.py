import numpy as np
import gymnasium as gym
from gymnasium import spaces
from dm_control import suite
from dm_env import StepType

class DMControlToGymWrapper(gym.Env):
    """Wrapper to convert DeepMind Control Suite env to a Gymnasium-compatible environment."""

    def __init__(self, domain_name=None, task_name=None, env=None, render_size=(84, 84)):
        super().__init__()
        if env is not None:
            self.env = env
        else:
            self.env = suite.load(domain_name=domain_name, task_name=task_name)
        self.render_size = render_size
        self.action_spec = self.env.action_spec()
        self.observation_spec = self.env.observation_spec()
        self.action_space = spaces.Box(
            low=np.array([self.action_spec.minimum.astype(np.float32)]*self.action_spec.shape[0]),
            high=np.array([self.action_spec.maximum.astype(np.float32)]*self.action_spec.shape[0]),
            dtype=np.float32
        )
        obs_dim = np.prod(self.observation_spec.shape).astype(int)
        self.observation_space = spaces.Box(
            low=0, high=255, shape=(obs_dim,), dtype=np.uint8
        )

    def _flatten_observation(self, time_step):
        return time_step.observation.flatten()

    def reset(self, seed=None, options=None):
        if seed is not None:
            np.random.seed(seed)
        time_step = self.env.reset()
        return self._flatten_observation(time_step), {}

    def step(self, action):
        time_step = self.env.step(action)
        obs = self._flatten_observation(time_step)
        reward = time_step.reward if time_step.reward is not None else 0.0
        terminated = time_step.step_type == StepType.LAST
        truncated = False  # DMC does not define truncation explicitly
        return obs, reward, terminated, truncated, {}

    def render(self):
        return self.env.physics.render(*self.render_size, camera_id=0)

    def close(self):
        pass

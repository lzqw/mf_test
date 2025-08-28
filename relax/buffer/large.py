import datetime
import io
import random
import traceback
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import IterableDataset, default_collate
import torch.multiprocessing as mp
from jax.tree_util import tree_map

def episode_len(episode):
    # subtract -1 because the dummy first transition
    return next(iter(episode.values())).shape[0] - 1


def save_episode(episode, fn):
    with io.BytesIO() as bs:
        np.savez_compressed(bs, **episode)
        bs.seek(0)
        with fn.open('wb') as f:
            f.write(bs.read())


def load_episode(fn):
    with fn.open('rb') as f:
        episode = np.load(f)
        episode = {k: episode[k] for k in episode.keys()}
        return episode
    

class RandomShiftsAug(nn.Module):
    def __init__(self, pad):
        super().__init__()
        self.pad = pad

    def forward(self, x):
        x = torch.from_numpy(x).to(device=torch.device('cpu'), dtype=torch.float32)
        if len(x.shape) == 1:
            x = x.reshape(1, -1)
        x = x.reshape(x.shape[0], -1, 84, 84)
        n, c, h, w = x.size()
        assert h == w
        padding = tuple([self.pad] * 4)
        x = F.pad(x, padding, 'replicate')
        eps = 1.0 / (h + 2 * self.pad)
        arange = torch.linspace(-1.0 + eps,
                                1.0 - eps,
                                h + 2 * self.pad,
                                device=x.device,
                                dtype=x.dtype)[:h]
        arange = arange.unsqueeze(0).repeat(h, 1).unsqueeze(2)
        base_grid = torch.cat([arange, arange.transpose(1, 0)], dim=2)
        base_grid = base_grid.unsqueeze(0).repeat(n, 1, 1, 1)

        shift = torch.randint(0,
                              2 * self.pad + 1,
                              size=(n, 1, 1, 2),
                              device=x.device,
                              dtype=x.dtype)
        shift *= 2.0 / (h + 2 * self.pad)

        grid = base_grid + shift
        x = F.grid_sample(x,
                        grid,
                        padding_mode='zeros',
                        align_corners=False)
        x = x.reshape(x.shape[0], -1).squeeze()
        return  x.detach().cpu().numpy()

class ReplayBufferStorage:
    def __init__(self, data_specs, replay_dir, num_envs):
        self._data_specs = data_specs
        self._replay_dir = replay_dir
        self._num_envs = num_envs
        replay_dir.mkdir(exist_ok=True)
        if num_envs > 0:
            self._current_episode = [defaultdict(list) for _ in range(self._num_envs)]
        else:
            self._current_episode = defaultdict(list)
        self._preload()

    def __len__(self):
        return self._num_transitions
    
    def add(self, time_step):
        time_step = time_step[0]
        for spec in self._data_specs:
            value = time_step[spec.name]
            if np.isscalar(value):
                value = np.full(spec.shape, value, spec.dtype)
            assert spec.shape == value.shape and spec.dtype == value.dtype
            self._current_episode[spec.name].append(value)
        if time_step['is_last']:
            episode = dict()
            for spec in self._data_specs:
                value = self._current_episode[spec.name]
                episode[spec.name] = np.array(value, spec.dtype)
            self._current_episode = defaultdict(list)
            self._store_episode(episode)

    def add_batch(self, time_steps):
        for idx, ts in enumerate(time_steps):
            for spec in self._data_specs:
                value = ts[spec.name]
                if np.isscalar(value):
                    value = np.full(spec.shape, value, spec.dtype)
                assert spec.shape == value.shape and spec.dtype == value.dtype
                self._current_episode[idx][spec.name].append(value)
            if ts['is_last']:
                episode = dict()
                for spec in self._data_specs:
                    value = self._current_episode[idx][spec.name]
                    episode[spec.name] = np.array(value, spec.dtype)
                self._current_episode[idx] = defaultdict(list)
                self._store_episode(episode)

    def _preload(self):
        self._num_episodes = 0
        self._num_transitions = 0
        for fn in self._replay_dir.glob('*.npz'):
            _, _, eps_len = fn.stem.split('_')
            self._num_episodes += 1
            self._num_transitions += int(eps_len)

    def _store_episode(self, episode):
        eps_idx = self._num_episodes
        eps_len = episode_len(episode)
        self._num_episodes += 1
        self._num_transitions += eps_len
        ts = datetime.datetime.now().strftime('%Y%m%dT%H%M%S')
        eps_fn = f'{ts}_{eps_idx}_{eps_len}.npz'
        save_episode(episode, self._replay_dir / eps_fn)


class ReplayBuffer(IterableDataset):
    def __init__(self, replay_dir, max_size, num_workers, nstep, discount,
                 fetch_every, save_snapshot, augment=False):
        self._replay_dir = replay_dir
        self._size = 0
        self._max_size = max_size
        self._num_workers = max(1, num_workers)
        self._episode_fns = []
        self._episodes = dict()
        self._nstep = nstep
        self._discount = discount
        self._fetch_every = fetch_every
        self._samples_since_last_fetch = fetch_every
        self._save_snapshot = save_snapshot
        self._augment = augment
        if self._augment:
            self._aug = RandomShiftsAug(pad=4)

    def _sample_episode(self):
        eps_fn = random.choice(self._episode_fns)
        return self._episodes[eps_fn]

    def _store_episode(self, eps_fn):
        try:
            episode = load_episode(eps_fn)
        except:
            return False
        eps_len = episode_len(episode)
        while eps_len + self._size > self._max_size:
            early_eps_fn = self._episode_fns.pop(0)
            early_eps = self._episodes.pop(early_eps_fn)
            self._size -= episode_len(early_eps)
            early_eps_fn.unlink(missing_ok=True)
        self._episode_fns.append(eps_fn)
        self._episode_fns.sort()
        self._episodes[eps_fn] = episode
        self._size += eps_len

        if not self._save_snapshot:
            eps_fn.unlink(missing_ok=True)
        return True

    def _try_fetch(self):
        if self._samples_since_last_fetch < self._fetch_every:
            return
        self._samples_since_last_fetch = 0
        try:
            worker_id = torch.utils.data.get_worker_info().id
        except:
            worker_id = 0
        eps_fns = sorted(self._replay_dir.glob('*.npz'), reverse=True)
        fetched_size = 0
        for eps_fn in eps_fns:
            eps_idx, eps_len = [int(x) for x in eps_fn.stem.split('_')[1:]]
            if eps_idx % self._num_workers != worker_id:
                continue
            if eps_fn in self._episodes.keys():
                break
            if fetched_size + eps_len > self._max_size:
                break
            fetched_size += eps_len
            if not self._store_episode(eps_fn):
                break

    def _sample(self):
        try:
            self._try_fetch()
        except:
            traceback.print_exc()
        self._samples_since_last_fetch += 1
        episode = self._sample_episode()
        # add +1 for the first dummy transition
        idx = np.random.randint(0, episode_len(episode) - self._nstep + 1) + 1
        obs = episode['observation'][idx - 1]
        action = episode['action'][idx]
        next_obs = episode['observation'][idx + self._nstep - 1]
        reward = np.zeros_like(episode['reward'][idx])
        discount = np.ones_like(episode['reward'][idx])
        for i in range(self._nstep):
            step_reward = episode['reward'][idx + i]
            reward += discount * step_reward
            # discount *= episode['discount'][idx + i] * self._discount
            discount *= self._discount
        #data aug
        if self._augment:
            obs = self._aug(obs)
            next_obs = self._aug(next_obs)
        return (obs, action, reward, next_obs, discount)

    def __iter__(self):
        while True:
            yield self._sample()

def _worker_init_fn(worker_id):
    seed = np.random.get_state()[1][0] + worker_id
    np.random.seed(seed)
    random.seed(seed)

def numpy_collate(batch):
  return tree_map(np.asarray, default_collate(batch))

def make_replay_loader(replay_dir, max_size, batch_size, num_workers,
                       save_snapshot, nstep, discount):
    max_size_per_worker = max_size // max(1, num_workers)

    iterable = ReplayBuffer(replay_dir,
                            max_size_per_worker,
                            num_workers,
                            nstep,
                            discount,
                            fetch_every=1000,
                            save_snapshot=save_snapshot)

    loader = torch.utils.data.DataLoader(iterable,
                                         batch_size=batch_size,
                                         num_workers=num_workers,
                                         pin_memory=False,
                                         worker_init_fn=_worker_init_fn,
                                         collate_fn=numpy_collate,
                                         multiprocessing_context=mp.get_context("spawn"))
    return loader

# def make_replay_loader(replay_dir, max_size, batch_size, save_snapshot, nstep, discount):
#     buffer = ReplayBuffer(replay_dir,
#                             max_size,
#                             nstep,
#                             discount,
#                             fetch_every=1000,
#                             save_snapshot=save_snapshot)
    
#     def generator():
#         while True:
#             batch = [buffer._sample() for _ in range(batch_size)]
#             obs_b, act_b, rew_b, next_b, discount_b = zip(*batch)
#             yield (np.stack(obs_b),
#                    np.stack(act_b),
#                    np.stack(rew_b),
#                    np.stack(next_b),
#                    np.stack(discount_b))

#     return generator()

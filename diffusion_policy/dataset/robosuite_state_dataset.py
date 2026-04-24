from typing import Dict, List, Optional, Sequence, Tuple
import copy
import json
from pathlib import Path

import numpy as np
import torch

from diffusion_policy.dataset.base_dataset import BaseImageDataset
from diffusion_policy.model.common.normalizer import LinearNormalizer, EmptyNormalizer


def state2agentenv(state: np.ndarray, env_name: str):
    if env_name in ["Door", "Door_Close", "Old_Door", "Old_Door_Close"]:
        # state: 24
        # agent: eef(3) + gripper(6) + door_to_eef(3) + handle_to_eef(3) + check_grasp(1) = 16
        # env: door_pos(3) + handle_pos(3) + hinge_qpos(1) + handle_qpos(1) = 8
        agent_state = np.concatenate([state[0:9], state[15:21],
                                      state[23:24]])
        env_state = np.concatenate([state[9:15], state[21:23]])
    elif env_name in ["NutAssemblyRound", "NutDisAssemblyRound"]:
        # state: 22
        # agent: eef(3)+eef_to_nut(3)+check_grasp(1) = 7
        # env: nut(3)+peg(3)+goal(3)+nut_to_peg(3)+nut_to_goal(3) = 15
        agent_state = np.concatenate([state[0:3], state[12:15],
                                      state[21:22]])
        env_state = np.concatenate([state[3:6], state[6:9],
                                    state[9:12], state[15:18], state[18:21]])
    elif env_name in ["TwoArmPegInHole", "TwoArmPegRemoval"]:
        # state: 26
        # agent: robot0_eef_pos(3)
        # env: remaining 23 dims
        agent_state = state[0:3]
        env_state = state[3:26]
    elif env_name in ["Stack", "UnStack"]:
        # state: 24
        # agent: eef(3)+eef_to_cubeA(3)+check_grasp(1) = 7
        # env: others = 17
        agent_state = np.concatenate([state[0:3], state[14:17],
                                      state[22:23]])
        env_state = np.concatenate([state[3:14], state[17:22],
                                    state[22:24]])
    else:
        raise ValueError(f"Unsupported env_name: {env_name}")
    return agent_state, env_state


def _recursive_to_torch(data):
    if isinstance(data, dict):
        return {k: _recursive_to_torch(v) for k, v in data.items()}
    if isinstance(data, np.ndarray):
        return torch.from_numpy(data)
    if isinstance(data, (np.integer, int)):
        return torch.tensor(data, dtype=torch.long)
    if isinstance(data, (np.floating, float)):
        return torch.tensor(data, dtype=torch.float32)
    return data


class RobosuiteStateDataset(BaseImageDataset):
    """
    transitions[i] = [state, action, reward, next_state, done]

    __getitem__ returns:
    {
        "obs": {
            "all_state":   (H, D_state),
            "agent_state": (H, D_agent),
            "env_state":   (H, D_env),
        },
        "action": (H, D_action),
        "episode_id": scalar int64,   # if return_meta=True
        "anchor_step": scalar int64,  # if return_meta=True
    }
    """
    def __init__(
            self,
            transition_path: str,
            env_name: str,
            horizon: int = 16,
            pad_before: int = 0,
            pad_after: int = 0,
            episode_horizon: int = 500,
            seed: int = 42,
            val_ratio: float = 0.0,
            max_train_episodes: Optional[int] = None,
            return_meta: bool = True,
            strict_success_episode: bool = True,
            norm_input_output: bool = True,
            _shared: Optional[Dict] = None,
            _split: str = "train",
            _episode_mask: Optional[np.ndarray] = None,
    ):
        super().__init__()
        self.transition_path = transition_path
        self.env_name = env_name
        self.horizon = int(horizon)
        self.pad_before = int(pad_before)
        self.pad_after = int(pad_after)
        self.episode_horizon = int(episode_horizon)
        self.seed = int(seed)
        self.val_ratio = float(val_ratio)
        self.max_train_episodes = max_train_episodes
        self.return_meta = bool(return_meta)
        self.strict_success_episode = bool(strict_success_episode)
        self.norm_input_output = norm_input_output
        self.obs_keys = ("all_state", "agent_state", "env_state")
        self._split = _split

        if self.horizon <= 0:
            raise ValueError(f"horizon must be positive, got {self.horizon}")
        if self.episode_horizon <= 0:
            raise ValueError(f"episode_horizon must be positive, got {self.episode_horizon}")
        if not (0.0 <= self.val_ratio < 1.0):
            raise ValueError(f"val_ratio must be in [0, 1), got {self.val_ratio}")

        self._shared = _shared if _shared is not None else self._build_shared()
        self._train_episode_mask = self._shared["train_episode_mask"]
        self._val_episode_mask = self._shared["val_episode_mask"]

        if _episode_mask is None:
            self._episode_mask = self._train_episode_mask if _split == "train" else self._val_episode_mask
        else:
            self._episode_mask = _episode_mask.astype(bool)

        self.train_mask = self._episode_mask.copy()
        self._sample_index = self._build_sample_index(self._episode_mask)

        self.state_dim = int(self._shared["states"].shape[1])
        self.action_dim = int(self._shared["actions"].shape[1])
        self.agent_state_dim = int(self._shared["agent_states"].shape[1])
        self.env_state_dim = int(self._shared["env_states"].shape[1])

        print(
            f"[RobosuiteStateDataset] split={self._split}, path={self.transition_path}, "
            f"episodes={int(self._episode_mask.sum())}, samples={len(self)}"
        )

    def _split_episodes(self, dones: np.ndarray, num_steps: int) -> List[Tuple[int, int]]:
        done_idx = np.where(dones > 0.5)[0]
        if len(done_idx) > 0:
            episodes = []
            start = 0
            for end_idx in done_idx:
                episodes.append((start, end_idx + 1))
                start = end_idx + 1
            if start < num_steps:
                episodes.append((start, num_steps))
            return episodes

        episodes = []
        for start in range(0, num_steps, self.episode_horizon):
            end = min(start + self.episode_horizon, num_steps)
            episodes.append((start, end))
        return episodes

    def _truncate_episodes_by_success(
            self,
            episodes: Sequence[Tuple[int, int]],
            rewards: np.ndarray,
    ) -> List[Tuple[int, int]]:
        trimmed = []
        missing_success = []

        for episode_id, (start, end) in enumerate(episodes):
            ep_rewards = rewards[start:end]
            success_steps = np.where(ep_rewards >= 0.5)[0]
            if len(success_steps) == 0:
                missing_success.append(episode_id)
                trimmed.append((start, end))
                continue
            first_success = int(success_steps[0])
            trimmed.append((start, start + first_success + 100))  # NOTE: append how many?

        if self.strict_success_episode and len(missing_success) > 0:
            raise ValueError(
                f"Episodes without success reward found: {missing_success}. "
                f"Set strict_success_episode=False to keep them."
            )
        return trimmed

    def _make_split_masks(self, n_episodes: int) -> Tuple[np.ndarray, np.ndarray]:
        if n_episodes <= 0:
            raise ValueError("No episodes found after parsing transitions.")

        val_mask = np.zeros(n_episodes, dtype=bool)
        n_val = int(n_episodes * self.val_ratio)
        if n_val > 0:
            rng = np.random.RandomState(self.seed)
            val_indices = rng.permutation(n_episodes)[:n_val]
            val_mask[val_indices] = True

        train_mask = ~val_mask
        if self.max_train_episodes is not None:
            max_n = int(self.max_train_episodes)
            if max_n <= 0:
                raise ValueError(f"max_train_episodes must be positive, got {self.max_train_episodes}")
            train_indices = np.where(train_mask)[0]
            if len(train_indices) > max_n:
                rng = np.random.RandomState(self.seed)
                selected = rng.choice(train_indices, size=max_n, replace=False)
                new_train_mask = np.zeros_like(train_mask)
                new_train_mask[selected] = True
                train_mask = new_train_mask

        return train_mask, val_mask

    def _build_shared(self) -> Dict:
        transitions = np.load(self.transition_path, allow_pickle=True)
        if transitions.ndim != 2 or transitions.shape[1] < 5:
            raise ValueError(f"Unexpected transitions shape {transitions.shape}, expected (N, 5)-like.")

        states = np.stack(transitions[:, 0], axis=0).astype(np.float32)
        actions = np.stack(transitions[:, 1], axis=0).astype(np.float32)
        rewards = np.array(transitions[:, 2], dtype=np.float32)
        dones = np.array(transitions[:, 4], dtype=np.float32)

        episodes = self._split_episodes(dones=dones, num_steps=len(states))
        episodes = self._truncate_episodes_by_success(episodes=episodes, rewards=rewards)

        agent_states, env_states = [], []
        for state in states:
            agent_state, env_state = state2agentenv(state, self.env_name)
            agent_states.append(np.asarray(agent_state, dtype=np.float32))
            env_states.append(np.asarray(env_state, dtype=np.float32))
        agent_states = np.stack(agent_states, axis=0)
        env_states = np.stack(env_states, axis=0)

        train_mask, val_mask = self._make_split_masks(n_episodes=len(episodes))

        return {
            "states": states,
            "actions": actions,
            "rewards": rewards,
            "dones": dones,
            "agent_states": agent_states,
            "env_states": env_states,
            "episodes": episodes,
            "train_episode_mask": train_mask,
            "val_episode_mask": val_mask,
        }

    def _build_sample_index(self, episode_mask: np.ndarray) -> List[Tuple[int, int, int, int, int]]:
        sample_index = []
        episodes = self._shared["episodes"]
        seq_len = self.horizon

        for episode_id, use_episode in enumerate(episode_mask):
            if not bool(use_episode):
                continue

            start, end = episodes[episode_id]
            ep_len = end - start
            if ep_len <= 0:
                continue

            min_start = -self.pad_before
            max_start = ep_len - seq_len + self.pad_after
            if max_start < min_start:
                max_start = min_start

            for window_start in range(min_start, max_start + 1):
                anchor_step = window_start + self.pad_before
                sample_index.append((episode_id, start, end, window_start, anchor_step))

        return sample_index

    def __len__(self) -> int:
        return len(self._sample_index)

    def get_validation_dataset(self):
        val_set = copy.copy(self)
        val_set._split = "val"
        val_set._episode_mask = val_set._val_episode_mask.copy()
        val_set.train_mask = val_set._episode_mask.copy()
        val_set._sample_index = val_set._build_sample_index(val_set._episode_mask)
        return val_set

    def get_normalizer(self, **kwargs) -> LinearNormalizer:
        normalizer = LinearNormalizer()
        for obs_key in self.obs_keys:
            normalizer[obs_key] = EmptyNormalizer.create_identity()
        normalizer["action"] = EmptyNormalizer.create_identity()
        return normalizer

    def compute_statistics(self) -> Dict:
        """
        计算整个数据集（共享底层 transitions）的统计信息。
        每个字段返回:
            {
                "min":  按维度最小值(list),
                "max":  按维度最大值(list),
                "mean": 按维度均值(list),
                "std":  按维度标准差(list),
                "count": 样本步数(int)
            }
        """
        arrays = {
            "all_state": self._shared["states"],
            "agent_state": self._shared["agent_states"],
            "env_state": self._shared["env_states"],
            "action": self._shared["actions"],
        }

        stats = {
            "num_episodes": int(len(self._shared["episodes"])),
            "num_steps": int(self._shared["states"].shape[0]),
            "stats": {},
        }

        for key, value in arrays.items():
            value = np.asarray(value, dtype=np.float64)
            stats["stats"][key] = {
                "min": value.min(axis=0).tolist(),
                "max": value.max(axis=0).tolist(),
                "mean": value.mean(axis=0).tolist(),
                "std": value.std(axis=0).tolist(),
                "count": int(value.shape[0]),
            }

        return stats

    def compute_statistics_and_save_json(self, json_path: str) -> Dict:
        stats = self.compute_statistics()
        json_file = Path(json_path)
        json_file.parent.mkdir(parents=True, exist_ok=True)
        with json_file.open("w", encoding="utf-8") as f:
            json.dump(stats, f, indent=2, ensure_ascii=False)
        return stats

    def _sample_to_data(self, descriptor: Tuple[int, int, int, int, int]) -> Dict:
        episode_id, start, end, window_start, anchor_step = descriptor
        ep_len = end - start
        if ep_len <= 0:
            raise RuntimeError(f"Invalid episode length {ep_len} for episode_id={episode_id}")

        local_indices = np.arange(window_start, window_start + self.horizon, dtype=np.int64)
        local_indices = np.clip(local_indices, 0, ep_len - 1)
        global_indices = start + local_indices

        all_state = self._shared["states"][global_indices].astype(np.float32)
        agent_state = self._shared["agent_states"][global_indices].astype(np.float32)
        env_state = self._shared["env_states"][global_indices].astype(np.float32)
        action = self._shared["actions"][global_indices].astype(np.float32)

        data = {
            "obs": {
                "all_state": all_state,
                "agent_state": agent_state,
                "env_state": env_state,
            },
            "action": action,
        }
        if self.return_meta:
            data["episode_id"] = np.asarray(episode_id, dtype=np.int64)
            data["anchor_step"] = np.asarray(anchor_step, dtype=np.int64)
        return data

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        data = self._sample_to_data(self._sample_index[idx])
        data = _recursive_to_torch(data)

        if self.norm_input_output:
            if not hasattr(self, "_minmax_stats_cache"):
                self._minmax_stats_cache = self.compute_statistics()["stats"]

            eps = 1e-12
            for key in ("all_state", "agent_state", "env_state"):
                tensor = data["obs"][key]
                min_v = torch.as_tensor(
                    self._minmax_stats_cache[key]["min"],
                    dtype=tensor.dtype,
                    device=tensor.device
                )
                max_v = torch.as_tensor(
                    self._minmax_stats_cache[key]["max"],
                    dtype=tensor.dtype,
                    device=tensor.device
                )
                data["obs"][key] = (tensor - min_v) / (max_v - min_v + eps)

            action = data["action"]
            action_min = torch.as_tensor(
                self._minmax_stats_cache["action"]["min"],
                dtype=action.dtype,
                device=action.device
            )
            action_max = torch.as_tensor(
                self._minmax_stats_cache["action"]["max"],
                dtype=action.dtype,
                device=action.device
            )
            data["action"] = (action - action_min) / (action_max - action_min + eps)
        else:
            pass

        return data

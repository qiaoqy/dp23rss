import os
import copy
import bisect
import numpy as np
import torch
import torch.nn.functional as F
from typing import List
from torch.utils.data import ConcatDataset
from torchvision.transforms import transforms

from robokit.datasets.tcl_datasets import TCLDataset, TCLDatasetHDF5
from robokit.debug_utils.printer import print_batch

from diffusion_policy.dataset.base_dataset import BaseImageDataset
from diffusion_policy.model.common.normalizer import LinearNormalizer, EmptyNormalizer


# source: https://github.com/facebookresearch/drqv2/blob/main/drqv2.py
class RandomShiftsAug(torch.nn.Module):
    def __init__(self, pad):
        super().__init__()
        self.pad = pad

    def forward(self, x_B_C_H_W: torch.Tensor) -> torch.Tensor:
        input_ndim = x_B_C_H_W.ndim
        if input_ndim == 3:
            x_B_C_H_W = x_B_C_H_W.unsqueeze(0)

        x = x_B_C_H_W.float()
        n, c, h, w = x.size()

        # 1. 对图像进行填充
        padding = tuple([self.pad] * 4)
        x_padded = F.pad(x, padding, "replicate")

        # 2. 创建一个标准化的坐标网格 ([-1, 1])
        # 这个网格对应于原始图像尺寸 (h, w)
        arange_h = torch.linspace(-1.0, 1.0, h, device=x.device, dtype=x.dtype)
        arange_w = torch.linspace(-1.0, 1.0, w, device=x.device, dtype=x.dtype)

        # 使用 meshgrid 创建一个 (h, w, 2) 的网格
        grid_h, grid_w = torch.meshgrid(arange_h, arange_w, indexing='ij')
        base_grid = torch.stack((grid_w, grid_h), dim=-1)  # (x, y) 坐标
        base_grid = base_grid.unsqueeze(0).repeat(n, 1, 1, 1)  # 扩展到批次维度 (n, h, w, 2)

        # 3. 生成单个随机位移，并将其应用于所有帧
        # 在填充后的像素空间中生成随机整数位移
        # 将 size 的第一个维度从 n 改为 1，以确保所有帧使用相同的位移
        shift_xy = torch.randint(0, 2 * self.pad + 1, size=(1, 1, 1, 2), device=x.device, dtype=x.dtype)

        # 4. 将像素位移转换为 [-1, 1] 坐标空间的位移
        # 注意：h 和 w 的缩放因子不同
        shift_w = shift_xy[..., 0] * 2.0 / (w + 2 * self.pad)
        shift_h = shift_xy[..., 1] * 2.0 / (h + 2 * self.pad)
        shift = torch.stack((shift_w, shift_h), dim=-1)

        # 5. 将位移应用到基础网格上
        # shift 的形状是 (1, 1, 1, 2)，会自动广播到 base_grid 的 (n, h, w, 2)
        grid = base_grid + shift

        # 6. 使用 grid_sample 进行采样
        grid_B_C_H_W = F.grid_sample(x_padded, grid, padding_mode="zeros", align_corners=False)

        if input_ndim == 3:
            grid_B_C_H_W = grid_B_C_H_W[0]  # (B,C,H,W) -> (C,H,W)

        return grid_B_C_H_W


class TCLImageDataset(BaseImageDataset):
    def __init__(self,
                 # RoboKit Dataset
                 data_root: str,
                 # Data sequence
                 horizon: int,
                 pad_before: int,
                 pad_after: int,
                 # Data format
                 shape_meta: dict,
                 norm_force_type: str = "quantile",
                 zero_force: bool = False,
                 p_camera_drop: float = 0.,
                 # Others
                 seed: int = 42,
                 val_ratio: float = 0.02,
                 max_train_episodes: int = 90,
                 transform_color_jitter: bool = True,
                 # RoboKit Dataset
                 h5_path: str = None,
                 use_h5: bool = False,
                 # Not used
                 **kwargs
                 ):
        super().__init__()
        # RoboKit Dataset
        self.data_root = data_root
        self.shape_meta = shape_meta
        self.norm_force_type = norm_force_type
        self.zero_force = zero_force
        self.p_camera_drop = p_camera_drop

        self.load_keys = ["rel_actions", "primary_rgb", "gripper_rgb", "robot_obs", "language_text", "force_torque"]
        self.h5_path = h5_path
        self.use_h5 = use_h5
        if not use_h5:
            self.tcl_dataset = TCLDataset(data_root, use_extracted=True, load_keys=self.load_keys)
        else:
            self.tcl_dataset = TCLDatasetHDF5(
                data_root, h5_path,
                use_extracted=True, load_keys=self.load_keys)
        print("[DEBUG] type of tcl_dataset:", type(self.tcl_dataset))
        self.data_meta = self.tcl_dataset.load_meta_from_json(os.path.join(data_root, "statistics.json"))
        self.all_rel_actions = self.tcl_dataset.extracted_data["rel_actions"]
        if "force_torque" in self.load_keys:
            self.tcl_dataset.load_npy_by_key("force_torque")
        self.all_force_torques = self.tcl_dataset.extracted_data.get("force_torque", None)
        self.dataset_stats = self.data_meta["stats"]  # key: `rel_actions`, `robot_obs`, `force_torque`
        self.dataset_total_len = self.data_meta["total_len"]
        self.dataset_action_min = np.array(self.dataset_stats["rel_actions"]["min"])
        self.dataset_action_max = np.array(self.dataset_stats["rel_actions"]["max"])

        # Calculate p01 and p99 for force_torque if available
        if 'force_torque' in self.dataset_stats:
            # Calculate p01 (1%) and p99 (99%) quantiles along the sample dimension (axis=0)
            p01 = np.quantile(self.all_force_torques, q=0.01, axis=0)
            p99 = np.quantile(self.all_force_torques, q=0.99, axis=0)

            # Add the calculated quantiles to the merged statistics dictionary
            self.dataset_stats['force_torque']['p01'] = p01
            self.dataset_stats['force_torque']['p99'] = p99

        self.tasks = self.tcl_dataset.tasks
        self.task_lengths = self.tcl_dataset.task_lengths
        self.ep_fns = self.tcl_dataset.ep_fns
        self.map_index_to_task_id = self.tcl_dataset.map_index_to_task_id

        # Others
        self.seed = seed
        self.val_ratio = val_ratio
        self.max_train_episodes = max_train_episodes

        # Sampling a data sequence
        self.horizon = horizon
        self.pad_before = pad_before
        self.pad_after = pad_after
        self.task_prefix_lengths = [0 for _ in range(len(self.task_lengths))]
        for i in range(1, len(self.task_prefix_lengths)):  # 3 means [0+1+2]
            self.task_prefix_lengths[i] = self.task_prefix_lengths[i - 1] + self.task_lengths[i - 1]

        # Data format and preprocessing
        self.obs_image_shape = shape_meta["obs"]["image"]["shape"]  # [3, H, W]
        self.obs_gripper_shape = shape_meta["obs"]["gripper"]["shape"] if "gripper" in shape_meta["obs"] else None
        self.joint_state_shape = shape_meta["obs"]["joint_state"]["shape"]
        self.force_torque_shape = shape_meta["obs"]["force_torque"]["shape"] if "force_torque" in shape_meta["obs"] else None
        self.action_shape = shape_meta["action"]["shape"]   # [7,]
        obs_image_wh_ratio = float(self.obs_image_shape[2]) / float(self.obs_image_shape[1])  # wh 4:3=16:12=12:9
        transform_list = [
            transforms.ToPILImage(),  # wh 16:9
            transforms.Resize(self.obs_image_shape[1:]),
        ]
        if transform_color_jitter:
            # transforms.RandomResizedCrop(size=self.obs_image_shape[1:], scale=(0.68, 0.82),  # 12/16=0.75
            #                              ratio=(0.9 * obs_image_wh_ratio, 1.1 * obs_image_wh_ratio)),  # not using this would be better?
            transform_list.append(transforms.ColorJitter(brightness=0.05,
                contrast=0.05,
                saturation=0.05,
                hue=0.05))
        transform_list.append(transforms.ToTensor())
        if transform_color_jitter:
            random_shift = RandomShiftsAug(pad=4)
            transform_list.append(random_shift)
        self.obs_image_transform = transforms.Compose(transform_list)  # Similar augmentation params with OCTO

        print(f"[diffusion_policy.dataset.TCLImageDataset] dataset loaded, "
              f"action_min={self.dataset_action_min}, action_max={self.dataset_action_max}, "
              f"zero_force={self.zero_force}, p_camera_drop={self.p_camera_drop}.")

    def get_validation_dataset(self) -> 'TCLImageDataset':
        return self.create_val_dataset(self)

    @classmethod
    def create_val_dataset(cls, instance: 'TCLImageDataset') -> 'TCLImageDataset':
        val_set = cls(
            data_root=instance.data_root,
            horizon=instance.horizon,
            pad_before=instance.pad_before,
            pad_after=instance.pad_after,
            shape_meta=instance.shape_meta,
            norm_force_type=instance.norm_force_type,
            zero_force=instance.zero_force,
            seed=instance.seed,
            val_ratio=instance.val_ratio,
            max_train_episodes=instance.max_train_episodes,
            use_h5=instance.use_h5,  # ori:no need to use h5
            h5_path=instance.h5_path,
        )
        val_set.tcl_dataset.total_length = 64
        return val_set

    def get_normalizer(self, **kwargs) -> LinearNormalizer:
        normalizer = LinearNormalizer()
        obs_keys = self.shape_meta["obs"].keys()
        for obs_key in obs_keys:
            normalizer[obs_key] = EmptyNormalizer.create_identity()
        # normalizer['image'] = EmptyNormalizer.create_identity()
        # normalizer['joint_state'] = EmptyNormalizer.create_identity()
        normalizer['action'] = EmptyNormalizer.create_identity()
        return normalizer

    def __len__(self):
        return len(self.tcl_dataset)

    def __getitem__(self, abs_idx):
        abs_idx = abs_idx % self.__len__()
        obs_data = self._get_obs_data(abs_idx)
        act_data = self._get_act_data(abs_idx)
        item_data = {
            "obs": obs_data,
            "action": act_data,
        }
        return item_data

    def _abs_idx_to_rel_idx(self, abs_idx: int):
        task_id = self.map_index_to_task_id[abs_idx]
        task_len = self.task_lengths[task_id]
        task_prefix_len = self.task_prefix_lengths[task_id]
        rel_idx = abs_idx - task_prefix_len
        assert 0 <= rel_idx < task_len
        return rel_idx, task_id

    def _rel_idx_to_abs_idx(self, rel_idx: int, task_id: int):
        task_prefix_len = self.task_prefix_lengths[task_id]
        task_len = self.task_lengths[task_id]
        if rel_idx < 0:
            abs_idx = None
        elif rel_idx >= task_len:
            abs_idx = None
        else:
            abs_idx = rel_idx + task_prefix_len
            assert task_prefix_len <= abs_idx < task_prefix_len + task_len
        return abs_idx

    def _get_abs_obs_indices(self, abs_now_idx: int):
        # [start_idx, end_idx)
        rel_now_idx, task_id = self._abs_idx_to_rel_idx(abs_now_idx)
        start_idx = rel_now_idx - self.pad_before
        end_idx = rel_now_idx + 1
        rel_indices = list(range(start_idx, end_idx))
        return [self._rel_idx_to_abs_idx(rel_id, task_id) for rel_id in rel_indices]

    def _get_abs_act_indices(self, abs_now_idx: int):
        # [start_idx, end_idx)
        rel_now_idx, task_id = self._abs_idx_to_rel_idx(abs_now_idx)
        start_idx = rel_now_idx
        end_idx = rel_now_idx + self.horizon
        rel_indices = list(range(start_idx, end_idx))
        return [self._rel_idx_to_abs_idx(rel_id, task_id) for rel_id in rel_indices]

    def _get_obs_data(self, abs_idx: int):
        task_id = self.map_index_to_task_id[abs_idx]
        abs_obs_indices = self._get_abs_obs_indices(abs_idx)
        # print(f"[DEBUG] _get_obs_data: abs_idx={abs_idx}, task_id={task_id}, "
        #       f"task_1st_ep={self.task_prefix_lengths[task_id]}, "
        #       f"task_last_ep={self.task_prefix_lengths[task_id] + self.task_lengths[task_id]}"
        #       )
        # print(f"[DEBUG] _get_obs_data: abs_obs_indices={abs_obs_indices} ")
        obs_keys = self.shape_meta["obs"].keys()
        obs_data = {
            k: [] for k in obs_keys
        }
        for idx in abs_obs_indices:
            if idx is None:
                c, h, w = self.obs_image_shape
                zero_rgb = torch.ones((c, h, w)).to(torch.float32) * -1  # all -1
                primary_rgb = zero_rgb
                gripper_rgb = zero_rgb
                tcp_pose = torch.zeros((6,)).to(torch.float32)
                force_torque = torch.zeros((6,)).to(torch.float32)
            else:
                sample_dict = self.tcl_dataset.__getitem__(idx)
                primary_rgb = sample_dict['primary_rgb']  # (H,W,C)
                gripper_rgb = sample_dict['gripper_rgb']  # (H,W,C)
                tcp_pose = joint_state = sample_dict['robot_obs'][:6]  # (6,)
                # Preprocess
                primary_rgb = self.obs_image_transform(primary_rgb)  # (C,H,W), in [0, 1]
                primary_rgb = primary_rgb * 2. - 1.  # in [-1, 1]
                tcp_pose = torch.from_numpy(tcp_pose).to(torch.float32)  # (6,)
                if "gripper" in obs_keys:
                    gripper_rgb = self.obs_image_transform(gripper_rgb)
                    gripper_rgb = gripper_rgb * 2. - 1.
                if "force" in obs_keys:
                    force_torque = sample_dict['force_torque']  # (6,)
                    force_torque = self.norm_state_or_force(
                        force_torque, self.norm_force_type, self.dataset_stats["force_torque"])
                    force_torque = torch.from_numpy(force_torque).to(torch.float32)

            # Randomly dropout camera views, robot_states
            if torch.rand(1).item() < self.p_camera_drop:
                primary_rgb = torch.zeros_like(primary_rgb)
            if "gripper" in obs_keys and torch.rand(1).item() < self.p_camera_drop:
                gripper_rgb = torch.zeros_like(gripper_rgb)
            if torch.rand(1).item() < self.p_camera_drop:
                tcp_pose = torch.zeros_like(tcp_pose)

            obs_data["image"].append(primary_rgb)
            obs_data["joint_state"].append(tcp_pose)
            if "gripper" in obs_keys:
                obs_data["gripper"].append(gripper_rgb)
            if "force" in obs_keys:
                if self.zero_force:
                    force_torque = torch.zeros_like(force_torque)
                obs_data["force"].append(force_torque)
        obs_data = {k: torch.stack(v) for k, v in obs_data.items()}
        # obs_data["image"] = torch.stack(obs_data["image"])  # should be (T,C,H,W)
        # obs_data["joint_state"] = torch.stack(obs_data["joint_state"])  # (T,6)
        return obs_data

    def _get_act_data(self, abs_idx: int):
        task_id = self.map_index_to_task_id[abs_idx]
        abs_act_indices = self._get_abs_act_indices(abs_idx)
        # print(f"[DEBUG] _get_act_data: abs_idx={abs_idx}, task_id={task_id}, "
        #       f"task_1st_ep={self.task_prefix_lengths[task_id]}, "
        #       f"task_last_ep={self.task_prefix_lengths[task_id] + self.task_lengths[task_id]}"
        #       )
        # print(f"[DEBUG] _get_obs_data: abs_obs_indices={abs_act_indices} ")
        act_data = []
        for idx in abs_act_indices:
            if idx is None:
                zero_act = np.zeros(self.action_shape).astype(np.float32)
                act_data.append(zero_act)
            else:
                rel_action = self.all_rel_actions[idx]  # (7,)
                act_data.append(rel_action)
        act_data = np.stack(act_data)  # (T,7), in [act_min, act_max]
        act_data = (act_data - self.dataset_action_min) / (self.dataset_action_max - self.dataset_action_min)  # norm here, in [0,1]
        return act_data * 2. - 1.  # in [-1,1]

    @staticmethod
    def norm_state_or_force(in_data: np.ndarray, norm_type: str, meta_data: dict):
        """
        `robot_obs`: (...,14)
            tcp pos (3), tcp ori (3), gripper width (1), joint_states (6) in rad, gripper_action (1)
        `force_torque`: (...,6)
        """
        D = in_data.shape[-1]
        if norm_type == "minmax":
            dataset_min = np.array(meta_data['min'])[:D]
            dataset_max = np.array(meta_data['max'])[:D]
            out_data = (in_data - dataset_min) / (dataset_max - dataset_min)  # norm here, in [0,1]
            out_data = out_data * 2. - 1.  # in [-1,1]
        elif norm_type == "mean":
            dataset_mean = np.array(meta_data['mean'])[:D]
            dataset_std = np.array(meta_data['std'])[:D]
            out_data = (in_data - dataset_mean) / dataset_std
        elif norm_type == "quantile":
            dataset_p01 = np.array(meta_data['p01'])[:D]
            dataset_p99 = np.array(meta_data['p99'])[:D]
            in_data = np.clip(in_data, dataset_p01, dataset_p99)  # different from minmax
            out_data = (in_data - dataset_p01) / (dataset_p99 - dataset_p01)  # norm here, in [0,1]
            out_data = out_data * 2. - 1.  # in [-1,1]
        else:
            assert norm_type == "identity"
            out_data = in_data
        return out_data


class TCLMasterSlaveDataset(BaseImageDataset):
    def __init__(
            self,
            # Master dataset config
            data_root: str,
            # Slave dataset config (must align one-to-one)
            slave_data_roots: List[str],
            slave_h5_paths: List[str],
            # Sequence config
            horizon: int,
            pad_before: int,
            pad_after: int,
            # Data format
            shape_meta: dict,
            norm_force_type: str = "quantile",
            zero_force: bool = False,
            p_camera_drop: float = 0.,
            # Others
            seed: int = 42,
            val_ratio: float = 0.02,
            max_train_episodes: int = 90,
            transform_color_jitter: bool = True,
            # H5 config for master
            h5_path: str = None,
            use_h5: bool = False,
    ):
        super().__init__()
        slave_data_roots = slave_data_roots or []
        slave_h5_paths = slave_h5_paths or []
        if len(slave_data_roots) != len(slave_h5_paths):
            raise ValueError(
                f"slave_data_roots and slave_h5_paths must have the same length, "
                f"got {len(slave_data_roots)} vs {len(slave_h5_paths)}."
            )
        if use_h5 and h5_path is None:
            raise ValueError("`h5_path` for master dataset must be provided when use_h5=True.")
        if use_h5 and any(p is None for p in slave_h5_paths):
            raise ValueError("All `slave_h5_paths` must be provided when use_h5=True.")

        self.master_dataset = TCLImageDataset(
            data_root=data_root,
            horizon=horizon,
            pad_before=pad_before,
            pad_after=pad_after,
            shape_meta=shape_meta,
            norm_force_type=norm_force_type,
            zero_force=zero_force,
            p_camera_drop=p_camera_drop,
            seed=seed,
            val_ratio=val_ratio,
            max_train_episodes=max_train_episodes,
            transform_color_jitter=transform_color_jitter,
            h5_path=h5_path,
            use_h5=use_h5,
        )
        self.slave_datasets: List[TCLImageDataset] = []
        for slave_root, slave_h5_path in zip(slave_data_roots, slave_h5_paths):
            slave_ds = TCLImageDataset(
                data_root=slave_root,
                horizon=horizon,
                pad_before=pad_before,
                pad_after=pad_after,
                shape_meta=shape_meta,
                norm_force_type=norm_force_type,
                zero_force=zero_force,
                p_camera_drop=p_camera_drop,
                seed=seed,
                val_ratio=val_ratio,
                max_train_episodes=max_train_episodes,
                transform_color_jitter=transform_color_jitter,
                h5_path=slave_h5_path,
                use_h5=use_h5,
            )
            self.slave_datasets.append(slave_ds)

        self.datasets = [self.master_dataset] + self.slave_datasets
        self.concat_dataset = ConcatDataset(self.datasets)
        self._sync_master_stats_to_slaves()

        # Keep compatibility with TCLImageDataset style attrs
        self.data_root = data_root
        self.h5_path = h5_path
        self.use_h5 = use_h5
        self.slave_data_roots = slave_data_roots
        self.slave_h5_paths = slave_h5_paths

        self.shape_meta = shape_meta
        self.horizon = horizon
        self.pad_before = pad_before
        self.pad_after = pad_after
        self.norm_force_type = norm_force_type
        self.zero_force = zero_force
        self.p_camera_drop = p_camera_drop
        self.seed = seed
        self.val_ratio = val_ratio
        self.max_train_episodes = max_train_episodes
        self.transform_color_jitter = transform_color_jitter

        print(
            f"[diffusion_policy.dataset.TCLMasterSlaveDataset] "
            f"loaded with 1 master + {len(self.slave_datasets)} slaves, "
            f"total_len={len(self.concat_dataset)}."
        )

    def _sync_master_stats_to_slaves(self):
        self.dataset_stats = self.master_dataset.dataset_stats
        self.dataset_total_len = self.master_dataset.dataset_total_len
        self.dataset_action_min = self.master_dataset.dataset_action_min
        self.dataset_action_max = self.master_dataset.dataset_action_max

        for slave_ds in self.slave_datasets:
            slave_ds.dataset_stats = self.dataset_stats
            slave_ds.dataset_action_min = self.dataset_action_min
            slave_ds.dataset_action_max = self.dataset_action_max

    def get_validation_dataset(self):
        val_set = copy.copy(self)
        val_set.master_dataset = self.master_dataset.get_validation_dataset()
        val_set.slave_datasets = [ds.get_validation_dataset() for ds in self.slave_datasets]
        val_set.datasets = [val_set.master_dataset] + val_set.slave_datasets
        val_set.concat_dataset = ConcatDataset(val_set.datasets)
        val_set._sync_master_stats_to_slaves()
        return val_set

    def get_normalizer(self, **kwargs) -> LinearNormalizer:
        return self.master_dataset.get_normalizer(**kwargs)

    def __len__(self):
        return len(self.concat_dataset)

    def __getitem__(self, idx):
        idx = idx % self.__len__()
        ds_idx = bisect.bisect_right(self.concat_dataset.cumulative_sizes, idx)
        start = 0 if ds_idx == 0 else self.concat_dataset.cumulative_sizes[ds_idx - 1]
        local_idx = idx - start

        item = self.datasets[ds_idx][local_idx]
        item["dataset_idx"] = torch.tensor(ds_idx, dtype=torch.long)
        item["local_idx"] = torch.tensor(local_idx, dtype=torch.long)
        item["is_master"] = torch.tensor(ds_idx == 0, dtype=torch.bool)
        return item


if __name__ == "__main__":
    from robokit.debug_utils.printer import print_batch

    dataset = TCLImageDataset(
        data_root="/home/geyuan/datasets/TCL/collected_data_0425",
        horizon=16, pad_before=1, pad_after=7,
        shape_meta={
            "obs": {
                "image": {
                    "shape": [3, 96, 96],
                    "type": "rgb"
                },
                "joint_state": {
                    "shape": [6],
                    "type": "low_dim"
                }
            },
            "action": {
                "shape": [7,]
            }
        }
    )
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=8, shuffle=True)
    for idx, batch in enumerate(dataloader):
        print_batch(f"tcl_dataset@{idx}", batch)

        print(batch['action'])
        exit()

        if idx >= 5:
            exit()


# TODO: class MergeTCLImageDataset
# params: data_root -> data_roots
# params: h5_path -> h5_paths (should be consistent with data_roots)


class MergeTCLImageDataset(BaseImageDataset):
    """Wrap multiple RoboKit data roots into a single training dataset.

    Each root is an independent RoboKit directory (containing per-frame ``*.npz``
    files organised in task/episode sub-folders). The wrapper constructs one
    :class:`TCLImageDataset` per root, concatenates their samples into a single
    global index space, and serves them through a unified ``__getitem__``.

    Action / force normalisation statistics are taken from ``stats_source``:
      - ``"first"``  : use sub[0]'s statistics (this is the recommended setting
        for incremental learning, matching the note in ``reverse_dp_force.yaml``
        that "模型仍要从最原始的数据集中读取 stats 进行归一化").
      - ``"merged"`` : recompute min/max/mean/std over the concatenation of all
        sub-datasets' ``extracted/rel_actions.npy`` and ``force_torque``.
      - a path string : load statistics from the specified ``statistics.json``.

    ``h5_paths`` must be list-aligned with ``data_roots``; entries may be
    ``None`` to force that specific root to fall back to npz loading.
    """

    def __init__(self,
                 data_roots,                     # list[str]
                 h5_paths=None,                  # list[str | None] | None
                 horizon: int = 16,
                 pad_before: int = 0,
                 pad_after: int = 0,
                 shape_meta: dict = None,
                 norm_force_type: str = "quantile",
                 seed: int = 42,
                 val_ratio: float = 0.02,
                 max_train_episodes: int = 90,
                 transform_color_jitter: bool = True,
                 use_h5: bool = True,
                 stats_source: str = "first",
                 ):
        super().__init__()
        assert len(data_roots) >= 1, "MergeTCLImageDataset needs at least one data root"
        if h5_paths is None:
            h5_paths = [None] * len(data_roots)
        assert len(h5_paths) == len(data_roots), \
            f"h5_paths ({len(h5_paths)}) must align with data_roots ({len(data_roots)})"

        self.data_roots = list(data_roots)
        self.h5_paths = list(h5_paths)
        self.shape_meta = shape_meta
        self.norm_force_type = norm_force_type
        self.horizon = horizon
        self.pad_before = pad_before
        self.pad_after = pad_after
        self.seed = seed
        self.val_ratio = val_ratio
        self.max_train_episodes = max_train_episodes
        self.transform_color_jitter = transform_color_jitter
        self.use_h5 = use_h5
        self.stats_source = stats_source

        # 1. Build each sub-dataset with its own stats (we will overwrite later).
        self.subs = []
        for data_root, h5_path in zip(self.data_roots, self.h5_paths):
            sub_use_h5 = bool(use_h5 and (h5_path is not None))
            sub = TCLImageDataset(
                data_root=data_root,
                h5_path=h5_path,
                use_h5=sub_use_h5,
                horizon=horizon,
                pad_before=pad_before,
                pad_after=pad_after,
                shape_meta=shape_meta,
                norm_force_type=norm_force_type,
                seed=seed,
                val_ratio=val_ratio,
                max_train_episodes=max_train_episodes,
                transform_color_jitter=transform_color_jitter,
            )
            self.subs.append(sub)

        # 2. Resolve merged statistics.
        self.dataset_stats, self.dataset_total_len = self._resolve_stats(stats_source)
        self.dataset_action_min = np.array(self.dataset_stats["rel_actions"]["min"])
        self.dataset_action_max = np.array(self.dataset_stats["rel_actions"]["max"])

        # 3. Broadcast the unified stats to every sub (keeps their internal
        #    normalisation path consistent, so delegating __getitem__ yields
        #    actions in the same [-1, 1] space regardless of which sub served
        #    the sample).
        for sub in self.subs:
            sub.dataset_stats = copy.deepcopy(self.dataset_stats)
            sub.dataset_action_min = self.dataset_action_min
            sub.dataset_action_max = self.dataset_action_max

        # 4. Build the global index by concatenating sub-dataset ranges.
        self.sub_lengths = [len(sub) for sub in self.subs]
        self.cum_offsets = [0]
        for L in self.sub_lengths:
            self.cum_offsets.append(self.cum_offsets[-1] + L)
        self.total_len = self.cum_offsets[-1]

        # 5. Shared shape cache (matches TCLImageDataset for downstream code).
        self.action_shape = shape_meta["action"]["shape"]
        self.obs_image_shape = shape_meta["obs"]["image"]["shape"]

        print(f"[MergeTCLImageDataset] {len(self.subs)} roots merged, "
              f"per-root lengths={self.sub_lengths}, total={self.total_len}, "
              f"stats_source={stats_source}")

    # ---------------------------------------------------------------- stats
    def _resolve_stats(self, stats_source: str):
        if stats_source == "first":
            meta = self.subs[0].data_meta
            return copy.deepcopy(meta["stats"]), int(meta["total_len"])

        if stats_source == "merged":
            merged_stats = {}
            total_len = 0
            # rel_actions / force_torque live as concatenable arrays on each sub.
            rel_actions_all = np.concatenate(
                [sub.all_rel_actions for sub in self.subs], axis=0)
            merged_stats["rel_actions"] = _compute_min_max_mean_std(rel_actions_all)
            # force_torque may not exist on all subs, but current pipeline
            # always loads it through TCLImageDataset.
            ft_all = np.concatenate(
                [np.asarray(sub.all_force_torques) for sub in self.subs], axis=0)
            merged_stats["force_torque"] = _compute_min_max_mean_std(ft_all)
            p01 = np.quantile(ft_all, q=0.01, axis=0)
            p99 = np.quantile(ft_all, q=0.99, axis=0)
            merged_stats["force_torque"]["p01"] = p01.tolist() \
                if isinstance(p01, np.ndarray) else p01
            merged_stats["force_torque"]["p99"] = p99.tolist() \
                if isinstance(p99, np.ndarray) else p99
            # robot_obs: reuse sub[0] stats (only used for optional norm)
            if "robot_obs" in self.subs[0].dataset_stats:
                merged_stats["robot_obs"] = copy.deepcopy(
                    self.subs[0].dataset_stats["robot_obs"])
            total_len = sum(int(sub.data_meta["total_len"]) for sub in self.subs)
            return merged_stats, total_len

        # Otherwise treat stats_source as a path to a statistics.json
        if not os.path.isabs(stats_source):
            stats_source = os.path.join(self.data_roots[0], stats_source)
        with open(stats_source, "r") as fp:
            blob = json.load(fp)
        return copy.deepcopy(blob["stats"]), int(blob.get("total_len", 0))

    # ------------------------------------------------------------- dispatch
    def _locate(self, abs_idx: int):
        """Map a global index to (sub_id, local_idx)."""
        abs_idx = int(abs_idx) % self.total_len
        # linear scan is fine, len(self.subs) is small (<=~4 in practice)
        for sub_id, offset in enumerate(self.cum_offsets[:-1]):
            if abs_idx < self.cum_offsets[sub_id + 1]:
                return sub_id, abs_idx - offset
        raise IndexError(abs_idx)

    def __len__(self):
        return self.total_len

    def __getitem__(self, abs_idx):
        sub_id, local_idx = self._locate(abs_idx)
        return self.subs[sub_id][local_idx]

    # -------------------------------------------------------- DP integration
    def get_normalizer(self, **kwargs) -> LinearNormalizer:
        # All sub-datasets share the identity normalizer path; reuse sub[0].
        return self.subs[0].get_normalizer(**kwargs)

    def get_validation_dataset(self):
        return self.create_val_dataset(self)

    @classmethod
    def create_val_dataset(cls, instance: 'MergeTCLImageDataset'):
        val_set = cls(
            data_roots=instance.data_roots,
            h5_paths=instance.h5_paths,
            horizon=instance.horizon,
            pad_before=instance.pad_before,
            pad_after=instance.pad_after,
            shape_meta=instance.shape_meta,
            norm_force_type=instance.norm_force_type,
            seed=instance.seed,
            val_ratio=instance.val_ratio,
            max_train_episodes=instance.max_train_episodes,
            transform_color_jitter=instance.transform_color_jitter,
            use_h5=instance.use_h5,
            stats_source=instance.stats_source,
        )
        # Shrink each sub to a small slice for validation, same convention as
        # TCLImageDataset.create_val_dataset.
        for sub in val_set.subs:
            sub.tcl_dataset.total_length = 64
        val_set.sub_lengths = [len(sub) for sub in val_set.subs]
        val_set.cum_offsets = [0]
        for L in val_set.sub_lengths:
            val_set.cum_offsets.append(val_set.cum_offsets[-1] + L)
        val_set.total_len = val_set.cum_offsets[-1]
        return val_set


def _compute_min_max_mean_std(arr: np.ndarray) -> dict:
    """Return min/max/mean/std along axis 0, compatible with statistics.json."""
    arr = np.asarray(arr)
    flat = arr.reshape(arr.shape[0], -1)
    mn = flat.min(axis=0)
    mx = flat.max(axis=0)
    # Guard against constant columns (matches RoboKit behaviour).
    eq = (mn == mx)
    mx = mx.copy()
    mx[eq] += 1e-6
    mean = flat.mean(axis=0)
    std = flat.std(axis=0)
    std = np.maximum(std, 1e-8)
    return {
        "min": mn.tolist(),
        "max": mx.tolist(),
        "mean": mean.tolist(),
        "std": std.tolist(),
    }
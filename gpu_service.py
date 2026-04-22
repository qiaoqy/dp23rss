import base64
import io
import os
from functools import lru_cache
from pathlib import Path
from typing import List, Tuple, Dict, Optional, Union
import copy
import shutil

import numpy as np
import pydantic
from PIL import Image
from fastapi import FastAPI

import json
import torch
from torchvision.transforms import transforms

from diffusion_policy.dataset.tcl_dataset import TCLImageDataset, TCLDatasetHDF5
# from robokit.service.service_connector import ServiceConnector
from robokit.connects.protocols import StepRequestFromEvaluator, StepRequestFromPolicy
from robokit.debug_utils.time_profiler import global_time_profiler as gtp


""" How to use me?
conda activate robodiff
cd code/dp23rss_fork
export PYTHONPATH=~/code/dp23rss_fork
CUDA_VISIBLE_DEVICES=4 uvicorn gpu_service:gpu_app --port 6070
"""
gpu_app = FastAPI()
max_cache_action = 32

log_time = "2026.04.21-22.58.02"
w_idx = -1

map_time_to_dataset = {
    "2025.11.08-10.27.18": "1021_sweep_bean",
    "2025.11.08-01.55.12": "1024_eggs_pick_place",
    "2025.11.08-10.28.20": "1024_pour_water",
    "2025.11.08-10.25.33": "1024_wipe_white_board",
    "2025.12.01-22.51.04": "1201_wipe_blackboard",
    "2025.12.03-23.52.04": "1201_pour_water",
    "2025.12.11-22.31.16": "1201_banana",
    "2025.12.11-23.08.31": "1201_pepper",
    "2025.12.14-20.22.43": "1201_pot",
    "2025.12.15-17.58.23": "1201_coffee",
    "2026.01.15-21.40.54": "1201_screw_bulb",
    "2026.01.17-00.06.29": "1201_screw_bulb_turn_off",
    "2026.02.10-00.06.53": "0209_tower_boby",
    "2026.03.16-21.09.19": "0209_tower_boby_hard",
    "2026.03.18-21.22.07": "0209_tower_boby_easy",
    "2026.04.18-00.45.25": "0417_put_mouse",
    "2026.04.18-00.47.33": "0417_ethernet",
    "2026.04.18-00.48.06": "0417_greenyellowred",
    "2026.04.21-01.55.22": "0209_tower_boby_easy",
    "2026.04.21-01.31.36": "0417_put_mouse",
    "2026.04.21-22.57.11": "0417_test_tube",
    "2026.04.21-22.58.02": "0417_french_press",
}
# train_project_dir = f"/home/geyuan/code/dp23rss_fork/data/outputs/{log_time}_train_diffusion_transformer_hybrid_pusht_images"
train_project_dir = f"/home/geyuan/code/dp23rss_fork/data/outputs/{log_time}_train_diffusion_transformer_hybrid_pusht_image"
train_project_dir = train_project_dir.replace('-', '/')
dataset_name = "pot_object"  # shovel; pot, pot_light; pepper
dataset_name = map_time_to_dataset.get(log_time, dataset_name)

if "2025.05.11" in log_time or "2025.05.13" in log_time:
    dataset_dir = "collected_data_0507"
elif dataset_name == "shovel":
    dataset_dir = "collected_data_0514_shovel_source"
elif dataset_name == "pot":
    dataset_dir = "0627_pot_source"
elif dataset_name == "pot_light":
    dataset_dir = "0627_pot_light"
elif dataset_name == "pot_object":
    dataset_dir = "0627_pot_object"
elif dataset_name == "pepper":
    dataset_dir = "0704_pepper_source"
else:
    dataset_dir = dataset_name
    print(f"[Warning] Using {dataset_name} for log_time={log_time}.")

# Load dataset statistics
dataset_statistics_file = f"/home/geyuan/datasets/TCL/{dataset_dir}/statistics.json"
train_project_statistics_file = os.path.join(train_project_dir, "statistics.json")
if os.path.exists(dataset_statistics_file):
    with open(dataset_statistics_file, 'r') as json_file:
        statistics = json.load(json_file)
        dataset_stats = statistics["stats"]
        datasets_total_len = statistics["total_len"]
        dataset_action_min = np.array(dataset_stats["rel_actions"]["min"])
        dataset_action_max = np.array(dataset_stats["rel_actions"]["max"])

        data_root = f"/home/geyuan/datasets/TCL/{dataset_dir}"
        h5_path = f"/home/geyuan/datasets/TCL/hdf5/{dataset_dir}_240p.h5"
        tcl_hdf5_dataset = TCLDatasetHDF5(
            data_root, h5_path,
            use_extracted=True,
            load_keys=["rel_actions", "primary_rgb", "gripper_rgb", "robot_obs", "language_text", "force_torque"]
        )
        all_force_torques = tcl_hdf5_dataset.dsets["force_torque"]

        # Calculate p01 and p99 for force_torque if available
        if 'force_torque' in dataset_stats:
            # Calculate p01 (1%) and p99 (99%) quantiles along the sample dimension (axis=0)
            p01 = np.quantile(all_force_torques, q=0.01, axis=0)
            p99 = np.quantile(all_force_torques, q=0.99, axis=0)

            # Add the calculated quantiles to the merged statistics dictionary
            dataset_stats['force_torque']['p01'] = p01.tolist()
            dataset_stats['force_torque']['p99'] = p99.tolist()

    # Dump updated statistics back to the JSON file
    if not os.path.exists(train_project_statistics_file):
        with open(train_project_statistics_file, 'w') as json_file:
            json.dump(statistics, json_file, indent=4)
        print("[Info] Dumped updated statistics.json to train_project_dir.")

# Load statistics from train project dir (to be compatible with ITX deployment)
assert os.path.exists(train_project_statistics_file), "[gpu_service] statistics.json not found in train_project_dir."
with open(train_project_statistics_file, 'r') as json_file:
    statistics = json.load(json_file)
    dataset_stats = statistics["stats"]
    datasets_total_len = statistics["total_len"]
    dataset_action_min = np.array(dataset_stats["rel_actions"]["min"])
    dataset_action_max = np.array(dataset_stats["rel_actions"]["max"])
    dataset_stats['force_torque']['p01'] = np.array(dataset_stats["force_torque"]['p01'])
    dataset_stats['force_torque']['p99'] = np.array(dataset_stats["force_torque"]['p99'])


@lru_cache()
def get_agent(device: str):
    ## Op1. Debug model, sleep only
    # model = DebugModel(sleep_duration=100)
    ## Op2. Replay model, load action data and sleep
    # model = ReplayModel(sleep_duration=25,
    #                     replay_root="/home/geyuan/datasets/TCL/collected_data")

    import hydra
    from omegaconf import OmegaConf

    # 1. Load hydra config
    train_dir = train_project_dir
    hydra_config_path = os.path.join(train_dir, ".hydra/config.yaml")
    hydra_config = OmegaConf.load(hydra_config_path)
    model = hydra.utils.instantiate(hydra_config.policy)
    print(type(model))

    # 2. Load weights
    weight_paths = os.listdir(os.path.join(train_dir, "checkpoints"))
    weight_paths = list(filter(lambda x: x.endswith(".ckpt"), weight_paths))
    weight_paths.sort()
    print(weight_paths)
    weight_path = os.path.join(train_dir, "checkpoints", weight_paths[w_idx])
    weight = torch.load(weight_path, map_location="cpu", weights_only=False)['state_dicts']
    weight = weight['model']
    # for k, v in weight.items():
    #     print(k, v.shape)

    model.load_state_dict(weight)
    model = model.to(device).eval()
    print(f"[get_agent] model loaded from: {weight_path}")

    # 3. Other settings
    model.infer_frame_idx = 0

    return model, hydra_config, weight_path


class SharedMemoryPool:
    """
    专为 Evaluator 和 Policy 通信设计的显式内存池
    """

    def __init__(self):
        self._buffers: Dict[str, np.ndarray] = {}
        self._shapes: Dict[str, Tuple] = {}

    def get_or_allocate(self, key_to_shape: Union[str, Dict[str, Tuple]],
                        dtype=np.float32) -> Union[np.ndarray, Dict[str, np.ndarray]]:
        """
        检查并返回 Buffer。如果 Shape 发生变化，会自动重新分配。
        """
        if isinstance(key_to_shape, str):
            return self._buffers[key_to_shape]
        assert isinstance(key_to_shape, dict), "key_to_shape must be str or dict."
        for key, shape in key_to_shape.items():
            # 如果 key 不存在，或者 shape 发生了改变，才重新分配内存
            if key not in self._buffers or self._shapes.get(key) != shape:
                self._buffers[key] = np.empty(shape, dtype=dtype)
                self._shapes[key] = shape
                print(f"[DEBUG] SharedMemoryPool: set key `{key}` to shape {shape}.")

        return {k: self._buffers[k] for k in key_to_shape.keys()}

    def clear(self):
        """支持手动释放内存"""
        self._buffers.clear()
        self._shapes.clear()


@lru_cache()
def get_mem_pool():
    return SharedMemoryPool()


@gpu_app.get("/")
def read_root():
    return {"message": "Hello, World!"}


@gpu_app.get("/init")
def model_init():
    return {"message": "Hello, World!", "max_cache_action": max_cache_action}


@gpu_app.get("/reset")
def model_reset():
    agent, image_shape, _ = get_agent("cuda")
    agent.reset()
    return {"max_cache_action": max_cache_action}


@gpu_app.post("/step")
def model_step(step_request: StepRequestFromEvaluator):
    agent, hydra_config, weight_path = get_agent("cuda")  # shape:[C,H,W]
    mem_buffer = get_mem_pool()
    print("[gpu_service] Using cached ckpt from: None. Model type:", type(agent), weight_path)

    # 1. Decode observation from received request
    image_shape_C_H_W = hydra_config.image_shape
    max_cache_action = step_request.max_cache_action
    num_camera_views = step_request.num_camera_views
    mem_buffer.get_or_allocate({
        "gt_video": (1, num_camera_views * max_cache_action,
                     image_shape_C_H_W[1], image_shape_C_H_W[2], image_shape_C_H_W[0])}, dtype=np.uint8)

    # [] Parse input observation
    with gtp("decode", group="process_request"):
        video_buffer = mem_buffer.get_or_allocate("gt_video")
        step_data = step_request.decode_to_raw_buffer(out_video_buffer=video_buffer)
    instruction_text = step_data["instruction"]
    stage_flag = step_data["stage_flag"]
    gt_video = step_data["gt_video"]  # (B,V*Ts,H,W,3) uint8, Ts can be larger than v1
    tcp_state = step_data["tcp_state"]  # (B,Ts,D+6) float32 or None, NOTE: includes force data

    with gtp("cpu_type_convert", group="process_request"):
        B, Ts, D_plus6 = tcp_state.shape
        gt_video = torch.from_numpy(gt_video).to("cuda").float() / 127.5 - 1.  # (B,V*Ts,H,W,3), in [-1,1]
        gt_view0_B_T_C_H_W = gt_video[:, :Ts].permute(0, 1, 4, 2, 3)  # (B,T,C,H,W)
        gt_view1_B_T_C_H_W = gt_video[:, Ts:].permute(0, 1, 4, 2, 3)  # (B,T,C,H,W)

        # instruction_text = step_request.instruction
        # joint_state = step_request.joint_state
        force_B_T_D = tcp_state[:, :, -6:].astype(np.float32)  # (B,T,6)
        tcp_pose_B_T_D = tcp_state[:, :, :6].astype(np.float32)  # (B,T,6)

    with gtp("cpu_norm_input", group="process_request"):
        # Norm input states
        force_B_T_D = TCLImageDataset.norm_state_or_force(
            force_B_T_D, norm_type="quantile", meta_data=dataset_stats['force_torque']
        )
        zero_force = getattr(hydra_config.task.dataset, "zero_force", False)
        if zero_force:
            force_B_T_D = force_B_T_D * 0.

    # joint_state = torch.from_numpy(np.array(joint_state)).to("cuda").unsqueeze(0)  # (B,T,6)
    obs_dict = {
        "joint_state": torch.from_numpy(tcp_pose_B_T_D).to("cuda"),  # should be (B,T,6)
        "force": torch.from_numpy(force_B_T_D).to("cuda"),  # should be (B,T,6)
    }

    if True or agent.infer_frame_idx % max_cache_action == 0:  # always enter
        primary_img = gt_view0_B_T_C_H_W[:, -1, :, :, :].unsqueeze(1)  # (B,1,C,H,W)
        gripper_img = gt_view1_B_T_C_H_W[:, -1, :, :, :].unsqueeze(1)  # (B,1,C,H,W)

        print("[DEBUG] primary_img shape:", primary_img.shape)

        obs_dict["image"] = primary_img  # should be (B,T,C,H,W)
        if "gripper" in hydra_config.shape_meta["obs"]:
            obs_dict["gripper"] = gripper_img

    # cond = {
    #     "lang_text": instruction_text,
    #     "proprioception": joint_state,
    # }

    # 3.a Model inference
    # action_idx = agent.infer_frame_idx % max_cache_action
    # if action_idx == 0:
    with gtp("predict_action", group="model_infer"):
        with torch.no_grad():  # always enter
            action = agent.predict_action(obs_dict)['action_pred']
            action = action[0, :]  # remove batch_dim, (T,D)

    # 3.b Postprocess
    # print(action.shape, action.min(dim=0)[0], action.max(dim=0)[0])
    action = (action * 0.5 + 0.5).cpu()  # in [0,1]
    action = action.clamp(0., 1.)
    action = action * (dataset_action_max - dataset_action_min) + dataset_action_min

    with gtp("binarize_action", group="model_infer"):
        cache_action = action
        for act_idx in range(cache_action.shape[0]):
            if cache_action[act_idx, 6:] >= 0.5:
                cache_action[act_idx, 6:] = 1
            else:
                cache_action[act_idx, 6:] = 0

    gtp.step()
    agent.infer_frame_idx += 1
    # return {"action": cache_action.detach().numpy().tolist()}
    out_action = cache_action[None, :, :].numpy()  # (1,T,D)
    print(f"[DEBUG] index={agent.infer_frame_idx}, gpu_service output action shape:", out_action.shape)
    with gtp("encode", group="process_request"):
        request_to_evaluator = StepRequestFromPolicy.encode_from_raw(action=out_action)

    if agent.infer_frame_idx % 15 == 0:
        gtp.report()
    return request_to_evaluator.model_dump(mode="json")


if __name__ == "__main__":
    import time
    agent = get_agent("cuda")

    zero_rgb = np.zeros((2, 480, 848, 3), dtype=np.uint8)  # (T,H,W,C)

    debug_request = StepRequestWithObservation(
        primary_rgb=ServiceConnector.img_np_to_base64(zero_rgb),
        gripper_rgb=ServiceConnector.img_np_to_base64(zero_rgb),
        instruction="none",
        joint_state=[[0.] * 6] * 2,
    )
    pred_action = model_step(debug_request)

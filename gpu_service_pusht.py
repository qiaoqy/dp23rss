import base64
import io
import os
from functools import lru_cache
from pathlib import Path
from typing import List, Tuple, Dict
import copy
import shutil
import yaml

import hydra
from omegaconf import OmegaConf
import numpy as np
import pydantic
from PIL import Image
from fastapi import FastAPI

import json
import torch
from torchvision.transforms import transforms

# from diffusion_policy.dataset.libero_dataset import LiberoFTDataset
# from robokit.service.service_connector import ServiceConnector
from robokit.connects.protocols import StepRequestFromEvaluator, StepRequestFromPolicy
from robokit.debug_utils.printer import print_batch


""" How to use me?
conda activate robodiff
cd code/dp23rss_fork
export PYTHONPATH=~/code/dp23rss_fork
CUDA_VISIBLE_DEVICES=6 uvicorn gpu_service_pusht:gpu_app --port 7076
"""
gpu_app = FastAPI()
max_cache_action = 8  # ori:32

# log_time = "2026.04.09-21.03.16"  # `None`
# log_time = "2026.04.09-21.05.34"  # `light`
# log_time = "2026.04.09-21.04.18"  # `goal`
# log_time = "2026.04.07-19.27.27"  # `None+goal+light+texture+block`

# log_time = "2026.04.15-16.41.59"  # `none`, vis_encoder from `none`
# log_time = "2026.04.15-16.44.03"  # `light`, vis_encoder from `none`
# log_time = "2026.04.15-16.59.43"  # `goal`, vis_encoder from `none`
# log_time = "2026.04.15-17.39.01"  # `block`, vis_encoder from `none`

# log_time = "2026.04.14-02.22.18"  # `none`, vis_encoder from `predict_all`
# log_time = "2026.04.14-11.19.46"  # `goal`, vis_encoder from `predict_all`
# log_time = "2026.04.15-01.26.00"  # `light`, vis_encoder from `predict_all`
# log_time = "2026.04.15-01.24.54"  # `block`, vis_encoder from `predict_all`

log_time = "2026.04.17-02.40.03"  # `none+goal+light`, moe from scratch
# log_time = "2026.04.11-00.08.53"  # `none+goal+light`, moe from teacher + freeze FFN
# log_time = "2026.04.16-17.04.04"  # `none+goal+light`, moe from teacher + finetune

w_idx = -2



def load_dataset_fields(yaml_path: str):
    with open(yaml_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    shape_meta = cfg.get("shape_meta", None)
    ds = cfg["task"]["dataset"]
    hdf5_fns = ds.get("hdf5_fns", [])
    dataset_root = ds.get("dataset_root", None)
    dataset_subname = ds.get("dataset_subname", None)
    return hdf5_fns, dataset_root, dataset_subname, shape_meta


train_project_dir = f"/home/geyuan/code/dp23rss_fork/data/outputs/{log_time}_train_diffusion_transformer_hybrid_pusht256"
train_project_dir = train_project_dir.replace('-', '/')
train_yaml_path = os.path.join(train_project_dir, ".hydra/config.yaml")
assert os.path.exists(train_yaml_path), f"[gpu_service_pusht] train_yaml_path not found: {train_yaml_path}"

hdf5_fns, dataset_root, dataset_subname, shape_meta = load_dataset_fields(train_yaml_path)
print(f"[INFO] Loaded dataset fields from {train_yaml_path}:")
print("  hdf5_fns:", hdf5_fns)
print("  dataset_root:", dataset_root)
print("  dataset_subname:", dataset_subname)
print("  shape_meta:", shape_meta)

# NO Need to: Load dataset statistics and save to train_project_dir if not exists

# No Need to: Load statistics from train project dir (to be compatible with ITX deployment)
"""
[DEBUG] PushTImageDataset: Dict, keys=['obs', 'action']
--obs: Dict, keys=['image', 'agent_pos']
----image, <class 'torch.Tensor'>, shape=torch.Size([10, 3, 256, 256]), min=-0.4902, max=1.0000, dtype=torch.float64
----agent_pos, <class 'torch.Tensor'>, shape=torch.Size([10, 2]), min=-0.1186, max=0.6088, dtype=torch.float32
--action, <class 'torch.Tensor'>, shape=torch.Size([10, 2]), min=-0.2812, max=0.6367, dtype=torch.float32
"""
dataset_action_max = np.array([512, 512], dtype=np.float32)  # hardcoded for pusht
dataset_action_min = np.array([0, 0], dtype=np.float32)
dataset_states_max = np.array([512, 512], dtype=np.float32)
dataset_states_min = np.array([0, 0], dtype=np.float32)


@lru_cache()
def get_agent(device: str, use_ema: bool = True):
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
    if not use_ema:
        weight = weight['model']
    else:
        weight = weight['ema_model']

    model.load_state_dict(weight)
    model = model.to(device).eval()
    print(f"[get_agent] model loaded from: {weight_path}, use_ema={use_ema}")

    # 3. Other settings
    model.infer_frame_idx = 0

    return model, hydra_config, weight_path


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
    print("[gpu_service] Using cached ckpt from: None. Model type:", type(agent), weight_path)

    # 1. Decode observation from received request
    image_shape = hydra_config.image_shape

    # [] Parse input observation
    step_data = step_request.decode_to_raw()
    instruction_text = step_data["instruction"]
    stage_flag = step_data["stage_flag"]
    gt_video = step_data["gt_video"]  # (B,V*Ts,H,W,3) uint8, Ts can be larger than v1
    tcp_state = step_data["tcp_state"]  # (B,Ts,2) float32 or None, NOTE: includes force data

    B, Ts, D_state = tcp_state.shape
    T_cond = 2  # hard coded for now, should be consistent with training
    gt_video = torch.from_numpy(gt_video).float() / 127.5 - 1.  # (B,V*Ts,H,W,3), in [-1,1]
    gt_view0_B_T_C_H_W = gt_video[:, -T_cond:].permute(0, 1, 4, 2, 3)  # (B,T,C,H,W), single view for now

    tcp_pose_B_T_D = tcp_state[:, -T_cond:, :].astype(np.float32)  # (B,T,2), in [0,512]

    # Norm input states
    tcp_pose_B_T_D = (tcp_pose_B_T_D - dataset_states_min) / (dataset_states_max - dataset_states_min) * 2. - 1.  # in [-1,1]

    # Should be consistent with the config yaml file
    obs_dict = {
        "agent_pos": torch.from_numpy(tcp_pose_B_T_D).to("cuda"),  # should be (B,T,2)
    }

    if True or agent.infer_frame_idx % max_cache_action == 0:  # always enter
        primary_img = gt_view0_B_T_C_H_W.to("cuda")
        # primary_img = gt_view0_B_T_C_H_W[:, :, :, :, :].unsqueeze(1)  # (B,1,C,H,W)
        print("[DEBUG] primary_img shape:", primary_img.shape)

        obs_dict["image"] = primary_img  # should be (B,T,C,H,W)


    # 3.a Model inference
    with torch.no_grad():  # always enter
        action = agent.predict_action(obs_dict)['action_pred']
        action = action[:, :]  # keep batch_dim, (B,T,D)

    # 3.b Postprocess
    # print(action.shape, action.min(dim=0)[0], action.max(dim=0)[0])
    action = (action * 0.5 + 0.5).cpu()  # in [0,1]
    action = action.clamp(0., 1.)
    action = action * (dataset_action_max[None, :] -
                       dataset_action_min[None, :]) + dataset_action_min[None, :]
    # # print(action.shape, action.min(dim=0), action.max(dim=0))
    cache_action = action

    agent.infer_frame_idx += 1
    # return {"action": cache_action.detach().numpy().tolist()}
    out_action = cache_action[:, :, :].numpy()  # (1,T,D)
    print("[DEBUG] gpu_service output action shape:", out_action.shape)
    request_to_evaluator = StepRequestFromPolicy.encode_from_raw(action=out_action)
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

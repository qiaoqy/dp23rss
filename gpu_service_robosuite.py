import os
import pathlib
from functools import lru_cache
from typing import Dict, Optional
import json
import warnings

import hydra
import numpy as np
import torch
from fastapi import FastAPI, HTTPException
from omegaconf import OmegaConf

from robokit.connects.protocols import StepRequestFromEvaluator, StepRequestFromPolicy
from robokit.debug_utils.printer import print_batch


"""How to use me?
conda activate robodiff
cd code/dp23rss_fork
export PYTHONPATH=~/code/dp23rss_fork
export CKPT_INDEX=4
export TRAIN_DIR=/home/geyuan/code/dp23rss_fork/data/outputs/2026.04.17/17.27.18_train_diffusion_transformer_hybrid_robosuite_state
CUDA_VISIBLE_DEVICES=7 uvicorn gpu_service_robosuite:gpu_app --port 7287
"""

gpu_app = FastAPI()
max_cache_action = int(os.getenv("MAX_CACHE_ACTION", "8"))

OmegaConf.register_new_resolver("eval", eval, replace=True)


def _str2bool(value: Optional[str], default: bool) -> bool:
    if value is None:
        return default
    value = value.strip().lower()
    if value in {"1", "true", "t", "yes", "y", "on"}:
        return True
    if value in {"0", "false", "f", "no", "n", "off"}:
        return False
    return default


def _find_latest_run_dir(output_root: pathlib.Path, run_name: str) -> Optional[pathlib.Path]:
    if not output_root.exists():
        return None
    candidates = sorted(
        output_root.glob(f"*/*_{run_name}"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    return candidates[0] if len(candidates) > 0 else None


def _train_dir_from_log_time(log_time: str, output_root: pathlib.Path, run_name: str) -> pathlib.Path:
    if "-" not in log_time:
        raise ValueError(
            f"`LOG_TIME` must be like YYYY.MM.DD-HH.MM.SS, got: {log_time}"
        )
    date_part, time_part = log_time.split("-", 1)
    return output_root / date_part / f"{time_part}_{run_name}"


def resolve_train_project_dir() -> pathlib.Path:
    train_dir_env = os.getenv("TRAIN_DIR", "").strip()
    if len(train_dir_env) > 0:
        return pathlib.Path(train_dir_env).expanduser().resolve()

    output_root = pathlib.Path(
        os.getenv("OUTPUT_ROOT", str(pathlib.Path(__file__).resolve().parent / "data" / "outputs"))
    ).expanduser().resolve()
    run_name = os.getenv("RUN_NAME", "train_diffusion_transformer_hybrid_robosuite_state")
    log_time = os.getenv("LOG_TIME", "").strip()

    if len(log_time) > 0:
        return _train_dir_from_log_time(log_time=log_time, output_root=output_root, run_name=run_name)

    latest = _find_latest_run_dir(output_root=output_root, run_name=run_name)
    if latest is None:
        raise FileNotFoundError(
            f"Cannot resolve train dir. Set `TRAIN_DIR` or `LOG_TIME`. "
            f"Searched under: {output_root}"
        )
    return latest


def _resolve_stats_json_path(train_dir: pathlib.Path) -> pathlib.Path:
    stats_json_env = os.getenv("STATS_JSON", "").strip()
    if len(stats_json_env) > 0:
        return pathlib.Path(stats_json_env).expanduser().resolve()
    return (train_dir / "statistics.json").resolve()


@lru_cache(maxsize=8)
def get_dataset_stats(train_dir_str: str) -> Dict[str, np.ndarray]:
    train_dir = pathlib.Path(train_dir_str)
    stats_json_path = _resolve_stats_json_path(train_dir)
    if not stats_json_path.exists():
        warnings.warn(
            f"Missing statistics json: {stats_json_path}. "
            f"Set `STATS_JSON` or ensure `statistics.json` exists in train dir."
        )
        return {}

    with stats_json_path.open("r", encoding="utf-8") as f:
        payload = json.load(f)

    stats = payload.get("stats", None)
    if stats is None:
        raise KeyError(f"`stats` key not found in {stats_json_path}")
    if "all_state" not in stats or "action" not in stats:
        raise KeyError(f"`all_state`/`action` keys not found under `stats` in {stats_json_path}")

    return {
        "all_state_min": np.asarray(stats["all_state"]["min"], dtype=np.float32),
        "all_state_max": np.asarray(stats["all_state"]["max"], dtype=np.float32),
        "action_min": np.asarray(stats["action"]["min"], dtype=np.float32),
        "action_max": np.asarray(stats["action"]["max"], dtype=np.float32),
        "stats_json_path": str(stats_json_path),
    }


@lru_cache(maxsize=8)
def get_agent(device: str, use_ema: bool = True):
    train_dir = resolve_train_project_dir()
    config_path = train_dir / ".hydra" / "config.yaml"
    if not config_path.exists():
        raise FileNotFoundError(f"Missing hydra config: {config_path}")

    hydra_config = OmegaConf.load(str(config_path))
    model = hydra.utils.instantiate(hydra_config.policy)
    print(f"[DEBUG] Instantiated model from config: {type(model)}")

    ckpt_dir = train_dir / "checkpoints"
    ckpt_list = os.listdir(ckpt_dir)
    ckpt_list.sort()
    print(f"[DEBUG] Found ckpts: {ckpt_list}")
    ckpt_paths = sorted([p for p in ckpt_dir.iterdir() if p.suffix == ".ckpt"])
    if len(ckpt_paths) == 0:
        raise FileNotFoundError(f"No .ckpt found under: {ckpt_dir}")

    ckpt_index = int(os.getenv("CKPT_INDEX", "-1"))
    if ckpt_index >= len(ckpt_paths) or ckpt_index < -len(ckpt_paths):
        raise IndexError(
            f"`CKPT_INDEX`={ckpt_index} out of range for {len(ckpt_paths)} checkpoints."
        )
    ckpt_path = ckpt_paths[ckpt_index]

    payload = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    state_dicts = payload["state_dicts"]
    weight = state_dicts["ema_model"] if use_ema else state_dicts["model"]
    model.load_state_dict(weight, strict=True)
    model = model.to(device).eval()
    model.infer_frame_idx = 0

    state_dim = int(hydra_config.shape_meta.obs.all_state.shape[0])
    action_dim = int(hydra_config.shape_meta.action.shape[0])
    n_obs_steps = int(hydra_config.n_obs_steps)
    stats = get_dataset_stats(str(train_dir))
    if stats != {}:
        if int(stats["all_state_min"].shape[0]) != state_dim:
            raise ValueError(
                f"State dim mismatch between config and statistics: config={state_dim}, "
                f"stats={int(stats['all_state_min'].shape[0])}, file={stats['stats_json_path']}"
            )
        if int(stats["action_min"].shape[0]) != action_dim:
            raise ValueError(
                f"Action dim mismatch between config and statistics: config={action_dim}, "
                f"stats={int(stats['action_min'].shape[0])}, file={stats['stats_json_path']}"
            )

    print(
        f"[gpu_service_robosuite] Loaded ckpt={ckpt_path}, use_ema={use_ema}, "
        f"state_dim={state_dim}, action_dim={action_dim}, n_obs_steps={n_obs_steps}, device={device}"
    )
    return model, hydra_config, str(ckpt_path), state_dim, action_dim, n_obs_steps


@gpu_app.get("/")
def read_root():
    return {"message": "Hello, World!"}


@gpu_app.get("/init")
def model_init():
    device = os.getenv("DEVICE", "cuda")
    use_ema = _str2bool(os.getenv("USE_EMA"), True)
    _, _, ckpt_path, state_dim, action_dim, n_obs_steps = get_agent(device=device, use_ema=use_ema)
    train_dir = str(resolve_train_project_dir())
    stats = get_dataset_stats(train_dir)
    return {
        "message": "Initialized.",
        "max_cache_action": max_cache_action,
        "ckpt_path": ckpt_path,
        "state_dim": state_dim,
        "action_dim": action_dim,
        "n_obs_steps": n_obs_steps,
    }


@gpu_app.get("/reset")
def model_reset():
    device = os.getenv("DEVICE", "cuda")
    use_ema = _str2bool(os.getenv("USE_EMA"), True)
    agent, _, _, _, _, _ = get_agent(device=device, use_ema=use_ema)
    if hasattr(agent, "reset"):
        agent.reset()
    return {"max_cache_action": max_cache_action}


@gpu_app.post("/step")
def model_step(step_request: StepRequestFromEvaluator):
    device = os.getenv("DEVICE", "cuda")
    use_ema = _str2bool(os.getenv("USE_EMA"), True)
    agent, hydra_config, ckpt_path, state_dim, action_dim, n_obs_steps = get_agent(device=device, use_ema=use_ema)
    train_dir = str(resolve_train_project_dir())
    stats = get_dataset_stats(train_dir)

    step_data = step_request.decode_to_raw()
    tcp_state = step_data.get("tcp_state", None)
    if tcp_state is None:
        raise HTTPException(status_code=400, detail="`tcp_state` is required for robosuite state-only policy.")
    if tcp_state.ndim != 3:
        raise HTTPException(status_code=400, detail=f"`tcp_state` must be [B,Ts,D], got shape={tcp_state.shape}.")

    batch_size, time_steps, state_dim_in = tcp_state.shape
    if state_dim_in != state_dim:
        raise HTTPException(
            status_code=400,
            detail=f"`tcp_state` dim mismatch: expected D={state_dim}, got D={state_dim_in}.",
        )
    if time_steps < n_obs_steps:
        raise HTTPException(
            status_code=400,
            detail=f"`tcp_state` time length mismatch: expected Ts>={n_obs_steps}, got Ts={time_steps}.",
        )

    print_batch("[DEBUG] tcp_state", tcp_state)
    use_norm = getattr(hydra_config.task.dataset, "norm_input_output", True)
    print(f"[DEBUG] ckpt_path={ckpt_path}, use_norm={use_norm}")
    obs_state = tcp_state[:, -n_obs_steps:, :].astype(np.float32)
    if stats != {} and hydra_config.task.dataset.norm_input_output:
        eps = 1e-12
        obs_state = (obs_state - stats["all_state_min"][None, None, :]) / (
            stats["all_state_max"][None, None, :] - stats["all_state_min"][None, None, :] + eps
        )
    obs_dict = {
        "all_state": torch.from_numpy(obs_state).to(device),
    }

    with torch.no_grad():
        result = agent.predict_action(obs_dict)
        action_pred = result["action_pred"]  # [B, H, D_action]

    if action_pred.ndim != 3:
        raise HTTPException(status_code=500, detail=f"`action_pred` must be rank-3, got shape={tuple(action_pred.shape)}.")
    if action_pred.shape[0] != batch_size:
        raise HTTPException(
            status_code=500,
            detail=f"`action_pred` batch mismatch: input B={batch_size}, output B={action_pred.shape[0]}.",
        )
    if action_pred.shape[-1] != action_dim:
        raise HTTPException(
            status_code=500,
            detail=f"`action_pred` dim mismatch: expected D={action_dim}, got D={action_pred.shape[-1]}.",
        )

    out_action = action_pred.detach().cpu().numpy().astype(np.float32)
    if stats != {} and hydra_config.task.dataset.norm_input_output:
        out_action = out_action * (
            stats["action_max"][None, None, :] - stats["action_min"][None, None, :]
        ) + stats["action_min"][None, None, :]
    request_to_evaluator = StepRequestFromPolicy.encode_from_raw(action=out_action)
    agent.infer_frame_idx += 1
    return request_to_evaluator.model_dump(mode="json")

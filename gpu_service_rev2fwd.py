"""Parametrised policy service for the dp_tr rev2fwd iterative pipeline.

Differences vs ``dp23rss/gpu_service_reverse.py`` (the prototype we forked):

* All paths (train dir, stats dataset, collect dir, task tag, port, GPU) are
  configured via environment variables, never hard-coded. This lets
  ``run_iter.sh`` spin up a service per iteration without editing source.
* ``OnlineDataCollector`` is wired into ``/step`` and ``/reset`` so every
  rollout frame is persisted with RoboKit layout under
  ``${COLLECT_DIR}/ep_XXX/*.npz`` + ``episode_index.json``.
* ``/episode_end`` endpoint lets the evaluator close episodes cleanly (and
  store the Y/N sparse label from ``instruction_text``).
* Works for both Policy A (hard task → ``task_tag="A"``) and Policy B
  (easy task → ``task_tag="B"``) — the only difference is which ckpt dir and
  which collect dir you point the env vars at.

Environment variables (documented defaults match the legacy prototype):
    DPTR_TRAIN_DIR       hydra run-dir of the trained policy (required)
    DPTR_STATS_DATA_ROOT RoboKit root whose ``statistics.json`` anchors stats
    DPTR_COLLECT_DIR     directory to write ``ep_XXX/*.npz`` (required)
    DPTR_TASK_TAG        short tag stored in ``task_tag.txt`` (default "A")
    DPTR_WEIGHT_IDX      index into sorted checkpoint list (default -1 = last)
    DPTR_MAX_CACHE_ACT   action chunk size (default 32)
    DPTR_PORT            uvicorn port (default 6070)

Launch::

    conda activate robodiff
    export DPTR_TRAIN_DIR=/path/to/iter00/train_A
    export DPTR_STATS_DATA_ROOT=/path/to/0209_tower_boby_hard
    export DPTR_COLLECT_DIR=/path/to/iter01/rollout_B
    export DPTR_TASK_TAG=B
    CUDA_VISIBLE_DEVICES=0 uvicorn gpu_service_rev2fwd:gpu_app --port 6070
"""

from __future__ import annotations

import json
import os
import queue
import threading
import time
from functools import lru_cache
from pathlib import Path

import numpy as np
import torch
from fastapi import FastAPI
from omegaconf import OmegaConf

from diffusion_policy.dataset.tcl_dataset import TCLImageDataset, TCLDatasetHDF5
from diffusion_policy.data_processing import OnlineDataCollector
from robokit.connects.protocols import StepRequestFromEvaluator, StepRequestFromPolicy


OmegaConf.register_new_resolver("eval", eval, replace=True)

# ---------------------------------------------------------- env parameters
TRAIN_DIR = os.environ.get("DPTR_TRAIN_DIR")
# STATS_DATA_ROOT is now optional. When unset we fall back to the
# statistics.json that lives inside TRAIN_DIR.
STATS_DATA_ROOT = os.environ.get("DPTR_STATS_DATA_ROOT") or TRAIN_DIR
COLLECT_DIR = os.environ.get("DPTR_COLLECT_DIR")
TASK_TAG = os.environ.get("DPTR_TASK_TAG", "A")
# Either a checkpoint *file name* under TRAIN_DIR/checkpoints, or fallback to
# WEIGHT_IDX-style lookup when CKPT_NAME is empty.
CKPT_NAME = os.environ.get("DPTR_CKPT_NAME", "latest.ckpt")
WEIGHT_IDX = int(os.environ.get("DPTR_WEIGHT_IDX", "-1"))
MAX_CACHE_ACTION = int(os.environ.get("DPTR_MAX_CACHE_ACT", "32"))

assert TRAIN_DIR, "DPTR_TRAIN_DIR must be set"
assert COLLECT_DIR, "DPTR_COLLECT_DIR must be set"

# ---------------------------------------------------------- stats loading
_stats_file_dataset = Path(STATS_DATA_ROOT) / "statistics.json"
_stats_file_train = Path(TRAIN_DIR) / "statistics.json"

if _stats_file_dataset.exists() and _stats_file_dataset.resolve() != _stats_file_train.resolve():
    with open(_stats_file_dataset) as f:
        _stats_blob = json.load(f)
    dataset_stats = _stats_blob["stats"]
    ft = dataset_stats.setdefault("force_torque", {})
    if "p01" not in ft or "p99" not in ft:
        # Backfill force_torque p01/p99 via the H5 extractor (mirrors legacy).
        h5_path = Path(STATS_DATA_ROOT) / "hdf5" / f"{Path(STATS_DATA_ROOT).name}_240p.h5"
        tcl_hdf5 = TCLDatasetHDF5(
            str(STATS_DATA_ROOT), str(h5_path),
            use_extracted=True,
            load_keys=["rel_actions", "primary_rgb", "gripper_rgb",
                       "robot_obs", "language_text", "force_torque"],
        )
        all_force_torques = tcl_hdf5.dsets["force_torque"]
        ft["p01"] = np.quantile(all_force_torques, 0.01, axis=0).tolist()
        ft["p99"] = np.quantile(all_force_torques, 0.99, axis=0).tolist()
    if not _stats_file_train.exists():
        _stats_file_train.parent.mkdir(parents=True, exist_ok=True)
        with open(_stats_file_train, "w") as f:
            json.dump(_stats_blob, f, indent=4)

assert _stats_file_train.exists(), f"statistics.json not found under {_stats_file_train}"
with open(_stats_file_train) as f:
    _blob = json.load(f)
dataset_stats = _blob["stats"]
dataset_action_min = np.asarray(dataset_stats["rel_actions"]["min"])
dataset_action_max = np.asarray(dataset_stats["rel_actions"]["max"])
ft = dataset_stats.setdefault("force_torque", {})
assert "p01" in ft and "p99" in ft, (
    f"force_torque p01/p99 missing in {_stats_file_train}; provide a "
    "DPTR_STATS_DATA_ROOT with hdf5/<name>_240p.h5 to backfill, or extend "
    "the statistics.json offline.")
ft["p01"] = np.asarray(ft["p01"])
ft["p99"] = np.asarray(ft["p99"])


# ---------------------------------------------------------- collector
collector = OnlineDataCollector(
    save_root=COLLECT_DIR,
    task_tag=TASK_TAG,
    unnorm_action_min=dataset_action_min,
    unnorm_action_max=dataset_action_max,
)

# Cache of the *previous* action chunk we returned to the remote. The remote
# executes all `MAX_CACHE_ACTION` frames in this chunk, then on the next
# /step call it ships back `MAX_CACHE_ACTION` observation frames covering
# what it actually saw while executing those actions. We pair them 1:1 to
# write `MAX_CACHE_ACTION` RoboKit npz frames per call. ``None`` indicates
# we are at the start of an episode (no actions executed yet → nothing to
# save for this /step, only fresh inference).
_prev_action_chunk: "np.ndarray | None" = None

# ---------------------------------------------------------- background writer
# Per-frame npz writes are I/O bound; doing 24 of them inline blows up the
# /step latency the remote sees. We hand frames to a single daemon writer
# thread via an unbounded queue so the request handler returns as soon as
# inference is done.
_WRITE_Q: "queue.Queue[dict | None]" = queue.Queue()


def _writer_loop() -> None:
    while True:
        item = _WRITE_Q.get()
        if item is None:
            _WRITE_Q.task_done()
            return
        try:
            collector.on_step(**item)
        except Exception as e:  # pragma: no cover
            print(f"[gpu_service_rev2fwd] writer error: {e}")
        finally:
            _WRITE_Q.task_done()


_writer_thread = threading.Thread(target=_writer_loop, name="dptr-writer",
                                  daemon=True)
_writer_thread.start()

# Latency log gating. 0 = log every /step; >0 = log every Nth.
_LOG_EVERY = int(os.environ.get("DPTR_LOG_EVERY", "1"))
_step_counter = 0


# ---------------------------------------------------------- model loader
gpu_app = FastAPI()

# Idle-watchdog: if no /step for this many seconds *after* a stream has
# started (i.e. at least one frame written for the current episode), treat
# the in-flight episode as finished. The guard on ``_ep_last_step_ts > 0``
# inside ``OnlineDataCollector.close_if_idle`` ensures we never close an
# empty episode, so this watchdog is dormant while the service is just
# waiting for the first remote /step. Set to 0 to disable entirely.
EPISODE_IDLE_SEC = float(os.environ.get("DPTR_EPISODE_IDLE_SEC", "8"))


@gpu_app.on_event("startup")
async def _start_idle_watchdog() -> None:
    if EPISODE_IDLE_SEC <= 0:
        return
    import asyncio

    async def _loop() -> None:
        # Poll every ~1s (or quarter of the threshold for larger thresholds);
        # the dormant guard inside collector.close_if_idle keeps this cheap.
        sleep_s = max(0.5, min(1.0, EPISODE_IDLE_SEC / 4))
        while True:
            await asyncio.sleep(sleep_s)
            try:
                if collector.close_if_idle(EPISODE_IDLE_SEC, default_label="N"):
                    print(f"[gpu_service_rev2fwd] idle>{EPISODE_IDLE_SEC}s → "
                          f"auto-saved episode (default label=N)")
            except Exception as e:  # pragma: no cover
                print(f"[gpu_service_rev2fwd] idle watchdog error: {e}")

    asyncio.create_task(_loop())


@gpu_app.on_event("shutdown")
async def _save_in_flight_on_shutdown() -> None:
    # Drain any pending writes before finalising the index.
    _WRITE_Q.join()
    summary = collector.finalize(default_label="N")
    print(f"[gpu_service_rev2fwd] shutdown finalise → {summary}")


@lru_cache()
def get_agent(device: str):
    import hydra
    cfg = OmegaConf.load(Path(TRAIN_DIR) / ".hydra/config.yaml")
    model = hydra.utils.instantiate(cfg.policy)
    ckpt_dir = Path(TRAIN_DIR) / "checkpoints"
    named = ckpt_dir / CKPT_NAME if CKPT_NAME else None
    if named is not None and named.exists():
        ckpt = named
    else:
        ckpts = sorted(p for p in ckpt_dir.iterdir() if p.suffix == ".ckpt")
        ckpt = ckpts[WEIGHT_IDX]
    blob = torch.load(str(ckpt), map_location="cpu", weights_only=False)
    weight = blob["state_dicts"]["model"]
    model.load_state_dict(weight)
    model = model.to(device).eval()
    model.infer_frame_idx = 0
    print(f"[gpu_service_rev2fwd] loaded {ckpt}")
    return model, cfg, str(ckpt)


# ---------------------------------------------------------- endpoints
@gpu_app.get("/")
def root():
    return {"ok": True, "task_tag": TASK_TAG, "train_dir": TRAIN_DIR}


@gpu_app.get("/init")
def init():
    return {"max_cache_action": MAX_CACHE_ACTION, "task_tag": TASK_TAG}


@gpu_app.get("/reset")
def reset():
    global _prev_action_chunk
    agent, _, _ = get_agent("cuda")
    agent.reset()
    # Drop the cached chunk: the next /step starts a brand-new episode and
    # has no "actions executed since last call" to pair with its obs.
    _prev_action_chunk = None
    # Save the previous episode (if any). When the remote evaluator does not
    # supply an explicit Y/N, we conservatively label as ``"N"`` so that
    # downstream advantage filtering does not silently treat unverified
    # rollouts as successes. on_episode_end still respects an explicit Y/N
    # parsed from the most recent instruction_text.
    collector.on_episode_end(aborted=False, default_label="N")
    return {"max_cache_action": MAX_CACHE_ACTION}


@gpu_app.get("/episode_end")
def episode_end(success: str = "N"):
    """Close the current episode with an explicit Y/N label.

    Defaults to ``N``: an unset query parameter is interpreted as
    "unverified / failure" so we do not poison downstream training with
    silent successes.
    """
    flag = (success or "N").strip().upper()
    if flag not in ("Y", "N"):
        flag = "N"
    collector._ep_last_text = flag  # bypass parser for explicit call
    collector.on_episode_end(aborted=(flag == "N"), default_label=flag)
    return {"ok": True, "num_episodes": len(collector._index), "label": flag}


@gpu_app.post("/step")
def step(step_request: StepRequestFromEvaluator):
    global _prev_action_chunk, _step_counter
    t_enter = time.perf_counter()
    agent, cfg, _ = get_agent("cuda")

    data = step_request.decode_to_raw()
    instruction_text = data["instruction"]
    stage_flag = data["stage_flag"]
    gt_video = data["gt_video"]            # (B, 2*Ts, H, W, 3) uint8
    tcp_state = data["tcp_state"]          # (B, Ts, 12) float32
    t_decode = time.perf_counter()

    B, Ts, _ = tcp_state.shape
    # ``gt_video`` is (B, 2*Ts, H, W, 3) uint8. View 0 is the front camera,
    # view 1 the gripper camera, each Ts frames long.
    view0_u8 = gt_video[:, :Ts]                     # (B, Ts, H, W, 3) uint8
    view1_u8 = gt_video[:, Ts:]                     # (B, Ts, H, W, 3) uint8
    force = tcp_state[:, :, -6:].astype(np.float32)
    tcp = tcp_state[:, :, :6].astype(np.float32)
    force_norm = TCLImageDataset.norm_state_or_force(
        force, norm_type="quantile", meta_data=dataset_stats["force_torque"])
    t_preproc = time.perf_counter()

    # ---- enqueue (prev_action_chunk) ↔ (this /step's obs frames) ----
    # All disk I/O is delegated to the background writer so /step latency
    # stays close to inference time only. We hand the writer the *raw uint8*
    # camera frames straight from the protocol payload — no float<->uint8
    # round-trip — and per-frame numpy slices for state/force.
    n_pair = 0
    if _prev_action_chunk is not None:
        n_pair = min(int(_prev_action_chunk.shape[0]), Ts)
        obs_offset = Ts - n_pair
        prim_chunk = view0_u8[0, obs_offset:obs_offset + n_pair]   # (n_pair, H, W, 3)
        grip_chunk = view1_u8[0, obs_offset:obs_offset + n_pair]
        for k in range(n_pair):
            t = obs_offset + k
            robot_obs_vec = (np.concatenate([tcp[0, t], np.zeros(7),
                                             tcp_state[0, t, 6:7]])[:14]
                             if tcp_state.shape[-1] >= 7 else np.zeros(14))
            _WRITE_Q.put({
                "obs_dict": {
                    "primary_rgb": prim_chunk[k],         # uint8, no copy
                    "gripper_rgb": grip_chunk[k],
                    "robot_obs": robot_obs_vec,
                    "force_torque": tcp_state[0, t, -6:].copy(),
                },
                "raw_action": _prev_action_chunk[k].copy(),
                "instruction_text": instruction_text,
                "stage_flag": stage_flag,
            })
        agent.infer_frame_idx += n_pair
    t_enqueue = time.perf_counter()

    # ---- inference for the next chunk --------------------------------
    # We only need the *most recent* camera frame for the policy, so do the
    # float conversion on a single slice instead of the full Ts*2 video.
    img_last_u8 = view0_u8[:, -1:]                  # (B, 1, H, W, 3) uint8
    grip_last_u8 = view1_u8[:, -1:]
    img_last = (torch.from_numpy(img_last_u8).to("cuda", non_blocking=True)
                .float().div_(127.5).sub_(1.0).permute(0, 1, 4, 2, 3))
    grip_last = (torch.from_numpy(grip_last_u8).to("cuda", non_blocking=True)
                 .float().div_(127.5).sub_(1.0).permute(0, 1, 4, 2, 3))
    obs_dict = {
        "joint_state": torch.from_numpy(tcp).to("cuda", non_blocking=True),
        "force": torch.from_numpy(force_norm).to("cuda", non_blocking=True),
        "image": img_last,
        "gripper": grip_last if "gripper" in cfg.shape_meta["obs"] else None,
    }
    obs_dict = {k: v for k, v in obs_dict.items() if v is not None}
    t_h2d = time.perf_counter()

    with torch.no_grad():
        action = agent.predict_action(obs_dict)["action_pred"][0]
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t_infer = time.perf_counter()

    action = (action * 0.5 + 0.5).cpu().clamp(0., 1.).numpy()
    action = action * (dataset_action_max - dataset_action_min) + dataset_action_min
    action[:, 6:] = (action[:, 6:] >= 0.5).astype(action.dtype)

    # Cache the slice the remote will actually execute (= MAX_CACHE_ACTION,
    # which the remote also reads from /init). If the policy returns more
    # than that, the trailing actions will be discarded by the remote and
    # are *not* paired with future obs.
    n_exec = min(int(MAX_CACHE_ACTION), int(action.shape[0]))
    _prev_action_chunk = action[:n_exec].copy()

    out_action = action[None, :, :]
    payload = StepRequestFromPolicy.encode_from_raw(action=out_action).model_dump(mode="json")
    t_encode = time.perf_counter()

    _step_counter += 1
    if _LOG_EVERY > 0 and (_step_counter % _LOG_EVERY == 0):
        total_ms = (t_encode - t_enter) * 1000
        print(
            f"[step #{_step_counter:04d}] total={total_ms:6.1f}ms "
            f"decode={(t_decode - t_enter) * 1000:5.1f} "
            f"preproc={(t_preproc - t_decode) * 1000:5.1f} "
            f"enqueue(n={n_pair})={(t_enqueue - t_preproc) * 1000:5.1f} "
            f"h2d={(t_h2d - t_enqueue) * 1000:5.1f} "
            f"infer={(t_infer - t_h2d) * 1000:5.1f} "
            f"encode={(t_encode - t_infer) * 1000:5.1f} "
            f"qlen={_WRITE_Q.qsize()} Ts={Ts}",
            flush=True,
        )
    return payload

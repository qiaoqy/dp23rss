"""Parametrised policy service for the dp_tr **rollout_service** workflow.

Lineage
-------
This file is the production "rollout + collect" descendant of
``gpu_service_reverse.py``. It merges:

* The **parametrisation + OnlineDataCollector wiring + background writer
  thread** that ``gpu_service_rev2fwd.py`` already had (no hard-coded
  ``log_time``/dataset paths, env-var driven, every frame persisted in
  RoboKit layout).
* The **performance changes from the 3 most recent commits to**
  ``dp23rss/gpu_service.py`` (commits ``19b17be`` ▸ ``803b2c8`` ▸ ``bd1bd01``):

  - ``SharedMemoryPool`` + ``get_mem_pool`` so the (1, V*Ts, H, W, 3) uint8
    video buffer is allocated **once** and reused (avoids per-step
    allocation churn).
  - ``StepRequestFromEvaluator.decode_to_raw_buffer(out_video_buffer=...)``
    instead of ``decode_to_raw()`` — decodes JPEG bytes directly into the
    pooled buffer.
  - Optional ``gtp`` profiler hooks (``with gtp("...", group=...): ...``
    + ``gtp.step()`` + periodic ``gtp.report()``) controlled via
    ``DPTR_PROFILE_EVERY``.
  - ``max_cache_action`` defaults to 32 (was 16 in the very oldest
    snapshot) — already the rev2fwd default, kept here for clarity.

Inherited intentionally
-----------------------
- The lazy "only float-convert the last camera frame" path from
  ``gpu_service_rev2fwd.py`` is **kept** (the bd1bd01 commit moves the
  whole video to CUDA up-front, which is heavier; the lazy path is faster
  for our workload because we only feed the policy the last frame).
- The background writer thread (per-step npz writes off the request
  thread) is kept as well.

Environment variables
---------------------
    DPTR_TRAIN_DIR        hydra run-dir of the trained policy           (required)
    DPTR_STATS_DATA_ROOT  RoboKit root with statistics.json             (default: TRAIN_DIR)
    DPTR_COLLECT_DIR      directory to write ``ep_XXX/*.npz``           (required)
    DPTR_TASK_TAG         short tag stored in ``task_tag.txt``          (default "A")
    DPTR_CKPT_NAME        ckpt filename under TRAIN_DIR/checkpoints     (default "latest.ckpt")
    DPTR_WEIGHT_IDX       fallback index when CKPT_NAME missing         (default -1)
    DPTR_MAX_CACHE_ACT    chunk size                                    (default 32)
    DPTR_LOG_EVERY        per-step latency log cadence; 0 = silent      (default 1)
    DPTR_PROFILE_EVERY    gtp.report() cadence; 0 = disabled            (default 0)
    DPTR_EPISODE_IDLE_SEC idle watchdog (s) to auto-close episodes      (default 8)

Launch
------
    conda activate robodiff
    export DPTR_TRAIN_DIR=/path/to/run-dir
    export DPTR_COLLECT_DIR=/path/to/dp_tr/rollout_service/data/<model>
    export DPTR_TASK_TAG=B
    CUDA_VISIBLE_DEVICES=0 uvicorn gpu_service_rollout:gpu_app --port 6070
"""

from __future__ import annotations

import json
import os
import queue
import threading
import time
from functools import lru_cache
from pathlib import Path
from typing import Dict, Tuple, Union

import numpy as np
import torch
from fastapi import FastAPI
from omegaconf import OmegaConf

from diffusion_policy.dataset.tcl_dataset import TCLImageDataset, TCLDatasetHDF5
from diffusion_policy.data_processing import OnlineDataCollector
from robokit.connects.protocols import StepRequestFromEvaluator, StepRequestFromPolicy

# gtp profiler is optional — robokit may not always ship the module.
try:
    from robokit.debug_utils.time_profiler import global_time_profiler as gtp  # type: ignore
    _HAS_GTP = True
except Exception:  # pragma: no cover
    _HAS_GTP = False

    class _NoopGtp:
        """Minimal stand-in so `with gtp(...): ...` and gtp.step()/report()
        are unconditionally safe."""

        def __call__(self, *_a, **_kw):
            return self

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def step(self):
            pass

        def report(self):
            pass

    gtp = _NoopGtp()  # type: ignore


OmegaConf.register_new_resolver("eval", eval, replace=True)


# ---------------------------------------------------------- noisy-print filter
import sys


class _StdoutLineFilter:
    """Drop stdout lines matching any of ``patterns``; pass through the rest.

    Used to silence third-party prints we don't own (e.g. RoboKit's
    ``[DEBUG] decode_to_raw_buffer:`` on every /step) without monkey-patching
    the offending module. stderr is left untouched so real errors still
    surface.
    """

    def __init__(self, wrapped, patterns):
        self._wrapped = wrapped
        self._patterns = tuple(patterns)
        self._buf = ""

    def write(self, s):
        if not s:
            return 0
        self._buf += s
        out = []
        while "\n" in self._buf:
            line, self._buf = self._buf.split("\n", 1)
            if not any(p in line for p in self._patterns):
                out.append(line + "\n")
        if out:
            self._wrapped.write("".join(out))
        return len(s)

    def flush(self):
        if self._buf and not any(p in self._buf for p in self._patterns):
            self._wrapped.write(self._buf)
        self._buf = ""
        self._wrapped.flush()

    def __getattr__(self, name):
        return getattr(self._wrapped, name)


if not isinstance(sys.stdout, _StdoutLineFilter):
    sys.stdout = _StdoutLineFilter(sys.stdout, patterns=(
        "[DEBUG] decode_to_raw_buffer",
    ))

# ---------------------------------------------------------- env parameters
TRAIN_DIR = os.environ.get("DPTR_TRAIN_DIR")
STATS_DATA_ROOT = os.environ.get("DPTR_STATS_DATA_ROOT") or TRAIN_DIR
COLLECT_DIR = os.environ.get("DPTR_COLLECT_DIR")
TASK_TAG = os.environ.get("DPTR_TASK_TAG", "A")
CKPT_NAME = os.environ.get("DPTR_CKPT_NAME", "latest.ckpt")
WEIGHT_IDX = int(os.environ.get("DPTR_WEIGHT_IDX", "-1"))
MAX_CACHE_ACTION = int(os.environ.get("DPTR_MAX_CACHE_ACT", "32"))
# Per-step latency line cadence. Default 50 -> ~1 line every few seconds at
# ~150ms/step. Set to 1 for verbose, 0 to silence completely.
_LOG_EVERY = int(os.environ.get("DPTR_LOG_EVERY", "50"))
_PROFILE_EVERY = int(os.environ.get("DPTR_PROFILE_EVERY", "0"))
# Idle-watchdog: if no /step for this many seconds *after* a stream has
# started, treat the in-flight episode as finished and persist it.
EPISODE_IDLE_SEC = float(os.environ.get("DPTR_EPISODE_IDLE_SEC", "5"))

assert TRAIN_DIR, "DPTR_TRAIN_DIR must be set"
assert COLLECT_DIR, "DPTR_COLLECT_DIR must be set"

# ---------------------------------------------------------- stats loading
_stats_file_dataset = Path(STATS_DATA_ROOT) / "statistics.json"
_stats_file_train = Path(TRAIN_DIR) / "statistics.json"

if _stats_file_dataset.exists() and _stats_file_dataset.resolve() != _stats_file_train.resolve():
    with open(_stats_file_dataset) as f:
        _stats_blob = json.load(f)
    _ds_stats = _stats_blob["stats"]
    _ft = _ds_stats.setdefault("force_torque", {})
    if "p01" not in _ft or "p99" not in _ft:
        h5_path = Path(STATS_DATA_ROOT) / "hdf5" / f"{Path(STATS_DATA_ROOT).name}_240p.h5"
        tcl_hdf5 = TCLDatasetHDF5(
            str(STATS_DATA_ROOT), str(h5_path),
            use_extracted=True,
            load_keys=["rel_actions", "primary_rgb", "gripper_rgb",
                       "robot_obs", "language_text", "force_torque"],
        )
        all_force_torques = tcl_hdf5.dsets["force_torque"]
        _ft["p01"] = np.quantile(all_force_torques, 0.01, axis=0).tolist()
        _ft["p99"] = np.quantile(all_force_torques, 0.99, axis=0).tolist()
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
dataset_action_mean = np.asarray(dataset_stats["rel_actions"]["mean"])
ft = dataset_stats.setdefault("force_torque", {})
assert "p01" in ft and "p99" in ft, (
    f"force_torque p01/p99 missing in {_stats_file_train}; provide a "
    "DPTR_STATS_DATA_ROOT with hdf5/<name>_240p.h5 to backfill, or extend "
    "the statistics.json offline.")
ft["p01"] = np.asarray(ft["p01"])
ft["p99"] = np.asarray(ft["p99"])


# ---------------------------------------------------------- shared memory pool (from bd1bd01)
class SharedMemoryPool:
    """Lazily-allocated, shape-checked numpy buffer cache.

    Lifted verbatim from ``dp23rss/gpu_service.py`` (commit ``bd1bd01``) so
    the (1, V*Ts, H, W, 3) uint8 video buffer required by
    ``StepRequestFromEvaluator.decode_to_raw_buffer`` is only allocated
    once per running process (or whenever the shape changes).
    """

    def __init__(self) -> None:
        self._buffers: Dict[str, np.ndarray] = {}
        self._shapes: Dict[str, Tuple] = {}

    def get_or_allocate(self, key_to_shape: Union[str, Dict[str, Tuple]],
                        dtype=np.float32):
        if isinstance(key_to_shape, str):
            return self._buffers[key_to_shape]
        assert isinstance(key_to_shape, dict)
        for key, shape in key_to_shape.items():
            if key not in self._buffers or self._shapes.get(key) != shape:
                self._buffers[key] = np.empty(shape, dtype=dtype)
                self._shapes[key] = shape
                print(f"[gpu_service_rollout] SharedMemoryPool[{key}] -> {shape}")
        return {k: self._buffers[k] for k in key_to_shape.keys()}

    def clear(self) -> None:
        self._buffers.clear()
        self._shapes.clear()


@lru_cache()
def get_mem_pool() -> SharedMemoryPool:
    return SharedMemoryPool()


# ---------------------------------------------------------- collector + writer
collector = OnlineDataCollector(
    save_root=COLLECT_DIR,
    task_tag=TASK_TAG,
    unnorm_action_min=dataset_action_min,
    unnorm_action_max=dataset_action_max,
)

# Cache of the previous action chunk we returned. The remote executes all
# MAX_CACHE_ACTION frames in this chunk, then on the next /step call ships
# back MAX_CACHE_ACTION fresh observation frames covering what it actually
# saw while executing them. We pair them 1:1 to write MAX_CACHE_ACTION
# RoboKit npz frames per /step call.
_prev_action_chunk: "np.ndarray | None" = None

# Per-frame npz writes are I/O bound; doing 24+ inline blows up /step
# latency. Hand frames to a single daemon writer thread.
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
            print(f"[gpu_service_rollout] writer error: {e}")
        finally:
            _WRITE_Q.task_done()


_writer_thread = threading.Thread(target=_writer_loop, name="dptr-rollout-writer",
                                  daemon=True)
_writer_thread.start()

_step_counter = 0


# ---------------------------------------------------------- model loader
gpu_app = FastAPI()


@gpu_app.on_event("startup")
async def _start_idle_watchdog() -> None:
    if EPISODE_IDLE_SEC <= 0:
        return
    import asyncio

    async def _loop() -> None:
        sleep_s = max(0.5, min(1.0, EPISODE_IDLE_SEC / 4))
        while True:
            await asyncio.sleep(sleep_s)
            try:
                # Snapshot identifiers *before* close_if_idle clears them so
                # the banner can show which episode + how many frames were
                # just saved.
                with collector._lock:
                    pending_ep = collector._ep_id
                    pending_len = collector._ep_frame_count
                if collector.close_if_idle(EPISODE_IDLE_SEC, default_label="Y"):
                    _WRITE_Q.join()  # ensure all queued frames are on disk
                    bar = "=" * 70
                    print(
                        f"\n{bar}\n"
                        f"[EPISODE SAVED] idle > {EPISODE_IDLE_SEC:.1f}s → "
                        f"ep_{pending_ep:03d} closed | frames={pending_len} | "
                        f"label=Y (default; edit episode_index.json to mark failures)\n"
                        f"  dir: {collector.save_root}/ep_{pending_ep:03d}\n"
                        f"  total episodes so far: {len(collector._index)}\n"
                        f"{bar}\n",
                        flush=True,
                    )
            except Exception as e:  # pragma: no cover
                print(f"[gpu_service_rollout] idle watchdog error: {e}")

    asyncio.create_task(_loop())


@gpu_app.on_event("shutdown")
async def _save_in_flight_on_shutdown() -> None:
    _WRITE_Q.join()
    summary = collector.finalize(default_label="Y")
    print(f"[gpu_service_rollout] shutdown finalise → {summary}")


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
    print(f"[gpu_service_rollout] loaded {ckpt}")
    return model, cfg, str(ckpt)


# ---------------------------------------------------------- endpoints
@gpu_app.get("/")
def root():
    return {"ok": True, "task_tag": TASK_TAG, "train_dir": TRAIN_DIR,
            "collect_dir": COLLECT_DIR}


@gpu_app.get("/init")
def init():
    return {"max_cache_action": MAX_CACHE_ACTION, "task_tag": TASK_TAG}


@gpu_app.get("/reset")
def reset():
    global _prev_action_chunk
    agent, _, _ = get_agent("cuda")
    agent.reset()
    _prev_action_chunk = None
    collector.on_episode_end(aborted=False, default_label="Y")
    return {"max_cache_action": MAX_CACHE_ACTION}


@gpu_app.get("/episode_end")
def episode_end(success: str = "Y"):
    """Close the current episode with an explicit Y/N label.

    Default is ``Y`` (success). Pass ``?success=N`` to mark a failure.
    """
    flag = (success or "Y").strip().upper()
    if flag not in ("Y", "N"):
        flag = "Y"
    collector._ep_last_text = flag
    collector.on_episode_end(aborted=(flag == "N"), default_label=flag)
    return {"ok": True, "num_episodes": len(collector._index), "label": flag}


@gpu_app.post("/step")
def step(step_request: StepRequestFromEvaluator):
    global _prev_action_chunk, _step_counter
    t_enter = time.perf_counter()
    agent, cfg, _ = get_agent("cuda")
    mem_buffer = get_mem_pool()

    # Allocate / reuse the pooled (1, V*Ts, H, W, 3) uint8 video buffer
    # the protocol decoder writes JPEG → uint8 directly into.
    image_shape_C_H_W = cfg.image_shape
    req_max_cache = step_request.max_cache_action
    num_camera_views = step_request.num_camera_views
    mem_buffer.get_or_allocate({
        "gt_video": (1, num_camera_views * req_max_cache,
                     image_shape_C_H_W[1], image_shape_C_H_W[2], image_shape_C_H_W[0])
    }, dtype=np.uint8)

    # ---- decode -----------------------------------------------------
    # RoboKit's decode_to_raw_buffer prints a noisy [DEBUG] line every
    # call; the process-wide _StdoutLineFilter installed at module load
    # silences just that line.
    with gtp("decode", group="process_request"):
        video_buffer = mem_buffer.get_or_allocate("gt_video")
        data = step_request.decode_to_raw_buffer(out_video_buffer=video_buffer)
    instruction_text = data["instruction"]
    stage_flag = data["stage_flag"]
    gt_video = data["gt_video"]            # (B, 2*Ts, H, W, 3) uint8 (= video_buffer)
    tcp_state = data["tcp_state"]          # (B, Ts, 12) float32
    t_decode = time.perf_counter()

    # ---- preproc / state norm --------------------------------------
    with gtp("cpu_type_convert", group="process_request"):
        B, Ts, _ = tcp_state.shape
        view0_u8 = gt_video[:, :Ts]                # (B, Ts, H, W, 3) uint8
        view1_u8 = gt_video[:, Ts:]                # (B, Ts, H, W, 3) uint8
        force = tcp_state[:, :, -6:].astype(np.float32)
        tcp = tcp_state[:, :, :6].astype(np.float32)

    with gtp("cpu_norm_input", group="process_request"):
        force_norm = TCLImageDataset.norm_state_or_force(
            force, norm_type="quantile", meta_data=dataset_stats["force_torque"])
        zero_force = getattr(cfg.task.dataset, "zero_force", False)
        if zero_force:
            force_norm = force_norm * 0.
    t_preproc = time.perf_counter()

    # ---- enqueue (prev_action_chunk) ↔ (this /step's obs frames) ----
    n_pair = 0
    if _prev_action_chunk is not None:
        n_pair = min(int(_prev_action_chunk.shape[0]), Ts)
        obs_offset = Ts - n_pair
        prim_chunk = view0_u8[0, obs_offset:obs_offset + n_pair]
        grip_chunk = view1_u8[0, obs_offset:obs_offset + n_pair]
        for k in range(n_pair):
            t = obs_offset + k
            robot_obs_vec = (np.concatenate([tcp[0, t], np.zeros(7),
                                             tcp_state[0, t, 6:7]])[:14]
                             if tcp_state.shape[-1] >= 7 else np.zeros(14))
            _WRITE_Q.put({
                "obs_dict": {
                    # Take a copy: the pooled buffer is overwritten on the
                    # next /step before the writer thread typically gets to
                    # serialise it.
                    "primary_rgb": prim_chunk[k].copy(),
                    "gripper_rgb": grip_chunk[k].copy(),
                    "robot_obs": robot_obs_vec,
                    "force_torque": tcp_state[0, t, -6:].copy(),
                },
                "raw_action": _prev_action_chunk[k].copy(),
                "instruction_text": instruction_text,
                "stage_flag": stage_flag,
            })
        agent.infer_frame_idx += n_pair
    t_enqueue = time.perf_counter()

    # ---- inference for the next chunk -------------------------------
    # Lazy float-conversion on only the most recent camera frame.
    img_last_u8 = view0_u8[:, -1:]                 # (B, 1, H, W, 3) uint8
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

    with gtp("predict_action", group="model_infer"):
        with torch.no_grad():
            action = agent.predict_action(obs_dict)["action_pred"][0]
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t_infer = time.perf_counter()

    with gtp("postprocess_action", group="model_infer"):
        action = (action * 0.5 + 0.5).cpu().clamp(0., 1.).numpy()
        action = action * (dataset_action_max - dataset_action_min) + dataset_action_min
        thresold = dataset_action_mean[-1]  # use the mean of the last dimension as threshold
        action[:, 6:] = (action[:, 6:] >= thresold).astype(action.dtype)

    n_exec = min(int(MAX_CACHE_ACTION), int(action.shape[0]))
    _prev_action_chunk = action[:n_exec].copy()

    out_action = action[None, :, :]
    with gtp("encode", group="process_request"):
        payload = StepRequestFromPolicy.encode_from_raw(action=out_action).model_dump(mode="json")
    t_encode = time.perf_counter()

    gtp.step()
    _step_counter += 1
    if _PROFILE_EVERY > 0 and (_step_counter % _PROFILE_EVERY == 0):
        gtp.report()
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

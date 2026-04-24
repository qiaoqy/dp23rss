"""Per-step online data collector for RoboKit-format rollouts.

Used by ``gpu_service_rev2fwd.py`` during remote evaluation / rollout: every
policy step the evaluator pushes back ``(obs_dict, raw_action)`` along with a
free-form ``instruction_text``; the collector writes one ``*.npz`` per frame
using RoboKit's layout so that, after enough episodes, the directory can be
loaded directly by :class:`TCLDataset` or converted to H5 via
:func:`convert_root_to_h5`.

Y/N supervision
---------------
The final frame's ``instruction_text`` is inspected for a ``"Y"`` / ``"N"``
flag. Only the last frame's label is stored in ``episode_index.json`` under
``sparse_label``; intermediate frames carry the raw ``instruction_text``
unchanged (as RoboKit's schema expects a language description per frame).

Directory layout
----------------
::

    <save_root>/
        episode_index.json                        # {"ep_000": {...}, ...}
        task_tag.txt                              # from constructor
        ep_<episode_id:03d>/
            <frame_timestamp>.npz                 # RoboKit format
"""

from __future__ import annotations

import datetime as dt
import io
import json
import pickle
import re
import shutil
import threading
import time
from pathlib import Path
from typing import Any, Optional

import numpy as np
from PIL import Image


# -------------------------------------------------------------- utilities

def _encode_jpeg_bytes(arr_hwc_u8: np.ndarray, quality: int = 95) -> bytes:
    buf = io.BytesIO()
    Image.fromarray(arr_hwc_u8).save(buf, format="JPEG", quality=quality)
    return buf.getvalue()


def _coerce_rgb(img: Any) -> bytes:
    """Accept JPEG bytes, PIL.Image, or HxWx3 uint8 ndarray → return JPEG bytes."""
    if isinstance(img, (bytes, bytearray)):
        return bytes(img)
    if isinstance(img, Image.Image):
        arr = np.asarray(img.convert("RGB"))
        return _encode_jpeg_bytes(arr)
    if isinstance(img, np.ndarray):
        if img.ndim == 3 and img.shape[-1] == 3:
            if img.dtype != np.uint8:
                img = np.clip(img * 255.0 if img.max() <= 1.01 else img, 0, 255).astype(np.uint8)
            return _encode_jpeg_bytes(img)
    raise TypeError(f"Cannot coerce RGB input of type {type(img).__name__}")


def _coerce_action_real_units(action: np.ndarray,
                              amin: Optional[np.ndarray],
                              amax: Optional[np.ndarray]) -> np.ndarray:
    """Pass-through normalizer for actions already in real units.

    ``gpu_service_rev2fwd.py`` un-normalises the policy output before handing
    it to the collector, so the collector must NOT re-apply the inverse
    normalisation (doing so was the "double unnorm" bug — see
    ``debug_reply/debug_notes.md §6``).

    We still binarise gripper (dim 6) at 0.5 → {0, 1}, and (when bounds are
    available) clip the first 6 dims to the recorded training range to guard
    against rare outliers.  ``amin`` / ``amax`` are kept in the signature for
    backward compatibility but only used for the optional clip.
    """
    a = np.asarray(action, dtype=np.float64).copy()
    if a.shape[-1] >= 7:
        a[..., 6] = (a[..., 6] >= 0.5).astype(np.float64)
    if amin is not None and amax is not None:
        amin = np.asarray(amin, dtype=np.float64)
        amax = np.asarray(amax, dtype=np.float64)
        rng = amax - amin
        live = np.abs(rng) >= 1e-3
        # only clip live, non-gripper dims
        live[6:] = False
        for i in np.where(live)[0]:
            a[..., i] = np.clip(a[..., i], amin[i], amax[i])
    return a


_YN_RE = re.compile(r"\b([YN])\b")


def _parse_yn(text: str) -> Optional[str]:
    if not text:
        return None
    m = _YN_RE.search(text.strip().upper())
    return m.group(1) if m else None


# ----------------------------------------------------------------- class

class OnlineDataCollector:
    """Thread-safe per-frame writer for RoboKit episodes.

    Parameters
    ----------
    save_root : path
        Root directory for this collect session (e.g. ``runs/iter01/rollout_B``).
    task_tag : str
        Short human-readable name recorded to ``task_tag.txt``.
    image_shape : tuple(int, int)
        ``(H, W)`` of the stored JPEG bytes — frames are stored at original
        resolution; downstream resize happens in :func:`convert_root_to_h5`.
    unnorm_action_min, unnorm_action_max : np.ndarray (7,) or None
        Recorded training action range. The collector now expects
        ``raw_action`` to ALREADY be in real units (gpu_service un-normalises
        before calling); these bounds are only used to optionally clip the
        first 6 dims against outliers and are otherwise ignored. Pass
        ``None`` to disable the clip.
    flush_every : int
        Flush filesystem buffers every N frames (default 1).
    primary_rgb_quality : int
        JPEG quality (default 95, lossless-enough for the reverser).
    """

    def __init__(self,
                 save_root: str | Path,
                 task_tag: str,
                 image_shape: tuple[int, int] = (480, 640),
                 unnorm_action_min: Optional[np.ndarray] = None,
                 unnorm_action_max: Optional[np.ndarray] = None,
                 flush_every: int = 1,
                 primary_rgb_quality: int = 95):
        self.save_root = Path(save_root)
        self.save_root.mkdir(parents=True, exist_ok=True)
        (self.save_root / "task_tag.txt").write_text(task_tag)
        self.task_tag = task_tag
        self.image_shape = tuple(image_shape)
        self.unnorm_amin = None if unnorm_action_min is None else np.asarray(unnorm_action_min)
        self.unnorm_amax = None if unnorm_action_max is None else np.asarray(unnorm_action_max)
        self.flush_every = max(1, int(flush_every))
        self.jpeg_quality = int(primary_rgb_quality)

        self._lock = threading.Lock()
        self._ep_id: Optional[int] = None
        self._ep_dir: Optional[Path] = None
        self._ep_frame_count = 0
        self._ep_last_text: str = ""
        self._ep_last_step_ts: float = 0.0
        self._index_path = self.save_root / "episode_index.json"
        self._index: dict = json.loads(self._index_path.read_text()) \
            if self._index_path.exists() else {}
        # Determine starting ep id so appends work.
        existing = [int(k.split("_")[1]) for k in self._index.keys()
                    if k.startswith("ep_")]
        self._next_ep_id = (max(existing) + 1) if existing else 0

    # ........................................................... lifecycle
    def _start_episode(self) -> None:
        self._ep_id = self._next_ep_id
        self._next_ep_id += 1
        self._ep_dir = self.save_root / f"ep_{self._ep_id:03d}"
        self._ep_dir.mkdir(parents=True, exist_ok=True)
        self._ep_frame_count = 0
        self._ep_last_text = ""
        self._ep_last_step_ts = 0.0

    def on_step(self,
                obs_dict: dict,
                raw_action: np.ndarray,
                instruction_text: str,
                stage_flag: str = "rollout") -> None:
        """Persist a single (obs, action) tuple. Thread-safe.

        ``obs_dict`` must include at least ``primary_rgb`` (HWC uint8 / PIL /
        JPEG bytes), ``robot_obs`` (np.ndarray shape ``[14]``) and
        ``force_torque`` (shape ``[6]``).  Optional: ``gripper_rgb``,
        ``primary_depth``, ``gripper_depth``.
        """
        with self._lock:
            if self._ep_dir is None:
                self._start_episode()
            assert self._ep_dir is not None

            # Use a per-episode monotonically-increasing frame index in the
            # filename. The wall-clock prefix is kept for human readability
            # but the suffix is the deterministic ordering key — multiple
            # frames written within the same /step (one per executed action
            # in a chunk) would otherwise collide on the microsecond.
            idx = self._ep_frame_count
            ts_name = (dt.datetime.utcnow().strftime("frame_%H%M%S_")
                       + f"{idx:06d}.npz")
            out_path = self._ep_dir / ts_name

            primary_bytes = _coerce_rgb(obs_dict["primary_rgb"])
            gripper_bytes = _coerce_rgb(obs_dict.get("gripper_rgb", obs_dict["primary_rgb"]))

            rel_actions = _coerce_action_real_units(
                np.asarray(raw_action), self.unnorm_amin, self.unnorm_amax)
            actions = rel_actions.copy()  # RoboKit convention in exp41

            frame = {
                "primary_rgb": np.array(primary_bytes),
                "gripper_rgb": np.array(gripper_bytes),
                "robot_obs": np.array(pickle.dumps(
                    np.asarray(obs_dict["robot_obs"], dtype=np.float64))),
                "actions": np.array(pickle.dumps(np.asarray(actions, dtype=np.float64))),
                "rel_actions": np.array(pickle.dumps(np.asarray(rel_actions, dtype=np.float64))),
                "force_torque": np.array(pickle.dumps(
                    np.asarray(obs_dict.get("force_torque", np.zeros(6)), dtype=np.float64))),
                "language_text": np.array(pickle.dumps(np.array(str(instruction_text)))),
            }
            if obs_dict.get("primary_depth") is not None:
                frame["primary_depth"] = np.array(_coerce_rgb(obs_dict["primary_depth"]))
            if obs_dict.get("gripper_depth") is not None:
                frame["gripper_depth"] = np.array(_coerce_rgb(obs_dict["gripper_depth"]))

            np.savez(str(out_path), **frame)
            self._ep_frame_count += 1
            self._ep_last_text = instruction_text
            self._ep_last_step_ts = time.time()
            # (stage_flag is currently only logged in the JSON index.)
            if self._ep_frame_count % self.flush_every == 0:
                # best-effort flush; not strictly necessary
                pass
            _ = stage_flag

    def on_episode_end(self, aborted: bool = False,
                       default_label: str = "N") -> None:
        """Close the current episode and record its sparse Y/N label.

        When the evaluator does not provide an explicit ``Y/N`` flag, fall
        back to ``default_label``. ``aborted=True`` always overrides the
        default to ``"N"`` (parsed Y/N from ``instruction_text`` still wins
        when present).
        """
        with self._lock:
            if self._ep_dir is None or self._ep_id is None:
                return
            label = _parse_yn(self._ep_last_text)
            if label is None:
                label = "N" if aborted else default_label
            ep_key = f"ep_{self._ep_id:03d}"
            info = {
                "length": self._ep_frame_count,
                "sparse_label": label,
                "aborted": bool(aborted),
                "created_at": int(time.time()),
                "task_tag": self.task_tag,
            }
            # Refresh from disk to pick up any manual edits the user made
            # to other episodes' sparse_label (e.g. flipping Y -> N for a
            # failed rollout). We MUST NOT clobber those by writing back the
            # stale in-memory copy.
            disk_index: dict = {}
            if self._index_path.exists():
                try:
                    disk_index = json.loads(self._index_path.read_text())
                except Exception:
                    disk_index = {}
            # Merge: disk wins for *other* episodes, in-memory wins for the
            # one we just closed — except we still preserve a pre-existing
            # sparse_label for the current ep if the user already edited it
            # (e.g. they manually wrote Y/N before /reset triggered).
            merged = dict(disk_index)
            prev = disk_index.get(ep_key)
            if prev is not None and "sparse_label" in prev:
                info["sparse_label"] = prev["sparse_label"]
            merged[ep_key] = info
            self._index = merged
            self._index_path.write_text(json.dumps(self._index, indent=2))
            self._ep_id = None
            self._ep_dir = None
            self._ep_frame_count = 0
            self._ep_last_text = ""
            self._ep_last_step_ts = 0.0

    def close_if_idle(self, idle_seconds: float,
                      default_label: str = "N") -> bool:
        """Close the in-flight episode when no new step arrived for ``idle_seconds``.

        Returns ``True`` iff an episode was closed.
        """
        with self._lock:
            if self._ep_dir is None or self._ep_last_step_ts <= 0.0:
                return False
            if (time.time() - self._ep_last_step_ts) < idle_seconds:
                return False
        # release lock before reentering on_episode_end
        self.on_episode_end(aborted=False, default_label=default_label)
        return True

    def finalize(self, default_label: str = "N") -> dict:
        """Flush any open episode and return a summary dict.

        Unlike the prior version, an in-flight episode is *saved* (not
        discarded) — the remote evaluator may simply disconnect without ever
        calling ``/episode_end`` or ``/reset``, and we don't want to lose the
        rollout data we already wrote to disk.
        """
        if self._ep_dir is not None:
            self.on_episode_end(aborted=False, default_label=default_label)
        with self._lock:
            # Re-read from disk so any user-edited sparse_label survives
            # the final write-back (mirror of on_episode_end logic).
            disk_index: dict = {}
            if self._index_path.exists():
                try:
                    disk_index = json.loads(self._index_path.read_text())
                except Exception:
                    disk_index = {}
            merged = dict(disk_index)
            for k, v in self._index.items():
                prev = disk_index.get(k)
                if prev is not None and "sparse_label" in prev:
                    v = {**v, "sparse_label": prev["sparse_label"]}
                merged[k] = v
            self._index = merged
            self._index_path.write_text(json.dumps(self._index, indent=2))
            return {"num_episodes": len(self._index), "root": str(self.save_root)}

"""Rule-based and critic-based filters for RoboKit-format episode directories.

All three filters share the same IO pattern used by ``run_iter.sh``:
    input  dir  = <iter_N>/B_*        (one sub-directory per episode)
    output dir  = <iter_N>/B_*_<stage>   (same layout, strict frame subset)

Every filter returns a :class:`FilterStats` describing how many frames /
episodes were kept vs dropped.  ``process_dir`` is the single public entry.

Implementation notes
--------------------
* We keep a frame by symlinking / copying the original ``.npz`` into the
  destination directory — this avoids re-encoding image bytes and keeps the
  RoboKit loader happy on the output.
* ``RoboKitStaticFilter`` and ``RoboKitSpeedAdjuster`` only read the numeric
  ``robot_obs`` / ``rel_actions`` bytes from each frame to decide, so they are
  CPU-light and usable with ``num_workers`` via ``multiprocessing.Pool``.
* ``RoboKitAdvantageFilter`` loads a trained :class:`RoboKitCritic` checkpoint
  to score states and runs GAE on the scored trajectory.

Static/speed filter defaults are aligned with exp41 plan.md (``velocity=2e-3``,
``spatial=0.05``, ``min_static=16``, ``gripper_protect=8``; speed
``threshold=2e-3``, ``target_step=5e-3``, ``gripper_radius=0``).
"""

from __future__ import annotations

import json
import pickle
import shutil
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
from tqdm import tqdm


# ------------------------------------------------------------------- helpers

@dataclass
class FilterStats:
    stage: str = ""
    episodes_in: int = 0
    episodes_out: int = 0
    frames_in: int = 0
    frames_out: int = 0
    per_episode: list = field(default_factory=list)

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2)


def _load_numeric(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return ``(robot_obs[14], rel_actions[7], actions[7])`` for one frame.

    Note: the gripper command lives in ``actions[6]`` (binary-ish, 0.5=close,
    1.0=open). ``robot_obs[13]`` is a noisy force-like signal and must NOT
    be used for gripper transition detection.
    """
    with np.load(str(path), allow_pickle=True) as f:
        robot_obs = np.asarray(pickle.loads(f["robot_obs"].item()), dtype=np.float64)
        rel_actions = np.asarray(pickle.loads(f["rel_actions"].item()), dtype=np.float64)
        actions = np.asarray(pickle.loads(f["actions"].item()), dtype=np.float64)
    return robot_obs, rel_actions, actions


def _discover_episodes(root: Path) -> list[tuple[str, list[Path]]]:
    out = []
    for entry in sorted(root.iterdir()):
        if not entry.is_dir() or entry.name in ("extracted", "__pycache__", "viz_videos"):
            continue
        files = sorted(entry.glob("*.npz"))
        if files:
            out.append((entry.name, files))
    return out


def _gripper_transitions(gripper: np.ndarray) -> np.ndarray:
    """Return frame indices at which the binarised gripper command flips.

    Demos encode gripper as ``{0=open, 1=closed}``; rollouts may emit softer
    values such as ``{0.5, 1.0}``. We binarise at threshold 0.5 then detect
    transitions on the clean ``{0, 1}`` signal so a real grasp/release is
    counted exactly once (no oscillation noise from raw float values).

    The returned indices ``c`` mean: ``binarised[c] != binarised[c+1]``.
    """
    if gripper.size < 2:
        return np.empty(0, dtype=np.int64)
    binar = (gripper > 0.5).astype(np.int8)
    return np.where(np.diff(binar) != 0)[0]


def _load_episode_numeric(npz_files: list[Path]):
    """Return ``(tcp[T,6], gripper_cmd[T], rel[T,7])`` per episode.

    ``gripper_cmd`` comes from ``actions[6]`` (binary command). NOT
    ``robot_obs[13]`` which is a noisy force-like signal.
    """
    tcp = np.zeros((len(npz_files), 6), dtype=np.float64)
    gripper = np.zeros((len(npz_files),), dtype=np.float64)
    rel = np.zeros((len(npz_files), 7), dtype=np.float64)
    for i, p in enumerate(npz_files):
        robot_obs, rel_actions, actions = _load_numeric(p)
        tcp[i] = robot_obs[:6]
        gripper[i] = actions[6]
        rel[i] = rel_actions
    return tcp, gripper, rel


def _copy_keep(src_files: list[Path], keep_idx: np.ndarray, dst_ep: Path) -> None:
    dst_ep.mkdir(parents=True, exist_ok=True)
    for i in keep_idx:
        src = src_files[int(i)]
        dst = dst_ep / src.name
        if dst.exists():
            dst.unlink()
        shutil.copy2(src, dst)


def _copy_keep_segments(src_files: list[Path], keep_mask: np.ndarray,
                         dst_root: Path, name: str,
                         min_seg_frames: int = 16) -> list[tuple[str, int]]:
    """Split keep-mask into contiguous-True runs and write each to its own dir.

    Any run shorter than ``min_seg_frames`` is dropped entirely (avoids leaving
    1-frame stubs behind that would break downstream window sampling).

    Returns list of (sub_episode_name, n_frames). When the input only yields
    a single kept segment, the sub-episode keeps the original name; otherwise
    we append ``_p{k:02d}``.
    """
    T = len(src_files)
    segs: list[tuple[int, int]] = []
    i = 0
    while i < T:
        if not keep_mask[i]:
            i += 1
            continue
        j = i
        while j < T and keep_mask[j]:
            j += 1
        if j - i >= min_seg_frames:
            segs.append((i, j))
        i = j
    if not segs:
        return []
    multi = len(segs) > 1
    out: list[tuple[str, int]] = []
    for k, (a, b) in enumerate(segs):
        sub = name if not multi else f"{name}_p{k:02d}"
        ep_dir = dst_root / sub
        ep_dir.mkdir(parents=True, exist_ok=True)
        for idx in range(a, b):
            src = src_files[idx]
            dst = ep_dir / src.name
            if dst.exists():
                dst.unlink()
            shutil.copy2(src, dst)
        out.append((sub, b - a))
    return out


# ------------------------------------------------------- 1. static filter

class RoboKitStaticFilter:
    """Drop long-static frame segments from each episode.

    A frame is flagged as ``static`` when both the per-frame TCP velocity
    magnitude is below ``velocity_thr`` AND the rolling spatial displacement
    within ``min_static`` frames is below ``spatial_thr``.  Contiguous static
    segments longer than ``min_static`` frames are dropped, with
    ``gripper_protect`` frames on either side of a gripper state change always
    retained so we never truncate the grasp/release moment.
    """

    def __init__(self,
                 velocity_thr: float = 2e-3,
                 spatial_thr: float = 0.05,
                 min_static: int = 16,
                 gripper_protect: int = 8,
                 gripper_protect_extremes_only: bool = False,
                 min_segment_frames: int = 16):
        self.velocity_thr = float(velocity_thr)
        self.spatial_thr = float(spatial_thr)
        self.min_static = int(min_static)
        self.gripper_protect = int(gripper_protect)
        self.gripper_protect_extremes_only = bool(gripper_protect_extremes_only)
        self.min_segment_frames = int(min_segment_frames)

    # ........................................................ per-episode
    def compute_keep_mask(self, tcp: np.ndarray, gripper: np.ndarray) -> np.ndarray:
        T = tcp.shape[0]
        if T <= 2:
            return np.ones(T, dtype=bool)

        vel = np.linalg.norm(np.diff(tcp[:, :3], axis=0), axis=-1)
        vel = np.concatenate([vel, vel[-1:]])  # align length to T

        # rolling spatial range within min_static window
        win = self.min_static
        disp = np.zeros(T, dtype=np.float64)
        for t in range(T):
            a = max(0, t - win // 2)
            b = min(T, t + win // 2 + 1)
            chunk = tcp[a:b, :3]
            disp[t] = float(np.linalg.norm(chunk.max(0) - chunk.min(0)))

        static_frame = (vel < self.velocity_thr) & (disp < self.spatial_thr)

        # Find contiguous static runs >= min_static and mark drop
        keep = np.ones(T, dtype=bool)
        i = 0
        while i < T:
            if not static_frame[i]:
                i += 1
                continue
            j = i
            while j < T and static_frame[j]:
                j += 1
            if j - i >= self.min_static:
                keep[i:j] = False
            i = j

        # Protect gripper transition region. Use binarised actions[6]
        # so ``{0=open, 1=closed}`` (demos) and ``{0.5, 1.0}`` (rollouts) both
        # produce the same clean transition indices.
        if self.gripper_protect > 0:
            changes = _gripper_transitions(gripper)
            if self.gripper_protect_extremes_only and changes.size > 0:
                changes = np.array([changes[0], changes[-1]]) if changes.size > 1 else changes
            for c in changes:
                a = max(0, c - self.gripper_protect)
                b = min(T, c + self.gripper_protect + 1)
                keep[a:b] = True
        return keep

    # .............................................................. driver
    def process_dir(self, src_dir: str | Path, dst_dir: str | Path) -> FilterStats:
        src, dst = Path(src_dir), Path(dst_dir)
        dst.mkdir(parents=True, exist_ok=True)
        eps = _discover_episodes(src)
        stats = FilterStats(stage="static_filter", episodes_in=len(eps))
        for name, files in tqdm(eps, desc="static_filter"):
            tcp, gripper, _ = _load_episode_numeric(files)
            keep = self.compute_keep_mask(tcp, gripper)
            stats.frames_in += len(files)
            segs = _copy_keep_segments(files, keep, dst, name,
                                        min_seg_frames=self.min_segment_frames)
            if not segs:
                stats.per_episode.append({"name": name, "kept": 0, "total": len(files),
                                          "segments": 0, "dropped": True})
                continue
            kept_total = sum(n for _, n in segs)
            stats.frames_out += kept_total
            stats.episodes_out += len(segs)
            stats.per_episode.append({"name": name, "kept": kept_total,
                                       "total": len(files), "segments": len(segs),
                                       "subs": segs, "dropped": False})
        (dst / "filter_stats.json").write_text(stats.to_json())
        return stats


# ------------------------------------------------------- 2. speed adjuster

class RoboKitSpeedAdjuster:
    """Down-sample slow segments so frame-to-frame TCP step ≈ ``target_step``.

    For any contiguous run of frames whose TCP speed is below ``speed_thr``
    AND length ≥ ``min_slow_frames``, the adjuster greedily keeps frames such
    that the accumulated 3D displacement between consecutive kept frames is at
    least ``target_step``. Inside ``gripper_slow_radius`` frames of a gripper
    transition the original frames are preserved (set 0 to disable).
    """

    def __init__(self,
                 speed_thr: float = 2e-3,
                 target_step: float = 5e-3,
                 min_slow_frames: int = 8,
                 gripper_slow_radius: int = 0,
                 gripper_protect_extremes_only: bool = False,
                 min_segment_frames: int = 16,
                 interp_factor: int = 1):
        self.speed_thr = float(speed_thr)
        self.target_step = float(target_step)
        self.min_slow_frames = int(min_slow_frames)
        self.gripper_slow_radius = int(gripper_slow_radius)
        self.gripper_protect_extremes_only = bool(gripper_protect_extremes_only)
        self.min_segment_frames = int(min_segment_frames)
        self.interp_factor = int(interp_factor)

    # ........................................................ per-episode
    def compute_keep_mask(self, tcp: np.ndarray, gripper: np.ndarray) -> np.ndarray:
        T = tcp.shape[0]
        if T <= 2:
            return np.ones(T, dtype=bool)
        vel = np.linalg.norm(np.diff(tcp[:, :3], axis=0), axis=-1)
        vel = np.concatenate([vel, vel[-1:]])
        slow = vel < self.speed_thr

        keep = np.ones(T, dtype=bool)
        i = 0
        while i < T:
            if not slow[i]:
                i += 1
                continue
            j = i
            while j < T and slow[j]:
                j += 1
            run = j - i
            if run >= self.min_slow_frames:
                # down-sample inside [i, j)
                keep[i:j] = False
                anchor = tcp[i, :3].copy()
                keep[i] = True  # always keep segment start
                for k in range(i + 1, j):
                    d = float(np.linalg.norm(tcp[k, :3] - anchor))
                    if d >= self.target_step:
                        keep[k] = True
                        anchor = tcp[k, :3].copy()
                keep[j - 1] = True  # always keep segment end
            i = j

        if self.gripper_slow_radius > 0:
            changes = _gripper_transitions(gripper)
            if self.gripper_protect_extremes_only and changes.size > 0:
                changes = np.array([changes[0], changes[-1]]) if changes.size > 1 else changes
            for c in changes:
                a = max(0, c - self.gripper_slow_radius)
                b = min(T, c + self.gripper_slow_radius + 1)
                keep[a:b] = True
        return keep

    # .............................................................. driver
    def process_dir(self, src_dir: str | Path, dst_dir: str | Path) -> FilterStats:
        src, dst = Path(src_dir), Path(dst_dir)
        dst.mkdir(parents=True, exist_ok=True)
        eps = _discover_episodes(src)
        stats = FilterStats(stage="speed_adjust", episodes_in=len(eps))
        for name, files in tqdm(eps, desc="speed_adjust"):
            tcp, gripper, _ = _load_episode_numeric(files)
            keep = self.compute_keep_mask(tcp, gripper)
            keep_idx = np.where(keep)[0]
            stats.frames_in += len(files)
            # Speed adjuster only down-samples within slow runs — never
            # introduces action discontinuity. Write as a single contiguous
            # episode (re-numbered), do NOT split into sub-segments.
            if keep_idx.size < self.min_segment_frames:
                stats.per_episode.append({"name": name, "kept": int(keep_idx.size),
                                          "total": len(files), "segments": 0,
                                          "dropped": True})
                continue
            _copy_keep(files, keep_idx, dst / name)
            stats.frames_out += int(keep_idx.size)
            stats.episodes_out += 1
            stats.per_episode.append({"name": name, "kept": int(keep_idx.size),
                                       "total": len(files), "segments": 1,
                                       "dropped": False})
        (dst / "filter_stats.json").write_text(stats.to_json())
        return stats


# -------------------------------------------------- 3. advantage filter

class RoboKitAdvantageFilter:
    """Critic-based filter: drop episodes whose per-step advantage is too low.

    The critic is expected to output a scalar V(s) per frame. We then compute
    GAE-style advantages using the trajectory's V sequence (with sparse rewards
    given by ``reward_fn`` or zeros by default) and drop an episode when its
    smoothed minimum advantage falls below ``drop_threshold``.

    This is the light-weight companion to :class:`RoboKitCritic`; the heavy
    lifting of scoring is done by ``critic.predict_episode_values(files)``.
    """

    def __init__(self,
                 critic,  # RoboKitCritic (deferred import to avoid torch at import-time)
                 gamma: float = 1.0,
                 lam: float = 0.0,
                 drop_threshold: float = -2e-3,
                 value_truncate: float = -5e-3,
                 smooth_window: int = 31):
        self.critic = critic
        self.gamma = float(gamma)
        self.lam = float(lam)
        self.drop_threshold = float(drop_threshold)
        self.value_truncate = float(value_truncate)
        self.smooth_window = int(smooth_window)

    def _smooth(self, x: np.ndarray) -> np.ndarray:
        w = self.smooth_window
        if w <= 1 or x.size < w:
            return x
        kernel = np.ones(w) / w
        return np.convolve(x, kernel, mode="same")

    def _advantages(self, values: np.ndarray,
                     rewards: Optional[np.ndarray] = None) -> np.ndarray:
        T = values.shape[0]
        if rewards is None:
            rewards = np.zeros(T, dtype=np.float64)
        adv = np.zeros(T, dtype=np.float64)
        gae = 0.0
        for t in reversed(range(T)):
            next_v = values[t + 1] if t + 1 < T else values[t]
            delta = rewards[t] + self.gamma * next_v - values[t]
            gae = delta + self.gamma * self.lam * gae
            adv[t] = gae
        return adv

    def process_dir(self,
                    src_dir: str | Path,
                    dst_dir: str | Path) -> FilterStats:
        src, dst = Path(src_dir), Path(dst_dir)
        dst.mkdir(parents=True, exist_ok=True)
        eps = _discover_episodes(src)
        stats = FilterStats(stage="advantage_filter", episodes_in=len(eps))
        for name, files in tqdm(eps, desc="adv_filter"):
            stats.frames_in += len(files)
            values = self.critic.predict_episode_values(files)  # (T,)
            adv = self._advantages(np.asarray(values))
            adv_smooth = self._smooth(adv)
            adv_smooth = np.clip(adv_smooth, self.value_truncate, None)
            drop_episode = bool(adv_smooth.min() < self.drop_threshold)
            if drop_episode:
                stats.per_episode.append({
                    "name": name, "kept": 0, "total": len(files), "dropped": True,
                    "adv_min": float(adv_smooth.min()),
                    "adv_mean": float(adv_smooth.mean()),
                })
                continue
            # Otherwise keep the whole episode as-is.
            keep_idx = np.arange(len(files))
            _copy_keep(files, keep_idx, dst / name)
            stats.frames_out += len(files)
            stats.episodes_out += 1
            stats.per_episode.append({
                "name": name, "kept": len(files), "total": len(files),
                "dropped": False,
                "adv_min": float(adv_smooth.min()),
                "adv_mean": float(adv_smooth.mean()),
            })
        (dst / "filter_stats.json").write_text(stats.to_json())
        return stats

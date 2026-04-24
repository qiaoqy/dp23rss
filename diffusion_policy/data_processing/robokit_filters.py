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

def _remove_short_runs(mask: np.ndarray, target_value: bool,
                        min_length: int) -> np.ndarray:
    """Flip runs of ``target_value`` shorter than ``min_length``.

    Mirrors exp41's
    ``rev2fwd_il.data.advantage_estimation._remove_short_runs``:

    * ``target_value=False, min_length=L`` — drop segments shorter than ``L``
      get converted back to keep (they are too short to be meaningful drops).
    * ``target_value=True,  min_length=L`` — keep segments shorter than ``L``
      that are *sandwiched* between drop runs become drop too (they are too
      short to train on after the surrounding drop has cut them out).
    """
    out = mask.copy()
    T = len(out)
    i = 0
    while i < T:
        if out[i] != target_value:
            i += 1
            continue
        j = i
        while j < T and out[j] == target_value:
            j += 1
        if j - i < min_length:
            if target_value:
                # only flip if truly sandwiched — never trim episode boundaries
                if i > 0 and j < T:
                    out[i:j] = not target_value
            else:
                out[i:j] = not target_value
        i = j
    return out


class RoboKitAdvantageFilter:
    """Critic-based **frame-level** filter, ported from exp41.

    Pipeline (per episode):
      1. ``values = critic.predict_episode_values(files)``  (T,)
      2. ``adv    = GAE(values, gamma, lam, rewards)``       (T,)
      3. (optional) ``adv = clip(adv, value_truncate, None)`` if ``value_truncate``
         is not ``None`` — by default *no clip* (matches exp41).
      4. ``smooth = moving_average(adv, smooth_window)``     (centered, reflect-padded)
      5. ``keep   = smooth >= drop_threshold``               — initial frame mask
      6. drop runs <``min_drop_length`` revert to keep      (too-short rejections)
      7. keep runs <``min_keep_length`` (sandwiched) revert to drop (unusable stubs)
      8. split keep mask into contiguous runs ≥ ``min_segment_frames`` and write
         each as its own ``ep_<name>_pXX/`` (action chunks never cross a drop).

    Defaults are exp37/40/41's reference values:
    ``gamma=0.995, lam=0.95, terminal_bootstrap=True, drop_threshold=0.0,
    smooth_window=51, min_drop_length=50, min_keep_length=50,
    value_truncate=None, step_reward=0`` — these match
    ``rev2fwd_il.data.advantage_estimation.{compute_gae_from_values,
    compute_frame_filter_mask}`` exactly.

    The per-step penalty is **already baked into the critic's value targets**
    (see :func:`compute_mc_returns`), so the filter takes ``rewards=zeros`` and
    a clean ``τ=0`` threshold has cross-task semantics ("smoothed advantage
    crossing zero" = "value sequence has stopped progressing toward the goal").

    Reward signal:
      * ``rewards=None`` (default): zeros — penalty is in V via MC labelling.
      * ``step_reward != 0``: every frame gets ``-|step_reward|`` (legacy path,
        only useful when paired with the legacy Bellman labelling).
    """

    def __init__(self,
                 critic,  # RoboKitCritic (deferred import to avoid torch at import-time)
                 gamma: float = 0.995,
                 lam: float = 0.95,
                 drop_threshold: float = 0.0,
                 drop_quantile: Optional[float] = None,
                 value_truncate: Optional[float] = None,
                 smooth_window: int = 51,
                 min_drop_length: int = 50,
                 min_keep_length: int = 50,
                 min_segment_frames: int = 24,
                 step_reward: float = 0.0,
                 terminal_bootstrap: bool = True):
        self.critic = critic
        self.gamma = float(gamma)
        self.lam = float(lam)
        self.drop_threshold = float(drop_threshold)
        self.drop_quantile = (None if drop_quantile is None
                               else float(drop_quantile))
        self.value_truncate = (None if value_truncate is None
                                else float(value_truncate))
        self.smooth_window = int(smooth_window)
        self.min_drop_length = int(min_drop_length)
        self.min_keep_length = int(min_keep_length)
        self.min_segment_frames = int(min_segment_frames)
        self.step_reward = float(step_reward)
        self.terminal_bootstrap = bool(terminal_bootstrap)

    # --------------------------------------------------------- math helpers
    def _smooth(self, x: np.ndarray) -> np.ndarray:
        """Centered moving average with reflect padding (matches exp41)."""
        w = self.smooth_window
        T = x.shape[0]
        if w <= 1 or T == 0:
            return x.astype(np.float64, copy=True)
        kernel = np.ones(w, dtype=np.float64) / w
        pad = w // 2
        padded = np.pad(x.astype(np.float64), (pad, pad), mode="reflect")
        return np.convolve(padded, kernel, mode="valid")[:T]

    def _advantages(self, values: np.ndarray,
                     rewards: Optional[np.ndarray] = None) -> np.ndarray:
        """GAE backward recursion, matching ``compute_gae_from_values``."""
        T = values.shape[0]
        if rewards is None:
            if self.step_reward != 0.0:
                rewards = np.full(T, -abs(self.step_reward), dtype=np.float64)
            else:
                rewards = np.zeros(T, dtype=np.float64)
        adv = np.zeros(T, dtype=np.float64)
        gae = 0.0
        for t in reversed(range(T)):
            if t == T - 1:
                next_v = values[t] if self.terminal_bootstrap else 0.0
            else:
                next_v = values[t + 1]
            delta = rewards[t] + self.gamma * next_v - values[t]
            gae = delta + self.gamma * self.lam * gae
            adv[t] = gae
        return adv

    def compute_keep_mask(self, values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return (keep_mask[T], smoothed_adv[T]) for a single episode."""
        adv = self._advantages(np.asarray(values, dtype=np.float64))
        if self.value_truncate is not None:
            adv = np.clip(adv, self.value_truncate, None)
        smoothed = self._smooth(adv)
        keep = smoothed >= self.drop_threshold
        keep = _remove_short_runs(keep, target_value=False,
                                   min_length=self.min_drop_length)
        keep = _remove_short_runs(keep, target_value=True,
                                   min_length=self.min_keep_length)
        return keep, smoothed

    # ------------------------------------------------------------- driver
    def process_dir(self,
                    src_dir: str | Path,
                    dst_dir: str | Path) -> FilterStats:
        src, dst = Path(src_dir), Path(dst_dir)
        dst.mkdir(parents=True, exist_ok=True)
        eps = _discover_episodes(src)
        stats = FilterStats(stage="advantage_filter", episodes_in=len(eps))

        # ---- Pass 1: predict V(s), compute *raw* (unsmoothed, unclipped)
        # advantages for every frame; cache them so we never re-run the critic.
        cached: list[tuple[str, list, np.ndarray]] = []  # (name, files, raw_adv)
        all_adv: list[np.ndarray] = []
        for name, files in tqdm(eps, desc="adv_predict"):
            T = len(files)
            stats.frames_in += T
            values = np.asarray(self.critic.predict_episode_values(files),
                                 dtype=np.float64)
            adv = self._advantages(values)
            cached.append((name, files, adv))
            all_adv.append(adv)

        # ---- Resolve threshold. If ``drop_quantile`` is set, take that
        # quantile of the *smoothed* per-frame advantages across the whole
        # dataset (matches the user-facing "保留 80% 的帧" intuition; smoothing
        # before quantile makes the threshold consistent with the keep rule).
        if self.drop_quantile is not None:
            smoothed_all = []
            for _, _, adv in cached:
                a = adv.copy()
                if self.value_truncate is not None:
                    a = np.clip(a, self.value_truncate, None)
                smoothed_all.append(self._smooth(a))
            pool = np.concatenate(smoothed_all) if smoothed_all else np.array([0.0])
            tau = float(np.quantile(pool, self.drop_quantile))
            print(f"[RoboKitAdvantageFilter] drop_quantile={self.drop_quantile} "
                  f"-> drop_threshold = {tau:+.6e} "
                  f"(pool size = {pool.size}, mean={pool.mean():+.3e}, "
                  f"std={pool.std():+.3e})")
            self.drop_threshold = tau
            stats.frames_out  # touch (just to keep linter happy)

        # ---- Pass 2: smooth + mask + write segments using the resolved tau.
        for name, files, adv in tqdm(cached, desc="adv_filter"):
            T = len(files)
            keep_mask, smoothed = self._mask_from_adv(adv)
            n_keep = int(keep_mask.sum())

            if n_keep == 0:
                stats.per_episode.append({
                    "name": name, "kept": 0, "total": T, "segments": 0,
                    "dropped": True,
                    "adv_min": float(smoothed.min()),
                    "adv_mean": float(smoothed.mean()),
                })
                continue

            segs = _copy_keep_segments(files, keep_mask, dst, name,
                                        min_seg_frames=self.min_segment_frames)
            if not segs:
                stats.per_episode.append({
                    "name": name, "kept": n_keep, "total": T, "segments": 0,
                    "dropped": True,
                    "adv_min": float(smoothed.min()),
                    "adv_mean": float(smoothed.mean()),
                })
                continue
            kept_total = sum(n for _, n in segs)
            stats.frames_out += kept_total
            stats.episodes_out += len(segs)
            stats.per_episode.append({
                "name": name, "kept": kept_total, "total": T,
                "segments": len(segs), "subs": segs, "dropped": False,
                "adv_min": float(smoothed.min()),
                "adv_mean": float(smoothed.mean()),
            })
        # persist resolved threshold + quantile for reproducibility
        stats_dict = json.loads(stats.to_json())
        stats_dict["resolved_drop_threshold"] = float(self.drop_threshold)
        stats_dict["drop_quantile"] = (None if self.drop_quantile is None
                                        else float(self.drop_quantile))
        (dst / "filter_stats.json").write_text(json.dumps(stats_dict, indent=2))
        return stats

    # internal: shared smooth+mask path used by both compute_keep_mask and
    # process_dir's pass 2.
    def _mask_from_adv(self, adv: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        a = adv.copy()
        if self.value_truncate is not None:
            a = np.clip(a, self.value_truncate, None)
        smoothed = self._smooth(a)
        keep = smoothed >= self.drop_threshold
        keep = _remove_short_runs(keep, target_value=False,
                                   min_length=self.min_drop_length)
        keep = _remove_short_runs(keep, target_value=True,
                                   min_length=self.min_keep_length)
        return keep, smoothed

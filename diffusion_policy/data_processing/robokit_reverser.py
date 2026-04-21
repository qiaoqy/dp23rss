"""Time-reversal for RoboKit per-frame NPZ data.

Ported from ``rev2fwd-il/scripts/scripts_task_inovo/2_time_reverse.py`` (which
implements the Rev2Fwd reversal rules for Inovo data). The logic is extracted
into a reusable class so the exp41 pipeline can call it programmatically.

Reversal rules (velocity actions):
    rel_actions[:6] -> negate AND reverse (move backwards along the trajectory)
    rel_actions[6]  -> reverse AND keep value (gripper sequence mirrored)
    actions         -> same treatment as rel_actions
    robot_obs[...]  -> reverse only
    primary_rgb / gripper_rgb / depth / force_torque -> reverse only
    language_text   -> optionally override with ``new_language``
"""

from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional

import numpy as np
from tqdm import tqdm


# ---------------------------------------------------------------------- utils

def _load_frame(path: str | Path) -> dict:
    f = np.load(str(path), allow_pickle=True)
    frame = {
        "primary_rgb_bytes": f["primary_rgb"].item(),
        "gripper_rgb_bytes": f["gripper_rgb"].item(),
        "primary_depth_bytes": f["primary_depth"].item() if "primary_depth" in f else None,
        "gripper_depth_bytes": f["gripper_depth"].item() if "gripper_depth" in f else None,
        "robot_obs": np.asarray(pickle.loads(f["robot_obs"].item()), dtype=np.float64),
        "actions": np.asarray(pickle.loads(f["actions"].item()), dtype=np.float64),
        "rel_actions": np.asarray(pickle.loads(f["rel_actions"].item()), dtype=np.float64),
        "force_torque": np.asarray(pickle.loads(f["force_torque"].item()), dtype=np.float64),
        "language_text": pickle.loads(f["language_text"].item()),
    }
    return frame


def _save_frame(path: str | Path, frame: dict) -> None:
    npz_data = {
        "primary_rgb": np.array(frame["primary_rgb_bytes"]),
        "gripper_rgb": np.array(frame["gripper_rgb_bytes"]),
        "robot_obs": np.array(pickle.dumps(frame["robot_obs"])),
        "actions": np.array(pickle.dumps(frame["actions"])),
        "rel_actions": np.array(pickle.dumps(frame["rel_actions"])),
        "force_torque": np.array(pickle.dumps(frame["force_torque"])),
        "language_text": np.array(pickle.dumps(np.array(frame["language_text"]))),
    }
    if frame.get("primary_depth_bytes") is not None:
        npz_data["primary_depth"] = np.array(frame["primary_depth_bytes"])
    if frame.get("gripper_depth_bytes") is not None:
        npz_data["gripper_depth"] = np.array(frame["gripper_depth_bytes"])
    np.savez(str(path), **npz_data)


def _discover_episodes(root: Path) -> List[dict]:
    episodes = []
    for entry in sorted(root.iterdir()):
        if not entry.is_dir() or entry.name in ("extracted", "__pycache__", "viz_videos"):
            continue
        npz_files = sorted(entry.glob("*.npz"))
        if not npz_files:
            continue
        episodes.append({"name": entry.name, "path": entry, "npz_files": npz_files})
    return episodes


# ---------------------------------------------------------------------- class

@dataclass
class ReverseStats:
    episodes_total: int = 0
    episodes_ok: int = 0
    episodes_failed: int = 0
    verify_warnings: int = 0


class RoboKitReverser:
    """Class-style wrapper around the 2_time_reverse.py reversal rules."""

    def __init__(self,
                 new_language: Optional[str] = None,
                 z_offset: float = 0.0,
                 verify: bool = True,
                 dt: float = 1.0 / 30.0,
                 verbose: bool = False):
        self.new_language = new_language
        self.z_offset = float(z_offset)
        self.verify = verify
        self.dt = float(dt)
        self.verbose = verbose

    # ................................................. per-episode primitives
    def reverse_episode(self, npz_files: Iterable[str | Path]) -> List[dict]:
        """Return a list of reversed frame dicts (ready for ``_save_frame``)."""
        files = list(npz_files)
        T = len(files)
        frames = [_load_frame(p) for p in files]

        if self.z_offset != 0.0: # current experiment: -0.02
            for f in frames:
                f["robot_obs"] = f["robot_obs"].copy()
                f["robot_obs"][2] += self.z_offset  # adjust z of TCP position

        reversed_frames = []
        for t in range(T):
            src = frames[T - 1 - t]
            new_frame = {
                "primary_rgb_bytes": src["primary_rgb_bytes"],
                "gripper_rgb_bytes": src["gripper_rgb_bytes"],
                "primary_depth_bytes": src.get("primary_depth_bytes"),
                "gripper_depth_bytes": src.get("gripper_depth_bytes"),
                "robot_obs": src["robot_obs"].copy(),
                "force_torque": src["force_torque"].copy(),
                "language_text": (
                    self.new_language if self.new_language is not None
                    else src["language_text"]
                ),
            }

            if t < T - 1:
                a_src = frames[T - 2 - t]
                actions_new = np.zeros(7, dtype=np.float64)
                actions_new[:6] = -a_src["actions"][:6]
                actions_new[6] = a_src["actions"][6]
                rel_new = np.zeros(7, dtype=np.float64)
                rel_new[:6] = -a_src["rel_actions"][:6]
                rel_new[6] = a_src["rel_actions"][6]
            else:
                actions_new = np.zeros(7, dtype=np.float64)
                actions_new[6] = frames[0]["actions"][6]
                rel_new = np.zeros(7, dtype=np.float64)
                rel_new[6] = frames[0]["rel_actions"][6]

            new_frame["actions"] = actions_new
            new_frame["rel_actions"] = rel_new
            reversed_frames.append(new_frame)

        return reversed_frames

    def verify_episode(self,
                       original_npz_files: List[Path],
                       reversed_frames: List[dict]) -> tuple[bool, str]:
        """Cheap correctness checks (boundaries + velocity integration)."""
        orig_first = _load_frame(original_npz_files[0])
        orig_last = _load_frame(original_npz_files[-1])
        rev_first, rev_last = reversed_frames[0], reversed_frames[-1]

        msgs: list[str] = []
        ok = True
        if not np.allclose(orig_last["robot_obs"][:3],
                           rev_first["robot_obs"][:3], atol=1e-5):
            msgs.append("FAIL orig_end != rev_start")
            ok = False
        if not np.allclose(orig_first["robot_obs"][:3],
                           rev_last["robot_obs"][:3], atol=1e-5):
            msgs.append("FAIL orig_start != rev_end")
            ok = False

        T = len(reversed_frames)
        tcp = np.array([f["robot_obs"][:3] for f in reversed_frames])
        vel = np.array([f["actions"][:3] for f in reversed_frames])
        integ = np.zeros((T, 3))
        integ[0] = tcp[0]
        for t in range(T - 1):
            integ[t + 1] = integ[t] + vel[t] * self.dt
        err = float(np.abs(integ - tcp).max())
        msgs.append(f"int_err={err:.4f}m")
        if err > 0.01:
            msgs.append("WARN integ>1cm")
        return ok, "; ".join(msgs)

    # ................................................................ driver
    def reverse_dir(self,
                    src_dir: str | Path,
                    dst_dir: str | Path) -> ReverseStats:
        src = Path(src_dir)
        dst = Path(dst_dir)
        dst.mkdir(parents=True, exist_ok=True)

        stats = ReverseStats()
        episodes = _discover_episodes(src)
        stats.episodes_total = len(episodes)
        iterator = episodes if self.verbose else tqdm(episodes, desc="reverse_dir")
        for ep in iterator:
            try:
                rev = self.reverse_episode(ep["npz_files"])
                if self.verify:
                    ok, msg = self.verify_episode(ep["npz_files"], rev)
                    if self.verbose or not ok:
                        print(f"[reverse_dir] {ep['name']}: {msg}")
                    if not ok:
                        stats.verify_warnings += 1
                ep_out = dst / ep["name"]
                ep_out.mkdir(parents=True, exist_ok=True)
                for i, frame in enumerate(rev):
                    _save_frame(ep_out / ep["npz_files"][i].name, frame)
                stats.episodes_ok += 1
            except Exception as e:  # noqa: BLE001
                stats.episodes_failed += 1
                print(f"[reverse_dir] ERROR on {ep['name']}: {e}")
        print(f"[RoboKitReverser] done: {stats}")
        return stats

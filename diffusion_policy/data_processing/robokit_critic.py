"""Simple V(s) critic trained on A-task rollouts for advantage filtering.

Design choices mirror exp41:
* Input: ``primary_rgb`` (cropped to ``crop`` px, optional color-jitter) plus
  ``robot_obs[:6]`` (TCP pose). No force, no language.
* Backbone: ResNet-18 feature pool → 512-dim embedding, concatenated with a
  4-layer MLP over TCP pose → ``V(s)`` scalar.
* Loss: MSE against Bellman returns from
  ``rev2fwd_il.data.value_labeling.compute_bellman_returns`` (imported lazily;
  re-implemented here so dp23rss does not need the other repo installed).
* Training: 1000 steps, AdamW(lr=5e-4, wd=1e-4), dropout=0.15, crop=112, color
  jitter always on (matches exp40/41).

Only the pieces that the orchestration actually calls are exposed:
* :meth:`RoboKitCritic.fit`  — train on a collect directory tree.
* :meth:`RoboKitCritic.load` — load a trained ckpt back.
* :meth:`RoboKitCritic.predict_episode_values` — score each frame of a list of
  ``*.npz`` files (used by :class:`RoboKitAdvantageFilter`).

``episode_index.json`` layout expected by ``fit``:

.. code-block:: json

    {
        "ep_000": {"length": 512, "sparse_label": "Y"},
        "ep_001": {"length": 489, "sparse_label": "N"}
    }

``sparse_label`` ∈ ``{"Y", "N"}`` (written by :class:`OnlineDataCollector`).
"""

from __future__ import annotations

import io
import json
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import DataLoader, Dataset
    from torchvision import models, transforms
    _TORCH_OK = True
except ImportError:  # pragma: no cover — torch is always present in robodiff
    _TORCH_OK = False


# --------------------------------------------------------- reward / returns

def compute_bellman_returns(lengths: list[int],
                             successes: list[bool],
                             gamma: float = 0.995,
                             success_reward: float = 1.0) -> list[np.ndarray]:
    """Episode-level Bellman return (re-implementation of ``value_labeling``).

    Success episode: V(t) = gamma^(T - t) * success_reward.
    Failure episode: V(t) = 0 for all t.
    """
    out = []
    for T, ok in zip(lengths, successes):
        if not ok:
            out.append(np.zeros(T, dtype=np.float32))
            continue
        exps = np.arange(T, 0, -1, dtype=np.float32)
        out.append((gamma ** exps) * float(success_reward))
    return out


# ------------------------------------------------------------------- utils

def _load_image_tcp(path: Path, crop: int) -> tuple[np.ndarray, np.ndarray]:
    with np.load(str(path), allow_pickle=True) as f:
        img_bytes = f["primary_rgb"].item()
        if isinstance(img_bytes, np.ndarray):
            img_bytes = img_bytes.tobytes()
        robot_obs = np.asarray(pickle.loads(f["robot_obs"].item()), dtype=np.float32)
    img = Image.open(io.BytesIO(img_bytes)).convert("RGB").resize((crop, crop))
    arr = np.asarray(img, dtype=np.uint8)
    return arr, robot_obs[:6].astype(np.float32)


def _read_episode_index(root: Path) -> dict:
    p = root / "episode_index.json"
    if not p.exists():
        raise FileNotFoundError(
            f"{p} missing; collect directory must carry per-episode Y/N labels")
    return json.loads(p.read_text())


# ------------------------------------------------------------------- torch

if _TORCH_OK:

    class _CriticDataset(Dataset):
        def __init__(self, root: Path, crop: int,
                     color_jitter: bool):
            self.root = root
            self.crop = crop
            index = _read_episode_index(root)
            self.episodes = []       # list of dict(name, files, T, success)
            for name, info in sorted(index.items()):
                ep_dir = root / name
                files = sorted(ep_dir.glob("*.npz"))
                if not files:
                    continue
                self.episodes.append({
                    "name": name,
                    "files": files,
                    "T": len(files),
                    "success": info.get("sparse_label", "N") == "Y",
                })
            # flat index of (ep_idx, frame_idx)
            self.flat = []
            for ei, ep in enumerate(self.episodes):
                for fi in range(ep["T"]):
                    self.flat.append((ei, fi))
            self.values = compute_bellman_returns(
                [e["T"] for e in self.episodes],
                [e["success"] for e in self.episodes],
            )
            aug = [transforms.ToPILImage()]
            if color_jitter:
                aug.append(transforms.ColorJitter(0.1, 0.1, 0.1, 0.05))
            aug.append(transforms.ToTensor())
            self.tf = transforms.Compose(aug)

        def __len__(self):
            return len(self.flat)

        def __getitem__(self, idx):
            ei, fi = self.flat[idx]
            ep = self.episodes[ei]
            arr, tcp = _load_image_tcp(ep["files"][fi], self.crop)
            img = self.tf(arr) * 2.0 - 1.0
            v = float(self.values[ei][fi])
            return {
                "image": img,
                "tcp": torch.from_numpy(tcp),
                "value": torch.tensor(v, dtype=torch.float32),
            }

    class _CriticNet(nn.Module):
        def __init__(self, dropout: float = 0.15):
            super().__init__()
            backbone = models.resnet18(weights=None)
            backbone.fc = nn.Identity()
            self.backbone = backbone  # outputs 512-dim
            self.tcp_mlp = nn.Sequential(
                nn.Linear(6, 128), nn.ReLU(inplace=True),
                nn.Linear(128, 128), nn.ReLU(inplace=True),
            )
            self.head = nn.Sequential(
                nn.Linear(512 + 128, 512), nn.ReLU(inplace=True),
                nn.Dropout(dropout),
                nn.Linear(512, 256), nn.ReLU(inplace=True),
                nn.Dropout(dropout),
                nn.Linear(256, 1),
            )

        def forward(self, img, tcp):
            feat = self.backbone(img)
            t = self.tcp_mlp(tcp)
            return self.head(torch.cat([feat, t], dim=-1)).squeeze(-1)


# ------------------------------------------------------------------- class

@dataclass
class CriticConfig:
    steps: int = 1000
    batch_size: int = 64
    lr: float = 5e-4
    weight_decay: float = 1e-4
    dropout: float = 0.15
    crop: int = 112
    color_jitter: bool = True
    num_workers: int = 8
    device: str = "cuda"
    gamma: float = 0.995
    success_reward: float = 1.0


class RoboKitCritic:
    """Trainable V(s) estimator used by :class:`RoboKitAdvantageFilter`.

    The class is designed as a thin controller so orchestration scripts (and
    the ``train_critic.sh`` entrypoint) can treat it as a black box:

    .. code-block:: python

        critic = RoboKitCritic()
        critic.fit(collect_A_dir, out_dir, config=CriticConfig(steps=1000))
        critic = RoboKitCritic.load(out_dir / "checkpoints" / "best" / "checkpoint.pt")
        values = critic.predict_episode_values(npz_files)   # (T,)
    """

    def __init__(self, config: Optional[CriticConfig] = None):
        if not _TORCH_OK:
            raise ImportError("PyTorch / torchvision required for RoboKitCritic.")
        self.config = config or CriticConfig()
        self.net: Optional[nn.Module] = None
        self.device = self.config.device

    # ----------------------------------------------------------------- fit
    def fit(self, collect_dir: str | Path, out_dir: str | Path) -> dict:
        cfg = self.config
        collect_dir = Path(collect_dir)
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        ds = _CriticDataset(collect_dir, cfg.crop, cfg.color_jitter)
        loader = DataLoader(ds, batch_size=cfg.batch_size,
                            num_workers=cfg.num_workers,
                            shuffle=True, drop_last=True,
                            pin_memory=True, persistent_workers=cfg.num_workers > 0)
        net = _CriticNet(cfg.dropout).to(self.device)
        opt = torch.optim.AdamW(net.parameters(), lr=cfg.lr,
                                 weight_decay=cfg.weight_decay)
        net.train()
        step = 0
        history = []
        done = False
        while not done:
            for batch in loader:
                img = batch["image"].to(self.device, non_blocking=True)
                tcp = batch["tcp"].to(self.device, non_blocking=True)
                tgt = batch["value"].to(self.device, non_blocking=True)
                pred = net(img, tcp)
                loss = F.mse_loss(pred, tgt)
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
                if step % 50 == 0:
                    print(f"[RoboKitCritic] step {step}/{cfg.steps} loss={loss.item():.4f}")
                    history.append({"step": step, "loss": float(loss.item())})
                step += 1
                if step >= cfg.steps:
                    done = True
                    break
        self.net = net
        ckpt_dir = out_dir / "checkpoints" / "best"
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        torch.save({"state_dict": net.state_dict(), "config": cfg.__dict__},
                   ckpt_dir / "checkpoint.pt")
        (out_dir / "train_history.json").write_text(json.dumps(history, indent=2))
        return {"steps": step, "final_loss": history[-1]["loss"] if history else None}

    # ----------------------------------------------------------------- load
    @classmethod
    def load(cls, ckpt_path: str | Path,
             device: str = "cuda") -> "RoboKitCritic":
        blob = torch.load(str(ckpt_path), map_location=device, weights_only=False)
        cfg = CriticConfig(**blob.get("config", {}))
        cfg.device = device
        self = cls(cfg)
        net = _CriticNet(cfg.dropout).to(device)
        net.load_state_dict(blob["state_dict"])
        net.eval()
        self.net = net
        return self

    # ----------------------------------------------------------- inference
    @torch.no_grad()
    def predict_episode_values(self, npz_files: list[Path]) -> np.ndarray:
        assert self.net is not None, "Call load() / fit() first."
        cfg = self.config
        vals = np.zeros(len(npz_files), dtype=np.float32)
        batch_imgs, batch_tcps, batch_idx = [], [], []

        def _flush():
            if not batch_imgs:
                return
            x = torch.from_numpy(np.stack(batch_imgs).astype(np.float32) / 127.5 - 1.0)
            x = x.permute(0, 3, 1, 2).to(self.device, non_blocking=True)
            t = torch.from_numpy(np.stack(batch_tcps)).to(self.device, non_blocking=True)
            v = self.net(x, t).detach().cpu().numpy()
            for out_i, v_i in zip(batch_idx, v):
                vals[out_i] = float(v_i)
            batch_imgs.clear(); batch_tcps.clear(); batch_idx.clear()

        for i, p in enumerate(npz_files):
            arr, tcp = _load_image_tcp(p, cfg.crop)
            batch_imgs.append(arr)
            batch_tcps.append(tcp)
            batch_idx.append(i)
            if len(batch_imgs) >= max(1, cfg.batch_size):
                _flush()
        _flush()
        return vals

"""Smoke test: MergeTCLImageDataset with easy + easy_reversed roots."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, "/mnt/dongxu-fs1/data-ssd/qiyuanqiao/workspace/RoboKit/src")

import numpy as np
from torch.utils.data import DataLoader

from diffusion_policy.dataset.tcl_dataset import MergeTCLImageDataset

EASY = "/mnt/dongxu-fs1/data-hdd/qiyuanqiao/workspace/rev2fwd-il/data/inovo_data/0209_tower_boby_easy"
EASY_REV = "/mnt/dongxu-fs1/data-hdd/qiyuanqiao/workspace/rev2fwd-il/data/inovo_data/0209_tower_boby_easy_reversed"
HARD = "/mnt/dongxu-fs1/data-hdd/qiyuanqiao/workspace/rev2fwd-il/data/inovo_data/0209_tower_boby_hard"

SHAPE_META = {
    "action": {"shape": [7]},
    "obs": {
        "image": {"shape": [3, 240, 320], "type": "rgb"},
        "gripper": {"shape": [3, 240, 320], "type": "rgb"},
        "joint_state": {"shape": [6], "type": "low_dim"},
        "force": {"shape": [6], "type": "low_dim"},
    },
}


def test_single_root():
    print("=== Test 1: single root (easy) ===")
    ds = MergeTCLImageDataset(
        data_roots=[EASY],
        h5_paths=[None],
        horizon=16, pad_before=1, pad_after=7,
        shape_meta=SHAPE_META,
        use_h5=False,
        stats_source="first",
    )
    print(f"  len={len(ds)}")
    sample = ds[0]
    print(f"  sample keys: {list(sample.keys())}")
    for k, v in sample.items():
        if hasattr(v, "shape"):
            print(f"    {k}: shape={v.shape}, dtype={v.dtype}")
    assert len(ds) > 0
    print("  PASS\n")


def test_merge_two_roots():
    print("=== Test 2: merge easy + hard (both preprocessed) ===")
    ds = MergeTCLImageDataset(
        data_roots=[EASY, HARD],
        h5_paths=[None, None],
        horizon=16, pad_before=1, pad_after=7,
        shape_meta=SHAPE_META,
        use_h5=False,
        stats_source="first",
    )
    print(f"  total len={len(ds)}, sub_lengths={ds.sub_lengths}")
    # sample from each sub
    s0 = ds[0]
    s1 = ds[ds.sub_lengths[0]]  # first sample from second sub
    print(f"  sample0 action range: [{s0['action'].min():.4f}, {s0['action'].max():.4f}]")
    print(f"  sample1 action range: [{s1['action'].min():.4f}, {s1['action'].max():.4f}]")
    assert len(ds) == sum(ds.sub_lengths)
    print("  PASS\n")


def test_merge_three_roots():
    print("=== Test 3: merge hard + easy (stats_source=first) ===")
    ds = MergeTCLImageDataset(
        data_roots=[HARD, EASY],
        h5_paths=[None, None],
        horizon=16, pad_before=1, pad_after=7,
        shape_meta=SHAPE_META,
        use_h5=False,
        stats_source="first",
    )
    print(f"  total len={len(ds)}, sub_lengths={ds.sub_lengths}")
    assert len(ds) == sum(ds.sub_lengths)
    print("  PASS\n")


def test_dataloader():
    print("=== Test 4: DataLoader batch (easy + hard, batch=4, workers=2) ===")
    ds = MergeTCLImageDataset(
        data_roots=[EASY, HARD],
        h5_paths=[None, None],
        horizon=16, pad_before=1, pad_after=7,
        shape_meta=SHAPE_META,
        use_h5=False,
        stats_source="first",
    )
    dl = DataLoader(ds, batch_size=4, shuffle=True, num_workers=2, pin_memory=False)
    batch = next(iter(dl))
    print(f"  batch keys: {list(batch.keys())}")
    for k, v in batch.items():
        if hasattr(v, "shape"):
            print(f"    {k}: shape={v.shape}")
    assert batch["action"].shape[0] == 4
    print("  PASS\n")


def test_normalizer():
    print("=== Test 5: get_normalizer ===")
    ds = MergeTCLImageDataset(
        data_roots=[EASY, HARD],
        h5_paths=[None, None],
        horizon=16, pad_before=1, pad_after=7,
        shape_meta=SHAPE_META,
        use_h5=False,
        stats_source="first",
    )
    normalizer = ds.get_normalizer()
    print(f"  normalizer type: {type(normalizer)}")
    print("  PASS\n")


if __name__ == "__main__":
    test_single_root()
    test_merge_two_roots()
    test_merge_three_roots()
    test_dataloader()
    test_normalizer()
    print("All tests passed!")

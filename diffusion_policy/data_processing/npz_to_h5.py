"""Thin wrapper that turns a RoboKit-format collect directory into a ``*.h5``.

Delegates to :class:`robokit.datasets.tcl_datasets.TCLDatasetHDF5` so that the
merged dataset loader (see :class:`MergeTCLImageDataset`) can read the data
with ``use_h5=True`` for fast training.

Invariant enforced here
-----------------------
* The H5 file is always named ``<dirname>_240p.h5`` under ``<root>/hdf5/``,
  matching the convention assumed in ``reverse_dp_force.yaml``.
* Images are re-encoded at ``(W, H) = (320, 240)`` JPEG bytes (PR-level plan
  §1 decision).
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

from robokit.datasets.tcl_datasets import TCLDatasetHDF5


def convert_root_to_h5(root: str | Path,
                       h5_path: Optional[str | Path] = None,
                       resize_wh: Tuple[int, int] = (320, 240),
                       batch_size: int = 16,
                       num_workers: int = 4,
                       use_extracted: bool = True) -> Path:
    """Convert a RoboKit collect directory to a single H5 file.

    Parameters
    ----------
    root : path
        Root directory holding ``ep_XXX/*.npz`` sub-directories (needs
        ``extracted/`` + ``statistics.json`` produced by
        ``RoboKit/scripts/01_preprocess_data.py``).
    h5_path : path, optional
        Destination. Defaults to ``<root>/hdf5/<root.name>_240p.h5``.
    resize_wh : (W, H), default ``(320, 240)``
        Image size written into the H5 JPEG blobs. Downstream training crops
        to 216×288 from a 240×320 canvas, matching
        ``reverse_dp_force.yaml``.
    """
    root = Path(root)
    if h5_path is None:
        h5_path = root / "hdf5" / f"{root.name}_240p.h5"
    h5_path = Path(h5_path)
    h5_path.parent.mkdir(parents=True, exist_ok=True)

    ds = TCLDatasetHDF5(
        root=str(root),
        h5_path=str(h5_path),
        use_h5=False,            # IMPORTANT: we are WRITING, not reading.
        use_extracted=use_extracted,
        is_img_decoded_in_h5=False,
    )
    ds.convert_to_hdf5(batch_size=batch_size, num_workers=num_workers,
                       resize_wh=resize_wh)
    return h5_path


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("-R", "--root", required=True)
    p.add_argument("--h5", default=None)
    p.add_argument("--resize-wh", nargs=2, type=int, default=(320, 240),
                   metavar=("W", "H"))
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--num-workers", type=int, default=4)
    args = p.parse_args()
    out = convert_root_to_h5(args.root, args.h5,
                              tuple(args.resize_wh),
                              batch_size=args.batch_size,
                              num_workers=args.num_workers)
    print(f"Wrote {out}")

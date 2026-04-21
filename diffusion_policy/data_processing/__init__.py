"""RoboKit-format data processing utilities for dp_tr / exp41 pipeline.

Modules here re-implement the key stages of
``rev2fwd-il/data/pick_place_isaac_lab_simulation/exp41`` against the RoboKit
per-frame ``*.npz`` format so that the whole iterative rollout → reverse →
filter → train loop can run inside the ``robodiff`` (dp23rss) environment.

The public surface intentionally mirrors the stage boundaries that
``dp_tr/scripts/run_iter.sh`` needs to orchestrate:

* :class:`RoboKitReverser`  — time-reverse a collect directory (optional z-fix).
* :class:`RoboKitStaticFilter` — drop long-static frames.
* :class:`RoboKitSpeedAdjuster` — equalise per-frame TCP step size.
* :class:`RoboKitCritic` / :class:`RoboKitAdvantageFilter` — critic-based
  advantage filtering.
* :class:`OnlineDataCollector` — write (obs, action, label) tuples during
  inference in a layout that :class:`TCLDataset` / :class:`TCLDatasetHDF5` can
  load back.
* :func:`convert_root_to_h5` — thin wrapper around
  :meth:`TCLDatasetHDF5.convert_to_hdf5` for post-merge NPZ→H5 conversion.
"""

from .robokit_reverser import RoboKitReverser  # noqa: F401
from .robokit_filters import (  # noqa: F401
    RoboKitStaticFilter,
    RoboKitSpeedAdjuster,
    RoboKitAdvantageFilter,
    FilterStats,
)
from .robokit_critic import RoboKitCritic  # noqa: F401
from .online_collector import OnlineDataCollector  # noqa: F401
from .npz_to_h5 import convert_root_to_h5  # noqa: F401

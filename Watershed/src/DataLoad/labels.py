from __future__ import annotations

import io
from typing import Any

import numpy as np


def load_label_sets_from_npz(content: bytes) -> tuple[np.ndarray, Any]:
    data = np.load(io.BytesIO(content), allow_pickle=True)
    label_map = data["label_map"]
    label_info = data["label_info"]
    return label_map, label_info

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .client import ApiClient
from .labels import load_label_sets_from_npz
from .raw_volume import (
    VolumeData,
    choose_rotation_angle,
    estimate_contrast,
    load_volume_from_raw,
    rotate_volume_for_ai,
)
from .settings import DataLoadConfig


@dataclass
class PipelineResult:
    volume_data: VolumeData
    volume_rotated: np.ndarray
    label_map: np.ndarray
    label_info: Any
    contrast: float


def run_pipeline(config: DataLoadConfig) -> PipelineResult:
    client = ApiClient(config)

    datasets = client.get_catalogue_nodes()
    if not datasets:
        raise RuntimeError("No datasets found for given catalogue key.")
    if config.select_index >= len(datasets):
        raise IndexError(
            f"select_index={config.select_index} is out of range (num datasets={len(datasets)})."
        )

    dataset_node_id = datasets[config.select_index]

    volume_attachment_uid = client.get_node_attachment_uid(dataset_node_id)
    volume_content = client.download_attachment(volume_attachment_uid)
    volume_data = load_volume_from_raw(volume_content)

    rot_angle = choose_rotation_angle(
        volume_data.volume, volume_data.px_size, volume_data.rot_angles
    )
    volume_rotated = rotate_volume_for_ai(volume_data.volume, rot_angle)

    label_attachment_uid = client.find_label_attachment_uid(dataset_node_id)
    label_content = client.download_attachment(label_attachment_uid)
    label_map, label_info = load_label_sets_from_npz(label_content)

    contrast = estimate_contrast(volume_data.volume)

    return PipelineResult(
        volume_data=volume_data,
        volume_rotated=volume_rotated,
        label_map=label_map,
        label_info=label_info,
        contrast=contrast,
    )

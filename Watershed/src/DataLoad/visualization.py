from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from scipy.ndimage import binary_dilation


def gray_to_color(img: np.ndarray) -> np.ndarray:
    image = np.zeros(img.shape + (3,), dtype=np.uint8)
    scale = np.clip((np.clip(img, -1, 5) + 1) / 6 * 255, 0, 255)
    image[:, :, 0] = scale
    image[:, :, 1] = scale
    image[:, :, 2] = scale
    return image


def append_color(image: np.ndarray, label_img: np.ndarray, color: list[float]) -> np.ndarray:
    weight = np.clip(label_img, 0, 1)
    image[:, :, 0] = image[:, :, 0] * (1 - weight) + (255 - image[:, :, 0]) * weight * color[0]
    image[:, :, 1] = image[:, :, 1] * (1 - weight) + (255 - image[:, :, 1]) * weight * color[1]
    image[:, :, 2] = image[:, :, 2] * (1 - weight) + (255 - image[:, :, 2]) * weight * color[2]
    return image


def show_slice(volume: np.ndarray, index: int, cmap: str = "gray") -> None:
    plt.imshow(volume[index], cmap=cmap)
    plt.title(str(index))
    _safe_show()


def overlay_multi_label(
    volume_std: np.ndarray,
    labels: list[np.ndarray],
    colors: list[list[float]],
    index: int,
    threshold: float = 0.0,
    border_only: bool = False,
) -> np.ndarray:
    image = gray_to_color(volume_std[index])
    for idx, label in enumerate(labels):
        mask = label[index] > threshold
        if border_only:
            mask = binary_dilation(mask, structure=np.ones((3, 3))) & (~mask)
        image = append_color(image, mask, colors[idx])
    return image


def show_overlay(image: np.ndarray, title: str) -> None:
    plt.imshow(image)
    plt.title(title)
    _safe_show()


def _safe_show() -> None:
    # In headless environments (Agg backend), show() is non-interactive and emits warnings.
    backend = matplotlib.get_backend().lower()
    if "agg" in backend:
        plt.close()
        return
    plt.show()

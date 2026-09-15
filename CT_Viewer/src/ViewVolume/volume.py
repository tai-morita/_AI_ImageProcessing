from __future__ import annotations

from pathlib import Path
from typing import Literal

import numpy as np

Plane = Literal["axial", "sagittal", "coronal"]


def load_volume(path: str | Path) -> np.ndarray:
    ## @brief Load and validate a 3D volume from a NumPy or TIFF file.
    ## @param path Input file path. Supported extensions are `.npy`, `.tif`,
    ## `.tiff`.
    ## @return The loaded volume with axis order `(z, y, x)`.
    ## @exception ValueError If the extension, dimensionality, dtype, or finite
    ## value requirement is invalid.
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".npy":
        volume = np.load(path, allow_pickle=False)
    elif suffix in {".tif", ".tiff"}:
        import tifffile

        volume = tifffile.imread(path)
    else:
        raise ValueError("Input must be a .npy, .tif, or .tiff file.")

    volume = np.asarray(volume)
    if volume.ndim != 3:
        raise ValueError(f"Expected a 3D volume, got shape={volume.shape}.")
    if not np.issubdtype(volume.dtype, np.number):
        raise ValueError(f"Expected a numeric volume, got dtype={volume.dtype}.")
    if not np.isfinite(volume).any():
        raise ValueError("The volume contains no finite values.")
    return volume


def finite_range(volume: np.ndarray) -> tuple[float, float]:
    ## @brief Return the finite minimum and maximum values in a volume.
    ## @param volume Numeric volume to inspect.
    ## @return A `(minimum, maximum)` pair.
    ## @exception ValueError If the volume has no finite values.
    finite_values = np.asarray(volume)[np.isfinite(volume)]
    return float(finite_values.min()), float(finite_values.max())


def plane_axis(plane: Plane) -> int:
    ## @brief Return the volume axis represented by a viewing plane.
    ## @param plane Plane name: `axial`, `sagittal`, or `coronal`.
    ## @return Axis index in `(z, y, x)` order.
    return {"axial": 0, "sagittal": 2, "coronal": 1}[plane]


def slice_count(volume: np.ndarray, plane: Plane) -> int:
    ## @brief Return the number of slices available for a viewing plane.
    ## @param volume Volume whose slices are counted.
    ## @param plane Viewing plane.
    ## @return Number of slices along the plane's fixed axis.
    return int(volume.shape[plane_axis(plane)])


def extract_slice(volume: np.ndarray, plane: Plane, index: int) -> np.ndarray:
    ## @brief Extract one 2D image from a 3D `(z, y, x)` volume.
    ## @param volume Source volume.
    ## @param plane Viewing plane.
    ## @param index Zero-based index of the fixed axis.
    ## @return The selected 2D image.
    ## @exception ValueError If `plane` is not a supported plane name.
    if plane == "axial":
        image = volume[index, :, :]
    elif plane == "sagittal":
        image = volume[:, :, index]
    elif plane == "coronal":
        image = volume[:, index, :]
    else:
        raise ValueError(f"Unknown plane: {plane}")
    return np.asarray(image)


def display_to_xyz(
    plane: Plane,
    slice_index: int,
    display_x: float,
    display_y: float,
    shape: tuple[int, int, int],
) -> tuple[int, int, int]:
    ## @brief Convert image coordinates to a clipped volume coordinate.
    ## @param plane Viewing plane.
    ## @param slice_index Current fixed-axis index.
    ## @param display_x Image column coordinate.
    ## @param display_y Image row coordinate.
    ## @param shape Volume shape in `(z, y, x)` order.
    ## @return Coordinate in logical `(x, y, z)` order.
    ## @exception ValueError If `plane` is not a supported plane name.
    x_size = shape[2]
    y_size = shape[1]
    z_size = shape[0]
    column = int(round(display_x))
    row = int(round(display_y))

    if plane == "axial":
        x, y, z = column, row, slice_index
    elif plane == "sagittal":
        x, y, z = slice_index, column, row
    elif plane == "coronal":
        x, y, z = column, slice_index, row
    else:
        raise ValueError(f"Unknown plane: {plane}")

    return (
        int(np.clip(x, 0, x_size - 1)),
        int(np.clip(y, 0, y_size - 1)),
        int(np.clip(z, 0, z_size - 1)),
    )


def window_limits(window_width: float, window_level: float) -> tuple[float, float]:
    ## @brief Calculate Matplotlib contrast limits from WW and WL.
    ## @param window_width Window width. Must be positive and finite.
    ## @param window_level Window level. Must be finite.
    ## @return `(vmin, vmax)` for the image color limits.
    ## @exception ValueError If either window parameter is invalid.
    if not np.isfinite(window_width) or window_width <= 0:
        raise ValueError("Window width must be a positive finite number.")
    if not np.isfinite(window_level):
        raise ValueError("Window level must be finite.")
    half_width = window_width / 2.0
    return window_level - half_width, window_level + half_width



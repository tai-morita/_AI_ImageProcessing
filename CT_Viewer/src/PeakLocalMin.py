from __future__ import annotations

import numpy as np
from scipy import ndimage as ndi


class PeakLocalMin:
    ## @brief Search for a local minimum in a 3D volume.
    ##
    ## The input volume uses `(z, y, x)` order. The search moves only to a
    ## strictly smaller finite value and excludes background voxels.

    def __init__(self, volume: np.ndarray, radius: int = 3) -> None:
        ## @brief Initialize the local-minimum searcher.
        ## @param volume Numeric volume in `(z, y, x)` order.
        ## @param radius Search radius for each axis. Radius 3 gives up to
        ## `7 x 7 x 7 - 1 = 342` candidates.
        if np.asarray(volume).ndim != 3:
            raise ValueError(f"Expected a 3D volume, got shape={np.asarray(volume).shape}.")
        if radius < 0:
            raise ValueError("radius must be non-negative")
        self.volume = np.asarray(volume)
        self.radius = radius
        self._neighbor_zero_count = self._calculate_neighbor_zero_count()

    def _calculate_neighbor_zero_count(self) -> np.ndarray:
        ## @brief Count zero-valued voxels in each voxel's 26-neighborhood.
        ##
        ## Voxels outside the array are treated as background zero, which keeps
        ## background detection consistent at the volume boundary.
        zero_mask = np.equal(self.volume, 0)
        return ndi.convolve(
            zero_mask.astype(np.uint8),
            np.ones((3, 3, 3), dtype=np.uint8),
            mode="constant",
            cval=1,
        ) - zero_mask

    def is_background(self, coordinate: tuple[int, int, int]) -> bool:
        ## @brief Return whether a `(z, y, x)` coordinate is background.
        ## @param coordinate Coordinate to test.
        ## @return True when the voxel is zero and at least nine neighbors are zero.
        z, y, x = coordinate
        return bool(
            self.volume[z, y, x] == 0
            and self._neighbor_zero_count[z, y, x] >= 9
        )

    def find(self, start_coord: tuple[int, int, int]) -> tuple[int, int, int]:
        ## @brief Move from a starting coordinate to a local minimum.
        ## @param start_coord Starting coordinate in `(z, y, x)` order.
        ## @return Local minimum coordinate in `(z, y, x)` order.
        ## @exception ValueError If the start value is non-finite or background.
        ## @exception IndexError If the starting coordinate is outside the volume.
        z, y, x = (int(value) for value in start_coord)
        if not (
            0 <= z < self.volume.shape[0]
            and 0 <= y < self.volume.shape[1]
            and 0 <= x < self.volume.shape[2]
        ):
            raise IndexError(f"start_coord is outside the volume: {start_coord}")
        if not np.isfinite(self.volume[z, y, x]):
            raise ValueError("The starting voxel value must be finite.")
        if self.is_background((z, y, x)):
            raise ValueError("The starting voxel is background.")

        current_coord = (z, y, x)
        while True:
            z, y, x = current_coord
            z_start = max(0, z - self.radius)
            y_start = max(0, y - self.radius)
            x_start = max(0, x - self.radius)
            z_end = min(self.volume.shape[0], z + self.radius + 1)
            y_end = min(self.volume.shape[1], y + self.radius + 1)
            x_end = min(self.volume.shape[2], x + self.radius + 1)

            neighborhood = np.asarray(
                self.volume[z_start:z_end, y_start:y_end, x_start:x_end]
            ).copy()
            center = (z - z_start, y - y_start, x - x_start)
            neighborhood[center] = np.inf
            neighborhood[~np.isfinite(neighborhood)] = np.inf

            neighbor_zero_count = self._neighbor_zero_count[
                z_start:z_end, y_start:y_end, x_start:x_end
            ]
            background = (neighborhood == 0) & (neighbor_zero_count >= 9)
            neighborhood[background] = np.inf
            if not np.isfinite(neighborhood).any():
                return current_coord

            local_index = np.unravel_index(np.argmin(neighborhood), neighborhood.shape)
            candidate = (
                z_start + local_index[0],
                y_start + local_index[1],
                x_start + local_index[2],
            )
            candidate_value = self.volume[candidate]
            if not np.isfinite(candidate_value) or candidate_value >= self.volume[current_coord]:
                return current_coord
            current_coord = candidate

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import tifffile as tiff


class SliceViewer3D:
	"""Simple 3D TIFF slice viewer with mouse wheel navigation."""

	def __init__(self, volume: np.ndarray, cmap: str = "gray") -> None:
		if volume.ndim != 3:
			raise ValueError(f"Expected 3D volume (z, y, x), got shape={volume.shape}")

		self.volume = volume
		self.cmap = cmap
		self.num_slices = volume.shape[0]
		self.slice_index = self.num_slices // 2

		self.fig, self.ax = plt.subplots()
		self.image = self.ax.imshow(self.volume[self.slice_index], cmap=self.cmap)
		self.title = self.ax.set_title("")
		self.ax.set_xlabel("Mouse wheel: next/prev slice")

		self._update_view()

		# ImageJ-like interaction: mouse wheel to move through z-slices.
		self.fig.canvas.mpl_connect("scroll_event", self._on_scroll)
		self.fig.canvas.mpl_connect("key_press_event", self._on_key)

	def _on_scroll(self, event: object) -> None:
		if not hasattr(event, "button"):
			return

		if event.button == "up":
			self.slice_index = min(self.slice_index + 1, self.num_slices - 1)
		elif event.button == "down":
			self.slice_index = max(self.slice_index - 1, 0)

		self._update_view()

	def _on_key(self, event: object) -> None:
		if not hasattr(event, "key"):
			return

		if event.key in {"up", "right", "d"}:
			self.slice_index = min(self.slice_index + 1, self.num_slices - 1)
		elif event.key in {"down", "left", "a"}:
			self.slice_index = max(self.slice_index - 1, 0)
		elif event.key == "home":
			self.slice_index = 0
		elif event.key == "end":
			self.slice_index = self.num_slices - 1
		else:
			return

		self._update_view()

	def _update_view(self) -> None:
		self.image.set_data(self.volume[self.slice_index])
		self.title.set_text(f"Slice {self.slice_index + 1}/{self.num_slices}")
		self.image.set_clim(
			vmin=float(np.min(self.volume[self.slice_index])),
			vmax=float(np.max(self.volume[self.slice_index])),
		)
		self.fig.canvas.draw_idle()

	def show(self) -> None:
		plt.tight_layout()
		plt.show()


def load_3d_tiff(tiff_path: str | Path) -> np.ndarray:
	volume = tiff.imread(str(tiff_path))
	if volume.ndim != 3:
		raise ValueError(f"Input must be a 3D TIFF. Got shape={volume.shape}")
	return volume


def main() -> None:
	parser = argparse.ArgumentParser(
		description="3D TIFF slice viewer (mouse wheel to change slice)",
	)
	parser.add_argument("tiff_path", help="Path to 3D TIFF file")
	parser.add_argument("--cmap", default="gray", help="Matplotlib colormap")
	args = parser.parse_args()

	volume = load_3d_tiff(args.tiff_path)
	viewer = SliceViewer3D(volume=volume, cmap=args.cmap)
	viewer.show()


if __name__ == "__main__":
	main()

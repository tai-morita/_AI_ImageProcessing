from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import tifffile as tiff
from matplotlib.widgets import Slider


class SliceViewer3D:
	"""Simple 3D TIFF slice viewer with mouse wheel navigation."""

	def __init__(self, volume: np.ndarray, cmap: str = "gray", undo_radius_px: float = 5.0) -> None:
		if volume.ndim != 3:
			raise ValueError(f"Expected 3D volume (z, y, x), got shape={volume.shape}")

		self.volume = volume
		self.cmap = cmap
		self.undo_radius_px = float(undo_radius_px)
		self.num_slices = volume.shape[0]
		self.slice_index = self.num_slices // 2
		self.clicked_points: list[tuple[int, int, int]] = []

		self.fig, self.ax = plt.subplots()
		self.fig.subplots_adjust(bottom=0.18)
		self.image = self.ax.imshow(self.volume[self.slice_index], cmap=self.cmap)
		self.point_overlay = self.ax.scatter([], [], c="red", s=36, marker="o")
		self.title = self.ax.set_title("")
		self.ax.set_xlabel("Mouse wheel / slider: next-prev slice")
		self.ax.set_ylabel("Left click: save / Right click: undo near point")

		slider_ax = self.fig.add_axes([0.15, 0.06, 0.7, 0.04])
		self.slice_slider = Slider(
			ax=slider_ax,
			label="Slice",
			valmin=1,
			valmax=self.num_slices,
			valinit=self.slice_index + 1,
			valstep=1,
		)
		self.slice_slider.on_changed(self._on_slider_changed)

		self._update_view()

		# ImageJ-like interaction: mouse wheel to move through z-slices.
		self.fig.canvas.mpl_connect("scroll_event", self._on_scroll)
		self.fig.canvas.mpl_connect("key_press_event", self._on_key)
		self.fig.canvas.mpl_connect("button_press_event", self._on_click)

	def _set_slice_index(self, index: int) -> None:
		new_index = int(np.clip(index, 0, self.num_slices - 1))
		if new_index == self.slice_index:
			return
		self.slice_index = new_index
		if int(round(self.slice_slider.val)) != self.slice_index + 1:
			self.slice_slider.set_val(self.slice_index + 1)
		self._update_view()

	def _on_slider_changed(self, value: float) -> None:
		self._set_slice_index(int(round(float(value))) - 1)

	def _on_scroll(self, event: object) -> None:
		if not hasattr(event, "button"):
			return

		if event.button == "up":
			self._set_slice_index(self.slice_index + 1)
		elif event.button == "down":
			self._set_slice_index(self.slice_index - 1)

	def _on_key(self, event: object) -> None:
		if not hasattr(event, "key"):
			return

		if event.key in {"up", "right", "d"}:
			self._set_slice_index(self.slice_index + 1)
		elif event.key in {"down", "left", "a"}:
			self._set_slice_index(self.slice_index - 1)
		elif event.key == "home":
			self._set_slice_index(0)
		elif event.key == "end":
			self._set_slice_index(self.num_slices - 1)
		else:
			return

	def _on_click(self, event: object) -> None:
		if not hasattr(event, "inaxes") or event.inaxes != self.ax:
			return
		if not hasattr(event, "button"):
			return
		if not hasattr(event, "xdata") or not hasattr(event, "ydata"):
			return
		if event.xdata is None or event.ydata is None:
			return

		x = int(round(float(event.xdata)))
		y = int(round(float(event.ydata)))

		if not (0 <= y < self.volume.shape[1] and 0 <= x < self.volume.shape[2]):
			return

		if event.button == 1:
			saved = (self.slice_index + 1, y, x)
			self.clicked_points.append(saved)
			print(f"Saved click: slice={saved[0]}, y={saved[1]}, x={saved[2]}")
			self._update_view()
			return

		if event.button == 3:
			removed = self._remove_nearest_point_on_current_slice(y=y, x=x)
			if removed is None:
				print(
					f"No point found within radius={self.undo_radius_px:.1f}px "
					f"on slice={self.slice_index + 1}"
				)
			else:
				print(f"Removed point: slice={removed[0]}, y={removed[1]}, x={removed[2]}")
			self._update_view()

	def _remove_nearest_point_on_current_slice(self, y: int, x: int) -> tuple[int, int, int] | None:
		current_slice_1based = self.slice_index + 1
		candidates: list[tuple[float, int]] = []

		for idx, (saved_slice, saved_y, saved_x) in enumerate(self.clicked_points):
			if saved_slice != current_slice_1based:
				continue
			distance = float(np.hypot(saved_y - y, saved_x - x))
			if distance <= self.undo_radius_px:
				candidates.append((distance, idx))

		if not candidates:
			return None

		_, remove_idx = min(candidates, key=lambda item: item[0])
		return self.clicked_points.pop(remove_idx)

	def _update_view(self) -> None:
		self.image.set_data(self.volume[self.slice_index])
		current_slice_1based = self.slice_index + 1
		current_points = [
			(x, y)
			for saved_slice, y, x in self.clicked_points
			if saved_slice == current_slice_1based
		]
		if current_points:
			self.point_overlay.set_offsets(np.asarray(current_points, dtype=float))
		else:
			self.point_overlay.set_offsets(np.empty((0, 2), dtype=float))
		self.title.set_text(
			f"Slice {self.slice_index + 1}/{self.num_slices} | Saved points: {len(self.clicked_points)}"
		)
		self.image.set_clim(
			vmin=float(np.min(self.volume[self.slice_index])),
			vmax=float(np.max(self.volume[self.slice_index])),
		)
		self.fig.canvas.draw_idle()

	def show(self) -> None:
		plt.tight_layout()
		plt.show()

	def print_saved_points(self) -> None:
		print("\n=== Saved Click Points (slice, y, x) ===")
		if not self.clicked_points:
			print("No points saved.")
			return
		for index, (slice_idx, y, x) in enumerate(self.clicked_points, start=1):
			print(f"{index}: slice={slice_idx}, y={y}, x={x}")


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
	parser.add_argument(
		"--undo-radius-px",
		type=float,
		default=5.0,
		help="Right-click undo radius in pixels (default: 5.0)",
	)
	args = parser.parse_args()

	volume = load_3d_tiff(args.tiff_path)
	viewer = SliceViewer3D(volume=volume, cmap=args.cmap, undo_radius_px=args.undo_radius_px)
	viewer.show()
	viewer.print_saved_points()


if __name__ == "__main__":
	main()

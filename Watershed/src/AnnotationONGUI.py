from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import tifffile as tiff
from matplotlib.widgets import Button, Slider


class SliceViewer3D:
	"""4-panel 3D TIFF annotator with Axial/Coronal/Sagittal views."""

	def __init__(self, volume: np.ndarray, cmap: str = "gray", undo_radius_px: float = 5.0, max_labels: int = 30) -> None:
		if volume.ndim != 3:
			raise ValueError(f"Expected 3D volume (z, y, x), got shape={volume.shape}")

		self.volume = volume
		self.cmap = cmap
		self.undo_radius_px = float(undo_radius_px)
		self.n_z, self.n_y, self.n_x = volume.shape
		self.axial_z = self.n_z // 2
		self.coronal_y = self.n_y // 2
		self.sagittal_x = self.n_x // 2
		self.max_labels = int(max_labels)
		self.current_label = 1
		self.clicked_points: list[dict[str, int]] = []

		self.fig = plt.figure(figsize=(12, 8))
		self.fig.subplots_adjust(left=0.05, right=0.98, top=0.95, bottom=0.20, wspace=0.12, hspace=0.25)
		grid = self.fig.add_gridspec(2, 2)
		self.ax_axial = self.fig.add_subplot(grid[0, 0])
		self.ax_status = self.fig.add_subplot(grid[0, 1])
		self.ax_coronal = self.fig.add_subplot(grid[1, 0])
		self.ax_sagittal = self.fig.add_subplot(grid[1, 1])

		self.label_colors = plt.cm.get_cmap("tab20", self.max_labels)

		self.axial_image = self.ax_axial.imshow(self.volume[self.axial_z], cmap=self.cmap)
		self.coronal_image = self.ax_coronal.imshow(self.volume[:, self.coronal_y, :], cmap=self.cmap)
		self.sagittal_image = self.ax_sagittal.imshow(self.volume[:, :, self.sagittal_x], cmap=self.cmap)

		self.axial_overlay = self.ax_axial.scatter([], [], s=30, marker="o")
		self.coronal_overlay = self.ax_coronal.scatter([], [], s=30, marker="o")
		self.sagittal_overlay = self.ax_sagittal.scatter([], [], s=30, marker="o")

		self.axial_hline = self.ax_axial.axhline(self.coronal_y, color="lime", linestyle="--", linewidth=0.8, alpha=0.8)
		self.axial_vline = self.ax_axial.axvline(self.sagittal_x, color="cyan", linestyle="--", linewidth=0.8, alpha=0.8)
		self.coronal_hline = self.ax_coronal.axhline(self.axial_z, color="magenta", linestyle="--", linewidth=0.8, alpha=0.8)
		self.coronal_vline = self.ax_coronal.axvline(self.sagittal_x, color="cyan", linestyle="--", linewidth=0.8, alpha=0.8)
		self.sagittal_hline = self.ax_sagittal.axhline(self.axial_z, color="magenta", linestyle="--", linewidth=0.8, alpha=0.8)
		self.sagittal_vline = self.ax_sagittal.axvline(self.coronal_y, color="lime", linestyle="--", linewidth=0.8, alpha=0.8)

		self.ax_axial.set_xlabel("x")
		self.ax_axial.set_ylabel("y")
		self.ax_coronal.set_xlabel("x")
		self.ax_coronal.set_ylabel("z")
		self.ax_sagittal.set_xlabel("y")
		self.ax_sagittal.set_ylabel("z")

		axial_slider_ax = self.fig.add_axes([0.11, 0.51, 0.31, 0.03])
		coronal_slider_ax = self.fig.add_axes([0.11, 0.06, 0.31, 0.03])
		sagittal_slider_ax = self.fig.add_axes([0.57, 0.06, 0.31, 0.03])
		button_ax = self.fig.add_axes([0.57, 0.51, 0.31, 0.04])

		self.axial_slider = Slider(
			ax=axial_slider_ax,
			label="Axial(z)",
			valmin=1,
			valmax=self.n_z,
			valinit=self.axial_z + 1,
			valstep=1,
		)
		self.coronal_slider = Slider(
			ax=coronal_slider_ax,
			label="Coronal(y)",
			valmin=1,
			valmax=self.n_y,
			valinit=self.coronal_y + 1,
			valstep=1,
		)
		self.sagittal_slider = Slider(
			ax=sagittal_slider_ax,
			label="Sagittal(x)",
			valmin=1,
			valmax=self.n_x,
			valinit=self.sagittal_x + 1,
			valstep=1,
		)
		self.next_label_button = Button(button_ax, "Next Label")

		self.axial_slider.on_changed(self._on_axial_slider_changed)
		self.coronal_slider.on_changed(self._on_coronal_slider_changed)
		self.sagittal_slider.on_changed(self._on_sagittal_slider_changed)
		self.next_label_button.on_clicked(self._on_next_label)

		self._update_view()

		self.fig.canvas.mpl_connect("scroll_event", self._on_scroll)
		self.fig.canvas.mpl_connect("key_press_event", self._on_key)
		self.fig.canvas.mpl_connect("button_press_event", self._on_click)

	def _set_axial_index(self, index: int) -> None:
		new_index = int(np.clip(index, 0, self.n_z - 1))
		if new_index == self.axial_z:
			return
		self.axial_z = new_index
		if int(round(self.axial_slider.val)) != self.axial_z + 1:
			self.axial_slider.set_val(self.axial_z + 1)
		self._update_view()

	def _set_coronal_index(self, index: int) -> None:
		new_index = int(np.clip(index, 0, self.n_y - 1))
		if new_index == self.coronal_y:
			return
		self.coronal_y = new_index
		if int(round(self.coronal_slider.val)) != self.coronal_y + 1:
			self.coronal_slider.set_val(self.coronal_y + 1)
		self._update_view()

	def _set_sagittal_index(self, index: int) -> None:
		new_index = int(np.clip(index, 0, self.n_x - 1))
		if new_index == self.sagittal_x:
			return
		self.sagittal_x = new_index
		if int(round(self.sagittal_slider.val)) != self.sagittal_x + 1:
			self.sagittal_slider.set_val(self.sagittal_x + 1)
		self._update_view()

	def _on_axial_slider_changed(self, value: float) -> None:
		self._set_axial_index(int(round(float(value))) - 1)

	def _on_coronal_slider_changed(self, value: float) -> None:
		self._set_coronal_index(int(round(float(value))) - 1)

	def _on_sagittal_slider_changed(self, value: float) -> None:
		self._set_sagittal_index(int(round(float(value))) - 1)

	def _on_next_label(self, _event: object) -> None:
		if self.current_label >= self.max_labels:
			print(f"Reached maximum labels ({self.max_labels})")
			return
		self.current_label += 1
		print(f"Switched to label {self.current_label}")
		self._update_view()

	def _on_scroll(self, event: object) -> None:
		if not hasattr(event, "button") or not hasattr(event, "inaxes"):
			return
		if event.button == "up":
			delta = 1
		elif event.button == "down":
			delta = -1
		else:
			return

		if event.inaxes == self.ax_axial:
			self._set_axial_index(self.axial_z + delta)
		elif event.inaxes == self.ax_coronal:
			self._set_coronal_index(self.coronal_y + delta)
		elif event.inaxes == self.ax_sagittal:
			self._set_sagittal_index(self.sagittal_x + delta)

	def _on_key(self, event: object) -> None:
		if not hasattr(event, "key"):
			return

		if event.key in {"up", "right", "d"}:
			self._set_axial_index(self.axial_z + 1)
		elif event.key in {"down", "left", "a"}:
			self._set_axial_index(self.axial_z - 1)
		elif event.key == "home":
			self._set_axial_index(0)
		elif event.key == "end":
			self._set_axial_index(self.n_z - 1)
		elif event.key in {"n", "N"}:
			self._on_next_label(event)

	def _event_to_zyx(self, inaxes: Any, xdata: float, ydata: float) -> tuple[int, int, int] | None:
		if inaxes == self.ax_axial:
			x = int(round(xdata))
			y = int(round(ydata))
			if not (0 <= y < self.n_y and 0 <= x < self.n_x):
				return None
			return self.axial_z, y, x

		if inaxes == self.ax_coronal:
			x = int(round(xdata))
			z = int(round(ydata))
			if not (0 <= z < self.n_z and 0 <= x < self.n_x):
				return None
			return z, self.coronal_y, x

		if inaxes == self.ax_sagittal:
			y = int(round(xdata))
			z = int(round(ydata))
			if not (0 <= z < self.n_z and 0 <= y < self.n_y):
				return None
			return z, y, self.sagittal_x

		return None

	def _on_click(self, event: object) -> None:
		if not hasattr(event, "inaxes"):
			return
		if event.inaxes not in {self.ax_axial, self.ax_coronal, self.ax_sagittal}:
			return
		if not hasattr(event, "button"):
			return
		if not hasattr(event, "xdata") or not hasattr(event, "ydata"):
			return
		if event.xdata is None or event.ydata is None:
			return

		point_zyx = self._event_to_zyx(event.inaxes, float(event.xdata), float(event.ydata))
		if point_zyx is None:
			return
		z, y, x = point_zyx

		if event.button == 1:
			saved = {"label": self.current_label, "slice": z + 1, "y": y, "x": x}
			self.clicked_points.append(saved)
			print(f"Saved click: label={saved['label']}, slice={saved['slice']}, y={saved['y']}, x={saved['x']}")
			self._update_view()
			return

		if event.button == 3:
			removed = self._remove_nearest_point_on_current_plane(event.inaxes, z=z, y=y, x=x)
			if removed is None:
				print(f"No point found within radius={self.undo_radius_px:.1f}px")
			else:
				print(f"Removed point: label={removed['label']}, slice={removed['slice']}, y={removed['y']}, x={removed['x']}")
			self._update_view()

	def _remove_nearest_point_on_current_plane(self, inaxes: Any, z: int, y: int, x: int) -> dict[str, int] | None:
		candidates: list[tuple[float, int]] = []

		for idx, point in enumerate(self.clicked_points):
			sz = int(point["slice"]) - 1
			sy = int(point["y"])
			sx = int(point["x"])

			if inaxes == self.ax_axial and sz != self.axial_z:
				continue
			if inaxes == self.ax_coronal and sy != self.coronal_y:
				continue
			if inaxes == self.ax_sagittal and sx != self.sagittal_x:
				continue

			if inaxes == self.ax_axial:
				distance = float(np.hypot(sy - y, sx - x))
			elif inaxes == self.ax_coronal:
				distance = float(np.hypot(sz - z, sx - x))
			else:
				distance = float(np.hypot(sz - z, sy - y))

			if distance <= self.undo_radius_px:
				candidates.append((distance, idx))

		if not candidates:
			return None

		_, remove_idx = min(candidates, key=lambda item: item[0])
		return self.clicked_points.pop(remove_idx)

	def _label_to_color(self, label: int) -> tuple[float, float, float, float]:
		index = int(np.clip(label - 1, 0, self.max_labels - 1))
		return self.label_colors(index)

	def _set_overlay_points(self, ax_name: str) -> None:
		offsets: list[tuple[float, float]] = []
		colors: list[tuple[float, float, float, float]] = []

		for point in self.clicked_points:
			label = int(point["label"])
			sz = int(point["slice"]) - 1
			sy = int(point["y"])
			sx = int(point["x"])

			if ax_name == "axial" and sz == self.axial_z:
				offsets.append((sx, sy))
				colors.append(self._label_to_color(label))
			elif ax_name == "coronal" and sy == self.coronal_y:
				offsets.append((sx, sz))
				colors.append(self._label_to_color(label))
			elif ax_name == "sagittal" and sx == self.sagittal_x:
				offsets.append((sy, sz))
				colors.append(self._label_to_color(label))

		offsets_array = np.asarray(offsets, dtype=float) if offsets else np.empty((0, 2), dtype=float)

		if ax_name == "axial":
			self.axial_overlay.set_offsets(offsets_array)
			self.axial_overlay.set_color(colors if colors else "none")
		elif ax_name == "coronal":
			self.coronal_overlay.set_offsets(offsets_array)
			self.coronal_overlay.set_color(colors if colors else "none")
		else:
			self.sagittal_overlay.set_offsets(offsets_array)
			self.sagittal_overlay.set_color(colors if colors else "none")

	def _build_status_text(self) -> str:
		count_by_label: dict[int, int] = {}
		for point in self.clicked_points:
			label = int(point["label"])
			count_by_label[label] = count_by_label.get(label, 0) + 1

		lines = [
			f"Current label: {self.current_label}/{self.max_labels}",
			f"Total saved points: {len(self.clicked_points)}",
			"",
			"Controls:",
			"- Left click: add point",
			"- Right click: remove nearby point",
			"- Next Label button: move to next label",
			"",
			"Label summary:",
		]

		if not count_by_label:
			lines.append("  No points saved yet")
		else:
			for label in sorted(count_by_label.keys()):
				marker = "*" if label == self.current_label else " "
				lines.append(f"{marker} Label {label:02d}: {count_by_label[label]} points")

		return "\n".join(lines)

	def _update_view(self) -> None:
		self.axial_image.set_data(self.volume[self.axial_z])
		self.coronal_image.set_data(self.volume[:, self.coronal_y, :])
		self.sagittal_image.set_data(self.volume[:, :, self.sagittal_x])

		self.axial_image.set_clim(vmin=float(np.min(self.volume[self.axial_z])), vmax=float(np.max(self.volume[self.axial_z])))
		self.coronal_image.set_clim(vmin=float(np.min(self.volume[:, self.coronal_y, :])), vmax=float(np.max(self.volume[:, self.coronal_y, :])))
		self.sagittal_image.set_clim(vmin=float(np.min(self.volume[:, :, self.sagittal_x])), vmax=float(np.max(self.volume[:, :, self.sagittal_x])))

		self.ax_axial.set_title(f"Axial (z={self.axial_z + 1}/{self.n_z})")
		self.ax_coronal.set_title(f"Coronal (y={self.coronal_y + 1}/{self.n_y})")
		self.ax_sagittal.set_title(f"Sagittal (x={self.sagittal_x + 1}/{self.n_x})")

		self.axial_hline.set_ydata([self.coronal_y, self.coronal_y])
		self.axial_vline.set_xdata([self.sagittal_x, self.sagittal_x])
		self.coronal_hline.set_ydata([self.axial_z, self.axial_z])
		self.coronal_vline.set_xdata([self.sagittal_x, self.sagittal_x])
		self.sagittal_hline.set_ydata([self.axial_z, self.axial_z])
		self.sagittal_vline.set_xdata([self.coronal_y, self.coronal_y])

		self._set_overlay_points("axial")
		self._set_overlay_points("coronal")
		self._set_overlay_points("sagittal")

		self.ax_status.clear()
		self.ax_status.axis("off")
		self.ax_status.set_title("Labeling Status", fontsize=12)
		self.ax_status.text(0.02, 0.98, self._build_status_text(), va="top", ha="left", family="monospace", fontsize=10)

		self.fig.canvas.draw_idle()

	def show(self) -> None:
		plt.show()

	def print_saved_points(self) -> None:
		print("\n=== Saved Click Points (label, slice, y, x) ===")
		if not self.clicked_points:
			print("No points saved.")
			return
		for index, point in enumerate(self.clicked_points, start=1):
			print(
				f"{index}: label={point['label']}, slice={point['slice']}, y={point['y']}, x={point['x']}"
			)

	def export_grouped_points(self) -> list[dict[str, object]]:
		grouped: dict[int, list[tuple[int, int, int]]] = {}
		for point in self.clicked_points:
			label = int(point["label"])
			slice_1based = int(point["slice"])
			y = int(point["y"])
			x = int(point["x"])
			grouped.setdefault(label, []).append((slice_1based, y, x))

		result: list[dict[str, object]] = []
		for label in sorted(grouped.keys()):
			result.append({"label": label, "points": grouped[label]})
		return result


def load_3d_tiff(tiff_path: str | Path) -> np.ndarray:
	volume = tiff.imread(str(tiff_path))
	if volume.ndim != 3:
		raise ValueError(f"Input must be a 3D TIFF. Got shape={volume.shape}")
	return volume


def main(tiff_path: str) -> list[dict[str, object]]:
	parser = argparse.ArgumentParser(
		description="4-panel 3D TIFF annotator",
	)
	parser.add_argument("tiff_path", help="Path to 3D TIFF file")
	parser.add_argument("--cmap", default="gray", help="Matplotlib colormap")
	parser.add_argument(
		"--undo-radius-px",
		type=float,
		default=5.0,
		help="Right-click undo radius in pixels (default: 5.0)",
	)

	volume = load_3d_tiff(tiff_path)
	viewer = SliceViewer3D(volume=volume, cmap="gray", undo_radius_px=5.0, max_labels=30)
	viewer.show()
	viewer.print_saved_points()
	return viewer.export_grouped_points()


if __name__ == "__main__":
	main()

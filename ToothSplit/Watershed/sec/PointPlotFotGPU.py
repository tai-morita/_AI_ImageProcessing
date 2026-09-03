from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import plotly.graph_objects as go


def load_npy_volume(path: Path) -> np.ndarray:
	"""Load a 3D volume from .npy."""
	if path.suffix.lower() != ".npy":
		# .tif なら .npy に変換する
		if path.suffix.lower() == ".tif":
			from tifffile import imread

			volume = imread(path)
			if volume.ndim != 3:
				raise ValueError(f"Expected 3D array, but got shape={volume.shape}")
			return volume
		raise ValueError("input must be a .npy file")
	volume = np.load(path)
	if volume.ndim != 3:
		raise ValueError(f"Expected 3D array, but got shape={volume.shape}")
	return volume


def volume_to_point_cloud(
	volume: np.ndarray,
	threshold: float,
	max_points: int,
) -> tuple[np.ndarray, np.ndarray]:
	"""Convert voxels above threshold to point cloud and return points and values."""
	mask = volume > threshold
	zyx = np.argwhere(mask)
	if zyx.size == 0:
		raise ValueError("No points found. Try lowering --threshold.")

	values = volume[mask]
	if len(zyx) > max_points:
		idx = np.random.choice(len(zyx), max_points, replace=False)
		zyx = zyx[idx]
		values = values[idx]

	# Convert (z, y, x) -> (x, y, z)
	xyz = zyx[:, [2, 1, 0]].astype(np.float32)
	return xyz, values.astype(np.float32)


def show_gpu_point_cloud(
	xyz: np.ndarray,
	values: np.ndarray,
	point_size: float,
	opacity: float,
	output_html: Path | None,
) -> None:
	"""Render point cloud with Plotly WebGL (GPU)."""
	fig = go.Figure(
		data=[
			go.Scatter3d(
				x=xyz[:, 0],
				y=xyz[:, 1],
				z=xyz[:, 2],
				mode="markers",
				marker={
					"size": point_size,
					"opacity": opacity,
					"color": values,
					"colorscale": "Viridis",
					"colorbar": {"title": "voxel value"},
				},
			)
		]
	)

	fig.update_layout(
		title="3D Point Cloud (WebGL GPU Rendering)",
		scene={
			"xaxis_title": "X",
			"yaxis_title": "Y",
			"zaxis_title": "Z",
			"aspectmode": "data",
		},
		margin={"l": 0, "r": 0, "t": 40, "b": 0},
	)

	if output_html is not None:
		fig.write_html(str(output_html), include_plotlyjs="cdn")
		print(f"Saved: {output_html}")

	fig.show()


def main() -> None:
	parser = argparse.ArgumentParser(
		description="Convert 3D .npy volume to point cloud and visualize with GPU (WebGL)"
	)
	parser.add_argument("input", type=Path, help="Path to input .npy file")
	parser.add_argument(
		"--threshold",
		type=float,
		default=0.0,
		help="Use voxels where value > threshold (default: 0.0)",
	)
	parser.add_argument(
		"--max-points",
		type=int,
		default=400000,
		help="Maximum points to render for speed (default: 400000)",
	)
	parser.add_argument(
		"--point-size",
		type=float,
		default=1.5,
		help="Point size for rendering (default: 1.5)",
	)
	parser.add_argument(
		"--opacity",
		type=float,
		default=0.6,
		help="Point opacity in [0, 1] (default: 0.6)",
	)
	parser.add_argument(
		"--output-html",
		type=Path,
		default=None,
		help="Optional output HTML path for sharing/saving",
	)
	args = parser.parse_args()

	if not (0.0 <= args.opacity <= 1.0):
		raise ValueError("--opacity must be between 0 and 1")

	volume = load_npy_volume(args.input)
	xyz, values = volume_to_point_cloud(volume, args.threshold, args.max_points)

	print(f"Input shape: {volume.shape}")
	print(f"Rendered points: {len(xyz):,}")
	print("Renderer: Plotly WebGL (GPU)")

	show_gpu_point_cloud(
		xyz=xyz,
		values=values,
		point_size=args.point_size,
		opacity=args.opacity,
		output_html=args.output_html,
	)


if __name__ == "__main__":
	main()

"""
usage:
python ./study/test/20260812_3dplot.py 
--input ./study/Watershed/t-oe/20260716_NR_Label/CTHRs_50_label_binary_root.tif 
--threshold 0.5 
--output-html ./study/Watershed/t-oe/20260716_NR_Label/3DPlot_binary.html
"""
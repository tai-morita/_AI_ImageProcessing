import numpy as np
import tifffile as tiff
import matplotlib.pyplot as plt

try:
	from .EditSeed import label_previous_slice_on_large_area_diff, profile_component_area_by_slice
	from .AnnotationONGUI import main as annotation_main
except ImportError:
	from EditSeed import label_previous_slice_on_large_area_diff, profile_component_area_by_slice
	from AnnotationONGUI import main as annotation_main


def split_clicked_coordinates(
	clicked_coordinates: list[tuple[int, int, int]] | list[dict[str, object]],
) -> tuple[list[int], list[tuple[int, int]], list[int]]:
	"""GUI出力を slices / coordinates / labels に正規化する。"""
	slices: list[int] = []
	coords: list[tuple[int, int]] = []
	labels: list[int] = []

	if len(clicked_coordinates) == 0:
		return slices, coords, labels

	first = clicked_coordinates[0]
	if isinstance(first, dict):
		for entry in clicked_coordinates:
			if not isinstance(entry, dict):
				raise ValueError("clicked_coordinates entries must all be dict when grouped format is used")
			if "label" not in entry or "points" not in entry:
				raise ValueError("grouped format requires 'label' and 'points'")
			label = int(entry["label"])
			points = entry["points"]
			if not isinstance(points, list):
				raise ValueError("entry['points'] must be a list")
			for point in points:
				if not (isinstance(point, tuple) or isinstance(point, list)) or len(point) != 3:
					raise ValueError("point must be (slice, y, x)")
				s, y, x = point
				slices.append(int(s))
				coords.append((int(y), int(x)))
				labels.append(label)
		return slices, coords, labels

	for item in clicked_coordinates:
		if not (isinstance(item, tuple) or isinstance(item, list)) or len(item) != 3:
			raise ValueError("legacy format expects (slice, y, x)")
		s, y, x = item
		slices.append(int(s))
		coords.append((int(y), int(x)))

	labels = list(range(1, len(coords) + 1))
	return slices, coords, labels

def get_slice_range(slices: list[int], margin: int = 5) -> list[tuple[int, int]]:
	"""クリックされたスライスに対して、前後 margin スライスを含む範囲を返す。"""
	if len(slices) == 0:
		return []
	return [(max(0, s - margin), s + margin) for s in slices]

def plot_profiles_per_target(
	profile: list[dict[str, object]],
	coordinates: list[tuple[int, int]],
	labels: list[int],
	x_values: list[float] | list[list[float]] | None = None,
	x_label: str = "slice",
	slice_ranges: list[tuple[int, int]] | None = None,
) -> None:
	if len(profile) == 0:
		print("profile is empty")
		return

	# 複数ターゲット時
	if "target_index" in profile[0]:
		n_targets = len(coordinates)
		if slice_ranges is not None and len(slice_ranges) != n_targets:
			raise ValueError("When slice_ranges is set, its length must match number of targets.")

		if x_values is None:
			x_values_per_target: list[list[float] | None] = [None] * n_targets
		elif len(x_values) > 0 and isinstance(x_values[0], list):
			x_values_per_target = x_values  # type: ignore[assignment]
			if len(x_values_per_target) != n_targets:
				raise ValueError("When x_values is list[list], its length must match number of targets.")
		else:
			x_values_per_target = [x_values] * n_targets  # type: ignore[list-item]

		for target_index, (coordinate, label_no) in enumerate(zip(coordinates, labels)):
			target_rows = [row for row in profile if int(row["target_index"]) == target_index]
			if slice_ranges is not None:
				start_slice, end_slice = slice_ranges[target_index]
				target_rows = [
					row for row in target_rows
					if start_slice <= int(row["slice"]) <= end_slice
				]
				if len(target_rows) == 0:
					print(f"Target {target_index}: 指定範囲 {start_slice}-{end_slice} にデータがありません")
					continue

			slices = [int(row["slice"]) for row in target_rows]
			areas = [int(row["area"]) for row in target_rows]
			x_axis = x_values_per_target[target_index]
			if x_axis is None:
				x_axis = [float(v) for v in slices]
			if len(x_axis) != len(areas):
				raise ValueError(
					f"x_values length mismatch for target {target_index}: expected {len(areas)}, got {len(x_axis)}"
				)

			d_areas = np.diff(areas)
			d_x_axis = x_axis[1:]

			fig, ax1 = plt.subplots(figsize=(8, 4))
			line1 = ax1.plot(x_axis, areas, marker="o", linewidth=0.8, color="tab:blue", label="area")
			"""
            for x, y in zip(slices, areas):
				ax1.annotate(str(y), (x, y), textcoords="offset points", xytext=(0, 4), ha="center", fontsize=8)
            """
			ax1.set_title(f"Target {target_index}: coordinate={coordinate}, label={label_no}")
			ax1.set_xlabel(x_label)
			ax1.set_ylabel("component area", color="tab:blue")
			ax1.tick_params(axis="y", labelcolor="tab:blue")
			ax1.grid(True, alpha=0.3)

			ax2 = ax1.twinx()
			line2 = ax2.plot(
				d_x_axis,
				d_areas,
				marker="x",
				linestyle="--",
				linewidth=0.8,
				color="tab:red",
				label="d(area)",
			)
			"""
            for x, y in zip(d_slices, d_areas):
				ax2.annotate(str(int(y)), (x, y), textcoords="offset points", xytext=(0, -10), ha="center", fontsize=8)
			"""
			ax2.set_ylabel("d(area)", color="tab:red")
			ax2.tick_params(axis="y", labelcolor="tab:red")

			lines = line1 + line2
			labels_legend = [ln.get_label() for ln in lines]
			ax1.legend(lines, labels_legend, loc="upper right")

			fig.tight_layout()
			# plt.show()
		return

	# 単一ターゲット時
	target_rows = profile
	if slice_ranges is not None:
		start_slice, end_slice = slice_ranges[0]
		target_rows = [
			row for row in target_rows
			if start_slice <= int(row["slice"]) <= end_slice
		]
		if len(target_rows) == 0:
			print(f"Target 0: 指定範囲 {start_slice}-{end_slice} にデータがありません")
			return

	slices = [int(row["slice"]) for row in target_rows]
	areas = [int(row["area"]) for row in target_rows]
	if x_values is None:
		x_axis = [float(v) for v in slices]
	elif len(x_values) > 0 and isinstance(x_values[0], list):
		x_axis = x_values[0]  # type: ignore[index]
	else:
		x_axis = x_values  # type: ignore[assignment]

	if len(x_axis) != len(areas):
		raise ValueError(f"x_values length mismatch: expected {len(areas)}, got {len(x_axis)}")

	d_areas = np.diff(areas)
	d_x_axis = x_axis[1:]

	fig, ax1 = plt.subplots(figsize=(8, 4))
	line1 = ax1.plot(x_axis, areas, marker="o", linewidth=0.8, color="tab:blue", label="area")
	"""
	for x, y in zip(slices, areas):
		ax1.annotate(str(y), (x, y), textcoords="offset points", xytext=(0, 4), ha="center", fontsize=8)
    """
	ax1.set_title(f"Target 0: coordinate={coordinates[0]}, label={labels[0]}")
	ax1.set_xlabel(x_label)
	ax1.set_ylabel("component area", color="tab:blue")
	ax1.tick_params(axis="y", labelcolor="tab:blue")
	ax1.grid(True, alpha=0.3)

	ax2 = ax1.twinx()
	line2 = ax2.plot(d_x_axis, d_areas, marker="x", linestyle="--", linewidth=0.8, color="tab:red", label="d(area)")
	"""
	for x, y in zip(d_slices, d_areas):
		ax2.annotate(str(int(y)), (x, y), textcoords="offset points", xytext=(0, -10), ha="center", fontsize=8)
	"""
	ax2.set_ylabel("d(area)", color="tab:red")
	ax2.tick_params(axis="y", labelcolor="tab:red")

	lines = line1 + line2
	labels_legend = [ln.get_label() for ln in lines]
	ax1.legend(lines, labels_legend, loc="upper right")

	fig.tight_layout()
	plt.show() # プロファイル


def test_profile_component_area_by_slice(tmp_path):
	volume = np.zeros((3, 8, 8), dtype=np.uint8)
	volume[0, 3:5, 3:5] = 1
	volume[1, 2:6, 2:6] = 1
	volume[2, 1:7, 1:7] = 1

	input_path = tmp_path / "input.tif"
	tiff.imwrite(input_path, volume)

	profile = profile_component_area_by_slice(str(input_path), coordinate_yx=(3, 3), connectivity=1)

	assert [row["area"] for row in profile] == [4, 16, 36]


def test_label_previous_slice_on_large_area_diff(tmp_path):
	volume = np.zeros((4, 12, 12), dtype=np.uint8)
	volume[0, 5:7, 5:7] = 1
	volume[1, 5:7, 5:7] = 1
	volume[2, 1:11, 1:11] = 1
	volume[3, 1:11, 1:11] = 1

	input_path = tmp_path / "input.tif"
	output_path = tmp_path / "output.tif"
	tiff.imwrite(input_path, volume)

	profile, labeled = label_previous_slice_on_large_area_diff(
		str(input_path),
		str(output_path),
		coordinate_yx=(5, 5),
		labeling_no=7,
		diff_threshold=50,
		connectivity=1,
	)

	assert [row["area"] for row in profile] == [4, 4, 100, 100]
	assert np.all(labeled[1, 5:7, 5:7] == 7)
	assert np.count_nonzero(labeled[1] == 7) == 4
	assert np.count_nonzero(labeled[0]) == 0
	assert np.count_nonzero(labeled[2]) == 0
	assert output_path.exists()


def test_label_previous_slice_on_large_area_diff_multiple_targets(tmp_path):
	volume = np.zeros((4, 20, 20), dtype=np.uint8)
	volume[0, 5:7, 5:7] = 1
	volume[1, 5:7, 5:7] = 1
	volume[2, 1:11, 1:11] = 1
	volume[3, 1:11, 1:11] = 1

	volume[0, 15:17, 15:17] = 1
	volume[1, 15:17, 15:17] = 1
	volume[2, 10:19, 10:19] = 1
	volume[3, 10:19, 10:19] = 1

	input_path = tmp_path / "input_multi.tif"
	output_path = tmp_path / "output_multi.tif"
	tiff.imwrite(input_path, volume)

	profile, labeled = label_previous_slice_on_large_area_diff(
		str(input_path),
		str(output_path),
		coordinate_yx=[(5, 5), (15, 15)],
		labeling_no=[7, 9],
		diff_threshold=50,
		connectivity=1,
	)

	assert len(profile) == 8
	assert profile[0]["target_index"] == 0
	assert profile[4]["target_index"] == 1
	assert np.all(labeled[1, 5:7, 5:7] == 7)
	assert np.all(labeled[1, 15:17, 15:17] == 9)
	assert output_path.exists()


def test_label_max_area_when_threshold_not_reached(tmp_path):
	volume = np.zeros((5, 20, 20), dtype=np.uint8)
	# coordinate=(10,10) を含む成分面積: [4, 6, 8, 10, 9]
	volume[0, 9:11, 9:11] = 1
	volume[1, 9:11, 8:11] = 1
	volume[2, 8:11, 8:11] = 1
	volume[3, 8:11, 7:11] = 1
	volume[4, 8:11, 8:11] = 1

	input_path = tmp_path / "input_no_threshold.tif"
	output_path = tmp_path / "output_no_threshold.tif"
	tiff.imwrite(input_path, volume)

	profile, labeled = label_previous_slice_on_large_area_diff(
		str(input_path),
		str(output_path),
		coordinate_yx=(10, 10),
		labeling_no=5,
		diff_threshold=100,  # 閾値に到達しない値
		connectivity=1,
		slice_ranges=(0, 4),
	)

	assert [row["area"] for row in profile] == [4, 6, 9, 12, 9]
	# 最大面積はスライス3なので、そこがラベリングされる
	assert np.count_nonzero(labeled[3] == 5) == 12
	assert np.count_nonzero(labeled[0]) == 0
	assert np.count_nonzero(labeled[1]) == 0
	assert np.count_nonzero(labeled[2]) == 0
	assert np.count_nonzero(labeled[4]) == 0
	assert output_path.exists()

def test_edit_seed_main(input_path: str, output_path: str):
	# GUI 操作から Seed 作成まで一連の処理をする
	mark_coordinates = False
	coordinate_mark_value = 255

	# GUI 操作
	clicked_coordinates = annotation_main(input_path)

	# クリックした Seed の座標を編集
	slices, coordinates, labels = split_clicked_coordinates(clicked_coordinates)
	if len(coordinates) == 0:
		print("GUIで座標が選択されなかったため処理を終了します")
		return
	slice_ranges = get_slice_range(slices, margin=30)

	# Seed 編集とプロファイル作成
	profile, labeled = label_previous_slice_on_large_area_diff(
		volume_label_path=input_path,
		output_path=output_path,
		coordinate_yx=coordinates,
		labeling_no=labels,
		diff_threshold=1000,
		connectivity=1,
		slice_ranges=slice_ranges,
		mark_coordinates=mark_coordinates,
		coordinate_mark_value=coordinate_mark_value,
	)
	x_values = None  # 例: [0.0, 0.5, 1.0, ...] または [[target0用...], [target1用...]]
	plot_profiles_per_target(
		profile,
		coordinates,
		labels,
		x_values=x_values,
		x_label="slice",
		slice_ranges=slice_ranges,
	)

if __name__ == "__main__":
	volume_label_path = r"D:\_study\ImageProcessing\study\Watershed\Data\Input\20260402_filled255_No1\label_map_1_filled.tif"
	output_path       = r"D:\_study\ImageProcessing\study\Watershed\Data\Input\20260402_filled255_No1\output_labeled.tif"
	coordinates = [(289, 141), (66, 273), (65, 232), (91, 198), (114, 163), (148, 141), (194, 122), (250, 123),
				(34, 225), (53, 170), (164, 94), (257, 107), (298, 142),
				(120, 145)]
	labels = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]
	mark_coordinates = False
	coordinate_mark_value = 255
	slice_ranges = [(0, 50), (0, 50), (0, 50), (0, 50), (0, 50), (0, 50), (0, 50), (0, 50),
				 (70, 200), (70, 200), (70, 200), (70, 200), (70, 200),
				  (160, 200)]  # 例: [(start0, end0), (start1, end1), ...]
	profile, labeled = label_previous_slice_on_large_area_diff(
		volume_label_path=volume_label_path,
		output_path=output_path,
		coordinate_yx=coordinates,
		labeling_no=labels,
		diff_threshold=1000,
		connectivity=1,
		slice_ranges=slice_ranges,
		mark_coordinates=mark_coordinates,
		coordinate_mark_value=coordinate_mark_value,
	)
	x_values = None  # 例: [0.0, 0.5, 1.0, ...] または [[target0用...], [target1用...]]
	plot_profiles_per_target(
		profile,
		coordinates,
		labels,
		x_values=x_values,
		x_label="slice",
		slice_ranges=slice_ranges,
	)

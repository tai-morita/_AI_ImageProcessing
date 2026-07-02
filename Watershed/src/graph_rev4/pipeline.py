from dataclasses import dataclass
import os

import numpy as np
import tifffile as tiff
from scipy import ndimage as ndi

try:
    from .graph_build import create_graph, map_reverse_graph_to_original, merge_bidirectional_graph
    from .graph_prune import remove_edge
    from .graph_quality import (
        add_largest_area_seed_in_components,
        detect_non_bifurcating_components,
    )
    from .io_ops import (
        apply_execution_slice_mask,
        create_execution_slice_mask,
        ensure_dir,
        load_volume,
        save_json,
    )
    from .occlusal_align import align_occlusal_plane
    from .pca_split import (
        make_anterior_molar_split_overlay_volume,
        split_volume_anterior_molar_by_pca,
        summarize_region_boundary_slices_from_split_labels,
    )
    from .seed_ops import create_seed_for_watershed, merge_seed_non_overlap, relabel_seed_volume
    from .visualize import plot_graph_for_slice_range
    from .watershed_runner import watershed_3d_volume
except ImportError:
    from graph_build import create_graph, map_reverse_graph_to_original, merge_bidirectional_graph
    from graph_prune import remove_edge
    from graph_quality import (
        add_largest_area_seed_in_components,
        detect_non_bifurcating_components,
    )
    from io_ops import (
        apply_execution_slice_mask,
        create_execution_slice_mask,
        ensure_dir,
        load_volume,
        save_json,
    )
    from occlusal_align import align_occlusal_plane
    from pca_split import (
        make_anterior_molar_split_overlay_volume,
        split_volume_anterior_molar_by_pca,
        summarize_region_boundary_slices_from_split_labels,
    )
    from seed_ops import create_seed_for_watershed, merge_seed_non_overlap, relabel_seed_volume
    from visualize import plot_graph_for_slice_range
    from watershed_runner import watershed_3d_volume


@dataclass
class GraphRev4Config:
    """rev4パイプラインの実行設定を保持するデータクラス。

    Args:
        input_path: 入力ボリュームのパス。
        output_dir: 出力ディレクトリ。
        threshold: 共通面積閾値。
        anterior_threshold: 前歯領域用閾値。
        molar_threshold: 臼歯領域用閾値。
        labels_connectivity: 2D連結成分の近傍設定。
        split_percentile: PCA分割境界の百分位。
        ambiguous_band_px: 分割曖昧帯の幅。
        inverse_volume: 逆方向処理を有効化するフラグ。
        direction_mode: forward_only / reverse_only / bidirectional。
        execution_slice_range: 実行対象の1-basedスライス範囲。
        plot_execution_slice_graphs: 範囲グラフを保存するかどうか。
        show_graph_plots: グラフを画面表示するかどうか。
        run_watershed: watershed実行の有無。
        enable_occlusal_alignment: 咬合平面一致の有無。
        occlusal_spacing_zyx: 咬合平面一致で用いるボクセル間隔。
        occlusal_percentile: 咬合面候補抽出百分位。
        occlusal_side: 咬合面候補の方向設定。
        occlusal_pad: 咬合平面一致時のパディング量。
        occlusal_inplane_k45: 面内回転の45度単位指定。

    Returns:
        なし。
    """
    input_path: str
    output_dir: str
    threshold: int = 1000
    anterior_threshold: int | None = None
    molar_threshold: int | None = None
    labels_connectivity: int = 2
    split_percentile: float = 50.0
    ambiguous_band_px: float = 0.0
    inverse_volume: bool = True
    direction_mode: str | None = None
    execution_slice_range: tuple | None = None
    plot_execution_slice_graphs: bool = False
    show_graph_plots: bool = False
    run_watershed: bool = True

    # Occlusal alignment
    enable_occlusal_alignment: bool = True
    occlusal_spacing_zyx: tuple[float, float, float] = (1.0, 1.0, 1.0)
    occlusal_percentile: float = 85.0
    occlusal_side: str = "auto"
    occlusal_pad: int = 0
    occlusal_inplane_k45: int = 0


@dataclass
class JawRangeSpec:
    """顎領域のスライス範囲設定。"""

    name: str
    start_slice: int
    end_slice: int


def graph_seed_volume_bidirectional(
    volume: np.ndarray,
    threshold: int = 1000,
    labels_connectivity: int = 2,
    inverse_volume: bool = True,
    direction_mode: str | None = None,
):
    """方向モードに応じてグラフseedを生成する。

    Args:
        volume: 入力3次元ボリューム。
        threshold: 面積閾値。
        labels_connectivity: 2D連結成分の近傍設定。
        inverse_volume: direction_mode未指定時の既定動作に使うフラグ。
        direction_mode: forward_only / reverse_only / bidirectional。

    Returns:
        統合グラフ、seedボリューム、処理統計情報辞書。
    """
    if direction_mode is None:
        direction_mode = "bidirectional" if inverse_volume else "forward_only"
    direction_mode = str(direction_mode).lower()
    valid_modes = {"forward_only", "reverse_only", "bidirectional"}
    if direction_mode not in valid_modes:
        raise ValueError(f"direction_mode must be one of {sorted(valid_modes)}: {direction_mode}")

    use_forward = direction_mode in {"forward_only", "bidirectional"}
    use_reverse = direction_mode in {"reverse_only", "bidirectional"}

    graph_forward = None
    labeled_volume_forward = None
    if use_forward:
        graph_forward, labeled_volume_forward = create_graph(
            volume,
            threshold=threshold,
            labels_connectivity=labels_connectivity,
        )
        remove_edge(graph_forward)

    graph_reverse = None
    labeled_volume_reverse = None
    if use_reverse:
        volume_reverse = volume[::-1]
        graph_reverse, labeled_volume_reverse = create_graph(
            volume_reverse,
            threshold=threshold,
            labels_connectivity=labels_connectivity,
        )
        remove_edge(graph_reverse)

    if use_forward and use_reverse:
        merged_graph = merge_bidirectional_graph(graph_forward, graph_reverse, volume.shape[0])
    elif use_forward:
        merged_graph = graph_forward
    elif use_reverse:
        merged_graph = map_reverse_graph_to_original(graph_reverse, volume.shape[0])
    else:
        raise ValueError("No valid direction mode selected.")

    non_bifurcating = detect_non_bifurcating_components(merged_graph)
    added_seed_nodes = add_largest_area_seed_in_components(merged_graph, non_bifurcating)

    for node in added_seed_nodes:
        if use_forward and graph_forward is not None and node in graph_forward:
            graph_forward.nodes[node]["seed"] = True
        if use_reverse and graph_reverse is not None:
            reverse_node = (volume.shape[0] - node[0] + 1, node[1])
            if reverse_node in graph_reverse:
                graph_reverse.nodes[reverse_node]["seed"] = True

    seed_forward = (
        create_seed_for_watershed(graph_forward, labeled_volume_forward)
        if use_forward and graph_forward is not None and labeled_volume_forward is not None
        else None
    )
    seed_volume = seed_forward if seed_forward is not None else np.zeros_like(volume, dtype=np.uint16)

    if use_reverse and graph_reverse is not None and labeled_volume_reverse is not None:
        seed_reverse = create_seed_for_watershed(graph_reverse, labeled_volume_reverse)
        seed_reverse = seed_reverse[::-1]
        seed_volume = (
            np.where(seed_forward != 0, seed_forward, seed_reverse)
            if seed_forward is not None
            else seed_reverse
        )

    return merged_graph, seed_volume, {
        "threshold": int(threshold),
        "inverse_volume": bool(inverse_volume),
        "direction_mode": direction_mode,
        "node_count": int(merged_graph.number_of_nodes()),
        "edge_count": int(merged_graph.number_of_edges()),
        "seed_voxel_count": int(np.count_nonzero(seed_volume)),
        "added_seed_node_count": int(len(added_seed_nodes)),
        "non_bifurcating_component_count": int(len(non_bifurcating)),
    }


def _trim_by_top_far_region(mask_2d: np.ndarray, keep_ratio: float = 0.5) -> np.ndarray:
    """2D二値マスクのうち距離が大きい上位割合を残す。"""
    if mask_2d.ndim != 2:
        raise ValueError("mask_2d must be 2D.")
    if not (0 < keep_ratio <= 1):
        raise ValueError("keep_ratio must be in (0, 1].")

    mask = mask_2d.astype(bool)
    if not np.any(mask):
        return np.zeros_like(mask, dtype=bool)

    dist = ndi.distance_transform_edt(mask)
    vals = dist[mask]
    q = 100.0 * (1.0 - keep_ratio)
    th = np.percentile(vals, q)
    return mask & (dist >= th)


def _trim_volume_by_top_far_region(volume_3d: np.ndarray, keep_ratio: float = 0.5) -> np.ndarray:
    """3Dボリュームをスライスごとに2D距離ベースでトリミングする。"""
    if volume_3d.ndim != 3:
        raise ValueError("volume_3d must be 3D.")

    trimmed = np.zeros(volume_3d.shape, dtype=bool)
    for z in range(volume_3d.shape[0]):
        trimmed[z] = _trim_by_top_far_region(volume_3d[z], keep_ratio=keep_ratio)
    return trimmed


def _relabel_components_per_slice(mask_3d: np.ndarray) -> np.ndarray:
    """3D二値マスクをスライスごとに2D連結成分ラベリングし、全体連番化する。"""
    if mask_3d.ndim != 3:
        raise ValueError("mask_3d must be 3D.")

    relabeled = np.zeros(mask_3d.shape, dtype=np.uint16)
    next_label = 1
    for z in range(mask_3d.shape[0]):
        labeled_slice, num = ndi.label(mask_3d[z].astype(bool), structure=np.ones((3, 3), dtype=np.uint8))
        if num == 0:
            continue
        labeled_slice = labeled_slice.astype(np.uint32)
        nz = labeled_slice > 0
        labeled_slice[nz] += next_label - 1
        relabeled[z] = labeled_slice.astype(np.uint16)
        next_label += int(num)
    return relabeled


def _parse_slice_range(text: str) -> tuple[int, int]:
    """`start:end` 形式の0-basedスライス範囲を解析する(endは排他的)。"""
    if ":" not in text:
        raise ValueError(f"Invalid slice range: {text}")
    start_s, end_s = text.split(":", 1)
    start = int(start_s)
    end = int(end_s)
    if start < 0 or end <= start:
        raise ValueError(f"Invalid slice range: {text}")
    return start, end


def _build_seed_for_jaw_region(
    volume: np.ndarray,
    config: GraphRev4Config,
    direction_mode_override: str | None = None,
) -> np.ndarray:
    """1顎ボリュームに対して前歯/臼歯分割ベースでseedを作成する。"""
    volume_mask = volume > 0
    anterior_threshold = config.threshold if config.anterior_threshold is None else config.anterior_threshold
    molar_threshold = config.threshold if config.molar_threshold is None else config.molar_threshold

    anterior_volume_mask, molar_volume_mask, _, _, _ = split_volume_anterior_molar_by_pca(
        volume,
        split_percentile=config.split_percentile,
        ambiguous_band_px=config.ambiguous_band_px,
    )

    combined_seed = np.zeros_like(volume, dtype=np.uint16)
    next_seed_label = 1
    region_specs = {
        "anterior": {"mask": anterior_volume_mask, "threshold": int(anterior_threshold)},
        "molar": {"mask": molar_volume_mask, "threshold": int(molar_threshold)},
    }

    direction_mode = (
        config.direction_mode if direction_mode_override is None else str(direction_mode_override)
    )

    for region_spec in region_specs.values():
        region_mask = region_spec["mask"]
        region_threshold = region_spec["threshold"]
        region_volume = np.where(volume_mask & region_mask, volume, 0).astype(volume.dtype)

        _, seed_volume, _ = graph_seed_volume_bidirectional(
            region_volume,
            threshold=region_threshold,
            labels_connectivity=config.labels_connectivity,
            inverse_volume=config.inverse_volume,
            direction_mode=direction_mode,
        )
        relabeled_seed, next_seed_label = relabel_seed_volume(seed_volume, start_label=next_seed_label)
        combined_seed = merge_seed_non_overlap(combined_seed, relabeled_seed)

    combined_seed[~volume_mask] = 0
    return combined_seed


def run_pipeline_for_jaw_ranges(
    config: GraphRev4Config,
    lower_range: tuple[int, int],
    upper_range: tuple[int, int],
    trim_keep_ratio: float = 0.5,
):
    """上顎/下顎の範囲指定でパイプラインを実行し、最小ファイルのみ保存する。"""
    ensure_dir(config.output_dir)
    original_volume = load_volume(config.input_path)
    merged_seed_full = np.zeros_like(original_volume, dtype=np.uint16)
    next_global_label = 1

    specs = [
        JawRangeSpec(name="lower_jaw", start_slice=lower_range[0], end_slice=lower_range[1]),
        JawRangeSpec(name="upper_jaw", start_slice=upper_range[0], end_slice=upper_range[1]),
    ]

    for spec in specs:
        if spec.end_slice > original_volume.shape[0]:
            raise ValueError(
                f"{spec.name} range {spec.start_slice}:{spec.end_slice} exceeds depth {original_volume.shape[0]}"
            )

        jaw_dir = os.path.join(config.output_dir, spec.name)
        ensure_dir(jaw_dir)

        jaw_volume = original_volume[spec.start_slice : spec.end_slice]
        if not np.any(jaw_volume > 0):
            raise ValueError(f"No foreground voxel in {spec.name} range {spec.start_slice}:{spec.end_slice}")

        # 1) 顎のトリミングボリューム
        jaw_trimmed_volume_path = os.path.join(jaw_dir, f"{spec.name}_trimmed_volume.tif")
        tiff.imwrite(jaw_trimmed_volume_path, jaw_volume.astype(np.float32))

        # 2) edit前seed
        jaw_direction_mode = "reverse_only" if spec.name == "upper_jaw" else config.direction_mode
        seed_before_edit = _build_seed_for_jaw_region(
            jaw_volume,
            config,
            direction_mode_override=jaw_direction_mode,
        )
        seed_before_path = os.path.join(jaw_dir, f"{spec.name}_seed_before_edit.tif")
        tiff.imwrite(seed_before_path, seed_before_edit.astype(np.uint16))

        # 3) edit後seed（trim + relabel）
        trimmed_mask = _trim_volume_by_top_far_region(seed_before_edit > 0, keep_ratio=trim_keep_ratio)
        seed_after_edit = _relabel_components_per_slice(trimmed_mask)
        seed_after_path = os.path.join(jaw_dir, f"{spec.name}_seed_after_edit.tif")
        tiff.imwrite(seed_after_path, seed_after_edit.astype(np.uint16))

        relabeled_global_seed, next_global_label = relabel_seed_volume(
            seed_after_edit,
            start_label=next_global_label,
        )
        merged_seed_full[spec.start_slice : spec.end_slice] = merge_seed_non_overlap(
            merged_seed_full[spec.start_slice : spec.end_slice],
            relabeled_global_seed,
        )

        # 4) watershed + 5) watershed color
        watershed_path = os.path.join(jaw_dir, f"{spec.name}_watershed.tif")
        watershed_3d_volume(
            volume=jaw_volume,
            seed_volume=seed_after_edit,
            labels_out=watershed_path,
            save_distance_volume=False,
        )

    merged_seed_full[original_volume <= 0] = 0
    combined_seed_path = os.path.join(config.output_dir, "combined_jaw_seed_after_edit.tif")
    tiff.imwrite(combined_seed_path, merged_seed_full.astype(np.uint16))

    combined_watershed_path = os.path.join(config.output_dir, "combined_jaw_watershed.tif")
    watershed_3d_volume(
        volume=original_volume,
        seed_volume=merged_seed_full,
        labels_out=combined_watershed_path,
        save_distance_volume=False,
    )


def graph_main_anterior_molar_pca_rev4(config: GraphRev4Config):
    """rev4の本流パイプラインを実行する。

    Args:
        config: 実行設定を保持したGraphRev4Config。

    Returns:
        統合seedボリュームと実行情報辞書。
    """
    ensure_dir(config.output_dir)
    original_volume = load_volume(config.input_path)

    aligned_volume, occlusal_info = align_occlusal_plane(
        original_volume,
        enabled=config.enable_occlusal_alignment,
        spacing_zyx=config.occlusal_spacing_zyx,
        percentile=config.occlusal_percentile,
        side=config.occlusal_side,
        pad=config.occlusal_pad,
        inplane_k45=config.occlusal_inplane_k45,
    )

    execution_slice_mask, normalized_slice_range = create_execution_slice_mask(
        aligned_volume.shape[0],
        execution_slice_range=config.execution_slice_range,
    )
    volume = apply_execution_slice_mask(aligned_volume, execution_slice_mask)
    volume_mask = volume > 0
    if not np.any(volume_mask):
        raise ValueError(
            f"No foreground voxel in execution range: {normalized_slice_range}"
        )

    anterior_threshold = config.threshold if config.anterior_threshold is None else config.anterior_threshold
    molar_threshold = config.threshold if config.molar_threshold is None else config.molar_threshold

    (
        anterior_volume_mask,
        molar_volume_mask,
        split_label_volume,
        split_line_xy,
        pca_info,
    ) = split_volume_anterior_molar_by_pca(
        volume,
        split_percentile=config.split_percentile,
        ambiguous_band_px=config.ambiguous_band_px,
    )
    tiff.imwrite(
        os.path.join(config.output_dir, "anterior_molar_pca_split_labels.tif"),
        split_label_volume.astype(np.uint8),
    )

    split_overlay_volume = make_anterior_molar_split_overlay_volume(
        volume=volume,
        split_line_xy=split_line_xy,
        line_radius_px=2,
    )
    split_overlay_path = os.path.join(config.output_dir, "anterior_molar_pca_split_overlay.tif")
    tiff.imwrite(split_overlay_path, split_overlay_volume, photometric="rgb")

    region_boundary_slice_summary = summarize_region_boundary_slices_from_split_labels(
        split_label_volume
    )

    combined_seed = np.zeros_like(volume, dtype=np.uint16)
    next_seed_label = 1
    region_infos = {}
    graph_plot_paths = {}
    region_specs = {
        "anterior": {
            "mask": anterior_volume_mask,
            "threshold": int(anterior_threshold),
        },
        "molar": {
            "mask": molar_volume_mask,
            "threshold": int(molar_threshold),
        },
    }

    for region_name, region_spec in region_specs.items():
        region_mask = region_spec["mask"]
        region_threshold = region_spec["threshold"]
        region_volume = np.where(volume_mask & region_mask, volume, 0).astype(volume.dtype)

        graph, seed_volume, graph_info = graph_seed_volume_bidirectional(
            region_volume,
            threshold=region_threshold,
            labels_connectivity=config.labels_connectivity,
            inverse_volume=config.inverse_volume,
            direction_mode=config.direction_mode,
        )

        if config.plot_execution_slice_graphs:
            graph_plot_path = os.path.join(config.output_dir, f"{region_name}_execution_slice_graph.png")
            saved_graph_path = plot_graph_for_slice_range(
                graph,
                slice_range=normalized_slice_range,
                output_path=graph_plot_path,
                title=f"{region_name} graph slices {normalized_slice_range or 'all'}",
                show=config.show_graph_plots,
            )
            graph_plot_paths[region_name] = saved_graph_path
            graph_info["graph_plot_path"] = saved_graph_path

        relabeled_seed, next_seed_label = relabel_seed_volume(seed_volume, start_label=next_seed_label)
        combined_seed = merge_seed_non_overlap(combined_seed, relabeled_seed)

        tiff.imwrite(
            os.path.join(config.output_dir, f"{region_name}_seed_volume.tif"),
            relabeled_seed.astype(np.uint16),
        )
        region_infos[region_name] = graph_info

    combined_seed[~volume_mask] = 0
    seed_path = os.path.join(config.output_dir, "combined_anterior_molar_seed_volume.tif")
    tiff.imwrite(seed_path, combined_seed.astype(np.uint16))

    watershed_output_path = None
    if config.run_watershed:
        watershed_output_path = os.path.join(config.output_dir, "watershed_anterior_molar_rev4.tif")
        watershed_3d_volume(
            volume=volume,
            seed_volume=combined_seed,
            labels_out=watershed_output_path,
        )

    resolved_direction_mode = (
        str(config.direction_mode)
        if config.direction_mode is not None
        else ("bidirectional" if config.inverse_volume else "forward_only")
    )

    info = {
        "input_path": config.input_path,
        "threshold": int(config.threshold),
        "anterior_threshold": int(anterior_threshold),
        "molar_threshold": int(molar_threshold),
        "labels_connectivity": int(config.labels_connectivity),
        "inverse_volume": bool(config.inverse_volume),
        "direction_mode": resolved_direction_mode,
        "execution_slice_range": normalized_slice_range,
        "plot_execution_slice_graphs": bool(config.plot_execution_slice_graphs),
        "graph_plot_paths": graph_plot_paths,
        "split_overlay_path": split_overlay_path,
        "region_boundary_slice_summary": region_boundary_slice_summary,
        "pca_split": pca_info,
        "occlusal_alignment": occlusal_info,
        "region_graphs": region_infos,
        "seed_path": seed_path,
        "watershed_output_path": watershed_output_path,
        "workflow": "load -> occlusal alignment -> execution mask -> PCA split -> directional graph seeds per region -> seed merge -> watershed",
    }
    info_path = os.path.join(config.output_dir, "rev4_pca_graph_watershed_info.json")
    save_json(info, info_path)

    return combined_seed, info


def run_rev4_example():
    """サンプル設定でrev4パイプラインを実行する。

    Args:
        なし。

    Returns:
        統合seedボリュームと実行情報辞書。
    """
    config = GraphRev4Config(
        # input_path=r"D:\_study\ImageProcessing\study\Watershed\Data\Output\test\edited_volume.tif",
        input_path=r"D:\_study\ImageProcessing\study\Watershed\Data\Output\test_rev2\lower_jaw.tif",
        output_dir=r"D:\_study\ImageProcessing\study\Watershed\Data\Output\test_rev2",
        threshold=1000,
        anterior_threshold=1000,
        molar_threshold=2000,
        labels_connectivity=2,
        split_percentile=50.0,
        ambiguous_band_px=0.0,
        inverse_volume=True,
        direction_mode="forward_only", # forward_only / reverse_only / bidirectional
        execution_slice_range=None,
        plot_execution_slice_graphs=False,
        show_graph_plots=False,
        run_watershed=True,
        enable_occlusal_alignment=True,
    )
    combined_seed, info = graph_main_anterior_molar_pca_rev4(config)
    print(f"anterior threshold: {info['anterior_threshold']}")
    print(f"molar threshold: {info['molar_threshold']}")
    print(f"direction mode: {info['direction_mode']}")
    print(f"execution slice range: {info['execution_slice_range']}")
    print(f"seed labels: {combined_seed.max()}")
    print(f"outputs: {config.output_dir}")
    return combined_seed, info


if __name__ == "__main__":
    # ---- 実行設定（必要に応じてここを書き換える） ----
    INPUT_PATH = r"D:\_study\ImageProcessing\study\Watershed\Data\Output\test_rev2\edited_volume.tif"
    OUTPUT_DIR = r"D:\_study\ImageProcessing\study\Watershed\Data\Output\test_rev2"

    # 0-based start:end（endは排他的）
    LOWER_RANGE = (0, 237)
    UPPER_RANGE = (154, 373)

    THRESHOLD = 1000
    ANTERIOR_THRESHOLD = 1000
    MOLAR_THRESHOLD = 2000
    LABELS_CONNECTIVITY = 2
    SPLIT_PERCENTILE = 50.0
    AMBIGUOUS_BAND_PX = 0.0
    INVERSE_VOLUME = True
    DIRECTION_MODE = "forward_only"  # forward_only / reverse_only / bidirectional
    TRIM_KEEP_RATIO = 0.5
    # ---------------------------------------------

    cfg = GraphRev4Config(
        input_path=INPUT_PATH,
        output_dir=OUTPUT_DIR,
        threshold=THRESHOLD,
        anterior_threshold=ANTERIOR_THRESHOLD,
        molar_threshold=MOLAR_THRESHOLD,
        labels_connectivity=LABELS_CONNECTIVITY,
        split_percentile=SPLIT_PERCENTILE,
        ambiguous_band_px=AMBIGUOUS_BAND_PX,
        inverse_volume=INVERSE_VOLUME,
        direction_mode=DIRECTION_MODE,
        run_watershed=True,
        enable_occlusal_alignment=False,
    )

    run_pipeline_for_jaw_ranges(
        config=cfg,
        lower_range=LOWER_RANGE,
        upper_range=UPPER_RANGE,
        trim_keep_ratio=TRIM_KEEP_RATIO,
    )
# 根管を二値化で抽出する
import numpy as np
from skimage.filters import threshold_otsu
from scipy import ndimage as ndi
from scipy.ndimage import gaussian_filter1d


def extract_root_canal(volume: np.ndarray) -> tuple[np.ndarray, float]:
    """
    根管を二値化で抽出する関数
    二値化の値はヒストグラム解析で平坦部から増加部への移行点を検出して決定する。

    Parameters:
        volume (np.ndarray): 入力の3D画像データ

    Returns:
        tuple: (volume, threshold)
            volume (np.ndarray): 0: 背景, 1: 根管
            threshold (float): ヒストグラム解析で検出された二値化の閾値
    """

    # ヒストグラム解析で平坦部から増加部への移行点を検出
    threshold = analyze_histogram(volume)

    volume = np.where(
        volume == 0,
        0,
        np.where(volume > threshold, 2, 1)
    ).astype(np.uint16)

    # 膨張してノイズ除去
    mask = (volume == 1)
    mask = ndi.binary_opening(mask, structure=np.ones((3, 3, 3), dtype=bool), iterations=1)
    mask = ndi.binary_dilation(mask, structure=np.ones((3, 3, 3), dtype=bool), iterations=1)

    # 根管だけの 2 値化にする
    root_volume_binary = np.where(mask, 1, 0).astype(np.uint16)

    return root_volume_binary, threshold


def analyze_histogram(volume):
    """
    Volume のヒストグラムを解析し、平坦部から増加部への移行点を検出する関数。

    Parameters:
        volume: 3D numpy array (フレーム, 高さ, 幅)
    
    Returns:
        bin_centers[intensity]: 平坦部から増加部への移行点の値
    """
    counts, bin_edges = np.histogram(volume, bins=256)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    background_bin_index = np.searchsorted(bin_edges, 0, side='right') - 1
    plot_counts = counts.astype(float)

    # 平坦部の背景値を補間する
    if 0 < background_bin_index < counts.size - 1:
        plot_counts[background_bin_index] = (
            counts[background_bin_index - 1] + counts[background_bin_index + 1]
        ) / 2

    # 平坦部から増加部への移行点を検出する
    start_index, dynamic_threshold, smoothed_counts = find_rising_start(bin_centers, plot_counts)
    if start_index is None:
        print("Rising start was not detected.")
    else:
        print(f"Rising start: index={start_index}, intensity={bin_centers[start_index]:.3f}, slope threshold={dynamic_threshold:.3f}")
    return bin_centers[start_index] if start_index is not None else None

def find_rising_start(bin_centers, counts, smoothing_sigma=2, consecutive_bins=3):
    """
    平坦部から継続的な増加へ移行する最初のビンを検出する関数。
    """
    smoothed_counts = gaussian_filter1d(counts.astype(float), sigma=smoothing_sigma)
    slopes = np.gradient(smoothed_counts, bin_centers)
    slope_median = np.median(slopes)
    slope_mad = np.median(np.abs(slopes - slope_median))
    dynamic_threshold = slope_median + 3 * 1.4826 * slope_mad
    rising_bins = slopes > dynamic_threshold
    consecutive_rising = np.convolve(
        rising_bins.astype(int), np.ones(consecutive_bins, dtype=int), mode='valid'
    ) == consecutive_bins

    start_indices = np.flatnonzero(consecutive_rising)
    if start_indices.size == 0:
        return None, dynamic_threshold, smoothed_counts

    return start_indices[0], dynamic_threshold, smoothed_counts
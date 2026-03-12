import numpy as np
import tifffile

def save_numpy(
    volume: np.ndarray,
    volume_rotated: np.ndarray,
    label_map: np.ndarray,
    extract_labels: list[int] = None,
    output_dir: str = None,
    index: int = None,
) -> None:
    """Save the loaded data as .npy files."""
    import os
    if extract_labels:
        label_map = np.isin(label_map, extract_labels).astype(np.uint16)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    if False:
        np.save(os.path.join(output_dir, f"volume_{index}.npy"), volume)
        np.save(os.path.join(output_dir, f"volume_rotated_{index}.npy"), volume_rotated)
        np.save(os.path.join(output_dir, f"label_map_{index}.npy"), label_map)
    # tiff形式で保存
    tifffile.imwrite(os.path.join(output_dir, f"volume_{index}.tiff"), volume.astype(np.float32))
    tifffile.imwrite(os.path.join(output_dir, f"volume_rotated_{index}.tiff"), volume_rotated.astype(np.float32))
    tifffile.imwrite(os.path.join(output_dir, f"label_map_{index}.tiff"), label_map.astype(np.uint16))
    print(f"Data saved to {output_dir}. index: {index}")
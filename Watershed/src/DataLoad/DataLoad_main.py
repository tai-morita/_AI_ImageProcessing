from __future__ import annotations

import numpy as np
import tifffile

from .pipeline import PipelineResult, run_pipeline
from .settings import DataLoadConfig
from .SaveLoadedData import save_numpy

def load_data(config: DataLoadConfig | None = None, index: int = 0) -> PipelineResult:
    """Load and decode API data into arrays and metadata."""
    resolved_config = config or DataLoadConfig.from_defaults(index)
    return run_pipeline(resolved_config)


def main(config: DataLoadConfig | None = None, index: int = 0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    # ロードしてtiffファイルで読み込む
    # label map のうち 3, 4 が歯列のラベルなので, これらを抽出して保存する
    """Return core numpy arrays: volume, rotated volume, and label map."""
    result = load_data(config, index=index)
    save_numpy(
        result.volume_data.volume,
        result.volume_rotated,
        result.label_map,
        extract_labels=[3, 4],
        output_dir="./study/Watershed/Data/Input/20260312",
        index=index,
    )
    return result.volume_data.volume, result.volume_rotated, result.label_map

if __name__ == "__main__":
    index = 0
    while True:
        try:
            print(f"Processing index: {index}")
            volume, volume_rotated, label_map = main(index=index)
            index += 1
        except Exception as exc:
            print(f"Stopped at index={index}: {exc}")
            break

    print("Download and decode finished.")
    print("Done.")
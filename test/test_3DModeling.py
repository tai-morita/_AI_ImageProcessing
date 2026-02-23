import numpy as np
import tifffile as tiff
import matplotlib.pyplot as plt
from pathlib import Path

# 3D tiffファイルを読み込み、ボリュームデータを可視化するサンプル

def visualize_3d_tiff(tiff_path, threshold=None, output_path=None, max_points=120000):
    # TIFF読み込み
    vol = tiff.imread(tiff_path)
    if vol.ndim == 3:
        vol_gray = vol
    elif vol.ndim == 4:
        # カラー4D (Z, Y, X, C) または (C, Z, Y, X) を強度ボリュームへ変換
        if vol.shape[-1] in (3, 4):
            vol_rgb = vol[..., :3]
            vol_gray = (
                0.299 * vol_rgb[..., 0]
                + 0.587 * vol_rgb[..., 1]
                + 0.114 * vol_rgb[..., 2]
            )
        elif vol.shape[0] in (3, 4):
            vol_rgb = np.moveaxis(vol[:3], 0, -1)
            vol_gray = (
                0.299 * vol_rgb[..., 0]
                + 0.587 * vol_rgb[..., 1]
                + 0.114 * vol_rgb[..., 2]
            )
        else:
            raise ValueError(f"4D入力ですがチャンネル次元を判定できません: {vol.shape}")
    else:
        raise ValueError(f"3D/4Dボリュームを想定していますが、形状が {vol.shape} です。")

    # 前処理: 正規化
    vol_min = vol_gray.min()
    vol_max = vol_gray.max()
    if vol_max == vol_min:
        raise ValueError("入力ボリュームの値が一定のため正規化できません。")
    vol = (vol_gray - vol_min) / (vol_max - vol_min)

    # しきい値で前景抽出（任意）
    if threshold is None:
        threshold = 0.5
    mask = vol > threshold

    # 3D座標取得
    z, y, x = np.where(mask)

    if "agg" in plt.get_backend().lower():
        if len(x) > max_points:
            idx = np.random.choice(len(x), size=max_points, replace=False)
            x_plot = x[idx]
            y_plot = y[idx]
            z_plot = z[idx]
        else:
            x_plot = x
            y_plot = y
            z_plot = z

        if output_path is None:
            output_path = Path(__file__).resolve().parent / "Output" / "3d_volume_visualization.html"
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            import plotly.graph_objects as go

            fig = go.Figure(
                data=[
                    go.Scatter3d(
                        x=x_plot,
                        y=y_plot,
                        z=z_plot,
                        mode='markers',
                        marker=dict(size=2, opacity=0.25),
                    )
                ]
            )
            fig.update_layout(
                title='3D Volume Visualization',
                scene=dict(xaxis_title='X', yaxis_title='Y', zaxis_title='Z'),
            )
            fig.write_html(str(output_path), include_plotlyjs=True)
            print(f"Interactive 3D saved to: {output_path}")
            print("Open this HTML in your browser to rotate/zoom the model.")
        except ImportError:
            png_path = output_path.with_suffix('.png')
            fig = plt.figure(figsize=(8, 8))
            ax = fig.add_subplot(111, projection='3d')
            ax.scatter(x_plot, y_plot, z_plot, s=1, alpha=0.1)
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Z')
            plt.title('3D Volume Visualization')
            fig.savefig(png_path, dpi=150, bbox_inches='tight')
            plt.close(fig)
            print(f"Plotly not found. Saved static image to: {png_path}")
            print("Install plotly for interactive view: pip install plotly")
    else:
        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(x, y, z, s=1, alpha=0.1)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        plt.title('3D Volume Visualization')
        plt.show()

if __name__ == "__main__":
    # サンプルパスを適宜変更
    visualize_3d_tiff("./study/test/Output/data_053.tif")
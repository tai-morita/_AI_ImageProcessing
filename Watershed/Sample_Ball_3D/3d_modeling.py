#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
3D TIFF を読み込み、(1) 体積表示（Volume Rendering）と (2) 等値面メッシュ を出力するスクリプト。
- 体積表示: Plotly Volume → HTML に保存
- メッシュ: marching cubes（scikit-image）→ Plotly mesh3d（HTML）＆ STL/PLY 保存（任意）

使い方:
  python tiff3d_viewer.py --input /path/to/stack.tif --outdir ./out \
      --downsample 2 --isovalue 0.5 --spacing 1.0 1.0 1.0 --no-volume

  引数詳細は `-h` 参照
"""

from __future__ import annotations
import argparse
from pathlib import Path
import sys

import numpy as np
import tifffile as tiff
from skimage import measure

# optional: trimesh があれば STL/PLY 保存可能
try:
    import trimesh
    TRIMESH_AVAILABLE = True
except Exception:
    TRIMESH_AVAILABLE = False

import plotly.graph_objects as go


def load_volume(path: Path) -> np.ndarray:
    vol = tiff.imread(str(path))  # shape: (Z, Y, X) 想定
    if vol.ndim != 3:
        raise ValueError(f"3D 配列ではありません: shape={vol.shape}")
    return vol


def normalize01(arr: np.ndarray) -> np.ndarray:
    arr = arr.astype(np.float32, copy=False)
    mn, mx = float(arr.min()), float(arr.max())
    if mx <= mn:
        return np.zeros_like(arr, dtype=np.float32)
    return (arr - mn) / (mx - mn)


def downsample3d(arr: np.ndarray, factor: int) -> np.ndarray:
    if factor <= 1:
        return arr
    # ざっくり間引き（最近傍）
    return arr[::factor, ::factor, ::factor]


def save_plotly_volume(v: np.ndarray, out_html: Path, opacity: float = 0.12,
                       surface_count: int = 15, colorscale: str = "Gray") -> None:
    """Plotly Volume で体積表示（HTML 保存）"""
    # 座標（整数グリッド）。meshgrid せず、ravel に合わせて展開
    zdim, ydim, xdim = v.shape
    x = np.arange(xdim).repeat(zdim * ydim)
    y = np.tile(np.arange(ydim).repeat(zdim), xdim)
    z = np.tile(np.arange(zdim), ydim * xdim)
    values = v.ravel(order="C")

    fig = go.Figure(data=go.Volume(
        x=x, y=y, z=z, value=values,
        opacity=opacity,
        surface_count=surface_count,
        caps=dict(x_show=False, y_show=False, z_show=False),
        colorscale=colorscale
    ))
    fig.update_layout(scene_aspectmode="data", width=1000, height=800, title="Volume Rendering")
    fig.write_html(str(out_html), include_plotlyjs="cdn")
    print(f"[INFO] Volume HTML saved: {out_html}")


def marching_cubes_mesh(v: np.ndarray, isovalue: float, spacing: tuple[float, float, float]) -> tuple[np.ndarray, np.ndarray]:
    """等値面メッシュ（頂点・三角形）を生成"""
    verts, faces, normals, values = measure.marching_cubes(
        volume=v, level=isovalue, spacing=spacing
    )
    # faces: (N,3) int
    return verts, faces


def save_plotly_mesh(verts: np.ndarray, faces: np.ndarray, out_html: Path) -> None:
    """Plotly mesh3d でメッシュ表示（HTML 保存）"""
    i, j, k = faces[:, 0], faces[:, 1], faces[:, 2]
    mesh = go.Mesh3d(
        x=verts[:, 0], y=verts[:, 1], z=verts[:, 2],
        i=i, j=j, k=k,
        color="orange", opacity=1.0, lighting=dict(ambient=0.5, diffuse=0.8, specular=0.3)
    )
    fig = go.Figure(data=[mesh])
    fig.update_layout(scene_aspectmode="data", width=1000, height=800, title="Isosurface (Marching Cubes)")
    fig.write_html(str(out_html), include_plotlyjs="cdn")
    print(f"[INFO] Mesh HTML saved: {out_html}")


def save_mesh_file(verts: np.ndarray, faces: np.ndarray, out_path: Path) -> None:
    """STL/PLY などへ保存（trimesh 使用）"""
    if not TRIMESH_AVAILABLE:
        print(f"[WARN] trimesh 未導入のため保存スキップ: {out_path.name}")
        return
    tri = trimesh.Trimesh(vertices=verts, faces=faces, process=False)
    tri.export(str(out_path))
    print(f"[INFO] Mesh saved: {out_path}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="3D TIFF volume viewer & isosurface modeler")
    p.add_argument("--input", "-i", type=str, required=True, help="入力 3D TIFF のパス")
    p.add_argument("--outdir", "-o", type=str, default="./out", help="出力フォルダ")
    p.add_argument("--downsample", "-d", type=int, default=1, help="間引き倍率（2なら 1/2）")
    p.add_argument("--isovalue", "-t", type=float, default=0.5, help="メッシュ抽出のしきい値（0..1 正規化後）")
    p.add_argument("--spacing", "-s", type=float, nargs=3, default=[1.0, 1.0, 1.0],
                   metavar=("DZ", "DY", "DX"),
                   help="voxel サイズ（Z, Y, X の順）。実寸を反映したいときに指定")
    p.add_argument("--no-volume", action="store_true", help="体積表示（Volume）をスキップ")
    p.add_argument("--no-mesh", action="store_true", help="メッシュ生成をスキップ")
    p.add_argument("--save-stl", type=str, default="", help="STL 保存パス（例: model.stl）")
    p.add_argument("--save-ply", type=str, default="", help="PLY 保存パス（例: model.ply）")
    return p.parse_args()


def main():
    args = parse_args()
    in_path = Path(args.input)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    if not in_path.exists():
        print(f"[ERROR] not found: {in_path}")
        sys.exit(1)

    print(f"[INFO] Load: {in_path}")
    vol = load_volume(in_path)           # (Z,Y,X)
    print(f"[INFO] shape={vol.shape}, dtype={vol.dtype}")

    v = normalize01(vol)
    if args.downsample > 1:
        v = downsample3d(v, args.downsample)
        print(f"[INFO] downsampled -> shape={v.shape}")

    # 体積表示（Plotly Volume）
    if not args.no-volume:
        vol_html = outdir / (in_path.stem + "_volume.html")
        save_plotly_volume(v, vol_html)

    # メッシュ生成（marching cubes → Plotly mesh3d）
    if not args.no_mesh:
        isoval = float(args.isovalue)
        spacing = tuple(float(s) for s in args.spacing)  # (dz, dy, dx)
        print(f"[INFO] marching cubes... isovalue={isoval}, spacing={spacing}")
        verts, faces = marching_cubes_mesh(v, isoval, spacing)
        print(f"[INFO] verts={len(verts)}, faces={len(faces)}")

        mesh_html = outdir / (in_path.stem + "_mesh.html")
        save_plotly_mesh(verts, faces, mesh_html)

        # STL / PLY 保存（任意）
        if args.save_stl:
            save_mesh_file(verts, faces, outdir / args.save_stl)
        if args.save_ply:
            save_mesh_file(verts, faces, outdir / args.save_ply)

    print("[INFO] done.")


if __name__ == "__main__":
    main()
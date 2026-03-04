#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
3d_modeling.py
dataset.py で作成した 3D ボリューム(.npy/.tif) を読み込み、
- (A) 体積レンダリング（Plotly Volume → HTML）
- (B) 等値面メッシュ抽出（marching_cubes）→ Plotly mesh3d（HTML）＋ STL/PLY 保存（任意）
を行う。

例:
  python 3d_modeling.py --input /workspace/out/spheres.npy --outdir /workspace/out_vis \
    --mode both --isovalue 0.4 --downsample 2 --save-stl model.stl
"""

from __future__ import annotations
import argparse
from pathlib import Path
import sys
import numpy as np
import tifffile as tiff
from skimage import measure, filters
import plotly.graph_objects as go

try:
    import trimesh
    TRIMESH_AVAILABLE = True
except Exception:
    TRIMESH_AVAILABLE = False

def parse_args():
    p = argparse.ArgumentParser(description="View 3D volume as volume-rendering or isosurface mesh")
    p.add_argument("--input", "-i", type=str, required=True, help=".npy or .tif")
    p.add_argument("--outdir", "-o", type=str, default="./out_vis", help="output directory")
    p.add_argument("--mode", "-m", type=str, choices=["volume", "mesh", "both"], default="both")
    p.add_argument("--downsample", "-d", type=int, default=1, help="subsample factor (2=half)")
    p.add_argument("--isovalue", "-t", type=float, default=0.5, help="isosurface threshold (0..1)")
    p.add_argument("--auto-isovalue", type=str, choices=["none","otsu","percentile"], default="none",
                   help="auto-select isovalue: none|otsu|percentile")
    p.add_argument("--percentile", type=float, default=95.0, help="percentile for auto-isovalue when using 'percentile'")
    p.add_argument("--spacing", "-s", type=float, nargs=3, default=[1.0,1.0,1.0], metavar=("DZ","DY","DX"),
                   help="voxel spacing for mesh (Z Y X)")
    p.add_argument("--save-stl", type=str, default="", help="filename to save STL (optional)")
    p.add_argument("--save-ply", type=str, default="", help="filename to save PLY (optional)")
    return p.parse_args()

def load_volume(path: Path) -> np.ndarray:
    if path.suffix.lower() == ".npy":
        vol = np.load(path)
    elif path.suffix.lower() in [".tif", ".tiff"]:
        vol = tiff.imread(str(path)).astype(np.float32)
        # 16bit → 0..1 正規化
        if vol.dtype == np.uint16 or vol.max() > 1.0:
            vol = vol / 65535.0
    else:
        raise ValueError(f"unsupported format: {path.suffix}")
    if vol.ndim != 3:
        raise ValueError(f"not 3D: shape={vol.shape}")
    # クリップして安全に
    vol = np.clip(vol.astype(np.float32), 0.0, 1.0)
    return vol

def downsample3d(v: np.ndarray, factor: int) -> np.ndarray:
    if factor <= 1:
        return v
    return v[::factor, ::factor, ::factor]

def save_plotly_volume(v: np.ndarray, out_html: Path, opacity: float = 0.12, surface_count: int = 15):
    Z, Y, X = v.shape
    x = np.arange(X).repeat(Z * Y)
    y = np.tile(np.arange(Y).repeat(Z), X)
    z = np.tile(np.arange(Z), Y * X)
    fig = go.Figure(data=go.Volume(
        x=x, y=y, z=z, value=v.ravel(order="C"),
        opacity=opacity, surface_count=surface_count,
        caps=dict(x_show=False,y_show=False,z_show=False),
        colorscale="Gray"
    ))
    fig.update_layout(scene_aspectmode="data", width=1000, height=800, title="Volume Rendering")
    out_html.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(out_html), include_plotlyjs="cdn")
    print(f"[INFO] volume html: {out_html}")

def marching_cubes_mesh(v: np.ndarray, isovalue: float, spacing: tuple[float,float,float]):
    verts, faces, normals, values = measure.marching_cubes(volume=v, level=isovalue, spacing=spacing)
    return verts, faces

def save_plotly_mesh(verts: np.ndarray, faces: np.ndarray, out_html: Path):
    i, j, k = faces[:,0], faces[:,1], faces[:,2]
    mesh = go.Mesh3d(
        x=verts[:,0], y=verts[:,1], z=verts[:,2],
        i=i, j=j, k=k, color="orange", opacity=1.0,
        lighting=dict(ambient=0.5, diffuse=0.9, specular=0.2)
    )
    fig = go.Figure(data=[mesh])
    fig.update_layout(scene_aspectmode="data", width=1000, height=800, title="Isosurface (Marching Cubes)")
    out_html.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(out_html), include_plotlyjs="cdn")
    print(f"[INFO] mesh html: {out_html}")

def save_mesh_file(verts: np.ndarray, faces: np.ndarray, out_path: Path):
    if not TRIMESH_AVAILABLE:
        print(f"[WARN] trimesh 未導入のため保存スキップ: {out_path.name}")
        return
    tri = trimesh.Trimesh(vertices=verts, faces=faces, process=False)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tri.export(str(out_path))
    print(f"[INFO] mesh saved: {out_path}")

def main():
    args = parse_args()
    in_path = Path(args.input)
    outdir  = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    if not in_path.exists():
        print(f"[ERROR] not found: {in_path}")
        sys.exit(1)

    v = load_volume(in_path)
    print(f"[INFO] volume loaded: shape={v.shape}, dtype={v.dtype}, min={v.min():.3f}, max={v.max():.3f}")

    if args.downsample > 1:
        v = downsample3d(v, args.downsample)
        print(f"[INFO] downsampled -> {v.shape}")

    if args.mode in ("volume", "both"):
        save_plotly_volume(v, outdir / (in_path.stem + "_volume.html"))

    if args.mode in ("mesh", "both"):
        vmin, vmax = float(v.min()), float(v.max())
        isovalue = float(args.isovalue)
        # auto-select isovalue if requested
        if args.auto_isovalue == "otsu":
            # Otsu は2値化のしきい値を求める。体積を1Dにして計算
            try:
                isovalue = float(filters.threshold_otsu(v.ravel()))
            except Exception:
                # フォールバック: 中間値
                isovalue = (vmin + vmax) * 0.5
            print(f"[INFO] auto isovalue (otsu): {isovalue:.3f}")
        elif args.auto_isovalue == "percentile":
            q = max(0.0, min(100.0, float(args.percentile)))
            isovalue = float(np.percentile(v, q))
            print(f"[INFO] auto isovalue (p{q:.1f}): {isovalue:.3f}")

        # 範囲外なら安全にクランプして回避
        if not (vmin <= isovalue <= vmax):
            # 中間にクランプ（min==max の場合はその値）
            isovalue = (vmin + vmax) * 0.5
            print(f"[WARN] isovalue out of range [{vmin:.3f},{vmax:.3f}] -> clamped to {isovalue:.3f}")

        spacing  = tuple(float(s) for s in args.spacing)  # (dz,dy,dx)
        print(f"[INFO] marching cubes... isovalue={isovalue}, spacing={spacing}")
        verts, faces = marching_cubes_mesh(v, isovalue, spacing)
        print(f"[INFO] verts={len(verts)}, faces={len(faces)}")
        save_plotly_mesh(verts, faces, outdir / (in_path.stem + "_mesh.html"))
        if args.save_stl:
            save_mesh_file(verts, faces, outdir / args.save_stl)
        if args.save_ply:
            save_mesh_file(verts, faces, outdir / args.save_ply)

    print("[INFO] done.")

if __name__ == "__main__":
    main()

"""
python ./Watershed/Sample_Ball_3D/3d_modeling.py \
  --input ./out/spheres.tif \
  --outdir ./out \
  --mode both \
  --downsample 2 \
  --isovalue 0.5 \
  --spacing 1.0 1.0 1.0
"""
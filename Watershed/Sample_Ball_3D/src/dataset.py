#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
dataset.py
重なり合う 2 球の 3D ボリューム（(Z, Y, X)）を生成して保存します。
- 生成を TensorFlow (GPU) で実行（--backend np で NumPy も可）
- 出力は .npy（必須）＋ .tif（任意）

例:
  python dataset.py --shape 256 256 256 \
    --center1 80 128 128 --radius1 70 \
    --center2 128 128 128 --radius2 70 \
    --out /workspace/out/spheres.npy --tif /workspace/out/spheres.tif
"""

from __future__ import annotations
import argparse
from pathlib import Path
import sys
import numpy as np
import tifffile as tiff

def parse_args():
    p = argparse.ArgumentParser(description="Create overlapping two-sphere 3D volume")
    p.add_argument("--shape", type=int, nargs=3, default=[256, 256, 256], metavar=("Z","Y","X"),
                   help="volume size (Z Y X)")
    p.add_argument("--center1", type=float, nargs=3, default=[96, 128, 96], metavar=("Z","Y","X"),
                   help="center of sphere1 in voxels")
    p.add_argument("--radius1", type=float, default=70.0, help="radius of sphere1 in voxels")
    p.add_argument("--center2", type=float, nargs=3, default=[160, 128, 160], metavar=("Z","Y","X"),
                   help="center of sphere2 in voxels")
    p.add_argument("--radius2", type=float, default=70.0, help="radius of sphere2 in voxels")
    p.add_argument("--smooth-edge", type=float, default=2.0,
                   help="境界をなめらかにする幅(vox)。0でカチッとした2値")
    p.add_argument("--intensity1", type=float, default=1.0, help="sphere1 intensity")
    p.add_argument("--intensity2", type=float, default=1.0, help="sphere2 intensity")
    p.add_argument("--noise", type=float, default=0.0, help="additive gaussian noise sigma (0..1 正規化後)")
    p.add_argument("--interior-profile", type=str, choices=["flat","linear","quadratic","gaussian"], default="flat",
                   help="球内部の強度プロファイル: flat=一定, linear=中心から線形低下, quadratic=二乗低下, gaussian=中心ガウス")
    p.add_argument("--gaussian-sigma", type=float, default=0.4,
                   help="gaussian プロファイルのσ(半径に対する比)。例: 0.4*r")
    p.add_argument("--blend", type=str, choices=["sum","max","mean","avg_active"], default="sum",
                   help="合成方法: sum=加算(クリップ), max=最大, mean=単純平均, avg_active=非ゼロ寄与の平均")
    p.add_argument("--backend", type=str, choices=["tf", "np"], default="tf",
                   help="生成バックエンド：tf=TensorFlow(GPU), np=NumPy(CPU)")
    p.add_argument("--out", type=str, required=True, help="output .npy path")
    p.add_argument("--tif", type=str, default="", help="also save .tif stack (optional)")
    return p.parse_args()

def save_outputs(vol: np.ndarray, out_npy: Path, out_tif: Path|None):
    out_npy.parent.mkdir(parents=True, exist_ok=True)
    np.save(out_npy, vol.astype(np.float32))
    print(f"[INFO] saved .npy: {out_npy} shape={vol.shape} dtype=float32, min={vol.min():.3f}, max={vol.max():.3f}")
    if out_tif:
        out_tif.parent.mkdir(parents=True, exist_ok=True)
        # 0..1 を 16bit にスケールして保存（任意）
        arr16 = np.clip(vol * 65535.0, 0, 65535).astype(np.uint16)
        tiff.imwrite(out_tif, arr16)
        print(f"[INFO] saved .tif: {out_tif} (uint16)")

def main_np(args):
    Z, Y, X = map(int, args.shape)
    z = np.arange(Z)[:, None, None]
    y = np.arange(Y)[None, :, None]
    x = np.arange(X)[None, None, :]

    def sphere_field(cz, cy, cx, r, intensity):
        # 距離場（負が内部）を滑らかに 0..1 にマップ
        d = np.sqrt((z - cz)**2 + (y - cy)**2 + (x - cx)**2)
        edge = max(1e-6, float(args.smooth_edge))
        # 境界の滑らかさ（外側への減衰）
        mask = 1.0 / (1.0 + np.exp((d - r) / edge))
        # 内部プロファイル
        dr = d / max(1e-6, float(r))
        prof = args.interior_profile
        if prof == "flat":
            w = 1.0
        elif prof == "linear":
            w = np.clip(1.0 - dr, 0.0, 1.0)
        elif prof == "quadratic":
            w = np.clip(1.0 - dr**2, 0.0, 1.0)
        elif prof == "gaussian":
            sigma = max(1e-6, float(args.gaussian_sigma) * float(r))
            w = np.exp(- (d**2) / (2.0 * sigma**2))
        else:
            w = 1.0
        v = intensity * (w * mask)
        return v

    v1 = sphere_field(*args.center1, args.radius1, args.intensity1)
    v2 = sphere_field(*args.center2, args.radius2, args.intensity2)

    # 合成（ブレンド）
    if args.blend == "sum":
        vol = v1 + v2
    elif args.blend == "max":
        vol = np.maximum(v1, v2)
    elif args.blend == "mean":
        vol = (v1 + v2) * 0.5
    elif args.blend == "avg_active":
        denom = (v1 > 0).astype(np.float32) + (v2 > 0).astype(np.float32)
        denom[denom == 0] = 1.0
        vol = (v1 + v2) / denom
    else:
        vol = v1 + v2
    vol = np.clip(vol, 0.0, 1.0)

    if args.noise > 0:
        vol = np.clip(vol + np.random.normal(0, args.noise, size=vol.shape).astype(np.float32), 0.0, 1.0)

    return vol.astype(np.float32)

def main_tf(args):
    import tensorflow as tf
    # GPU メモリを必要分だけ確保（推奨）
    for g in tf.config.list_physical_devices("GPU"):
        try:
            tf.config.experimental.set_memory_growth(g, True)
        except Exception:
            pass

    Z, Y, X = map(int, args.shape)
    with tf.device("/GPU:0" if tf.config.list_physical_devices("GPU") else "/CPU:0"):
        z = tf.range(Z, dtype=tf.float32)[:, None, None]
        y = tf.range(Y, dtype=tf.float32)[None, :, None]
        x = tf.range(X, dtype=tf.float32)[None, None, :]

        def sphere_field(cz, cy, cx, r, intensity):
            cz = tf.constant(float(cz), tf.float32)
            cy = tf.constant(float(cy), tf.float32)
            cx = tf.constant(float(cx), tf.float32)
            r  = tf.constant(float(r),  tf.float32)
            intensity = tf.constant(float(intensity), tf.float32)

            d = tf.sqrt((z - cz)**2 + (y - cy)**2 + (x - cx)**2)
            edge = tf.constant(max(1e-6, float(args.smooth_edge)), tf.float32)
            # 境界の滑らかさ（外側への減衰）
            mask = 1.0 / (1.0 + tf.exp((d - r) / edge))
            # 内部プロファイル
            dr = d / tf.maximum(tf.constant(1e-6, tf.float32), r)
            prof = args.interior_profile
            def w_flat():
                return tf.constant(1.0, tf.float32)
            def w_linear():
                return tf.clip_by_value(1.0 - dr, 0.0, 1.0)
            def w_quadratic():
                return tf.clip_by_value(1.0 - dr*dr, 0.0, 1.0)
            def w_gaussian():
                sigma = tf.maximum(tf.constant(1e-6, tf.float32), tf.constant(float(args.gaussian_sigma), tf.float32) * r)
                return tf.exp(- (d*d) / (2.0 * sigma*sigma))
            w = tf.case({
                tf.equal(tf.constant(prof), tf.constant("flat")): w_flat,
                tf.equal(tf.constant(prof), tf.constant("linear")): w_linear,
                tf.equal(tf.constant(prof), tf.constant("quadratic")): w_quadratic,
                tf.equal(tf.constant(prof), tf.constant("gaussian")): w_gaussian,
            }, default=w_flat)
            v = intensity * (w * mask)
            return v

        v1 = sphere_field(*args.center1, args.radius1, args.intensity1)
        v2 = sphere_field(*args.center2, args.radius2, args.intensity2)
        # 合成（ブレンド）
        if args.blend == "sum":
            vol = v1 + v2
        elif args.blend == "max":
            vol = tf.maximum(v1, v2)
        elif args.blend == "mean":
            vol = (v1 + v2) * 0.5
        elif args.blend == "avg_active":
            denom = tf.cast(tf.greater(v1, 0.0), tf.float32) + tf.cast(tf.greater(v2, 0.0), tf.float32)
            denom = tf.where(tf.equal(denom, 0.0), tf.ones_like(denom), denom)
            vol = (v1 + v2) / denom
        else:
            vol = v1 + v2
        vol = tf.clip_by_value(vol, 0.0, 1.0)

        if args.noise > 0:
            noise = tf.random.normal(tf.shape(vol), mean=0.0, stddev=float(args.noise), dtype=tf.float32)
            vol = tf.clip_by_value(vol + noise, 0.0, 1.0)

        vol_np = vol.numpy().astype(np.float32)

    return vol_np

def main():
    args = parse_args()
    out_npy = Path(args.out)
    out_tif = Path(args.tif) if args.tif else None

    if args.backend == "tf":
        vol = main_tf(args)
    else:
        vol = main_np(args)

    save_outputs(vol, out_npy, out_tif)

if __name__ == "__main__":
    main()

"""
python ./Watershed/Sample_Ball_3D/dataset.py \
  --backend np \
  --shape 256 256 256 \
  --center1 100 100 100 --radius1 40 \
  --center2 200 150 200 --radius2 40 \
  --smooth-edge 2.0 \
 --interior-profile quadratic \
  --out ./out/spheres.npy \
  --tif ./out/spheres.tif
"""
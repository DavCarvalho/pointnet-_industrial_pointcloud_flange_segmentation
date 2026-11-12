"""
PREPROCESS v8 — PLY rotulado → NPZ em metros reais.

Mudanças em relação à v7:
  1. Leitura, voxel, normais e curvatura vêm de common_v8.py, o MESMO código
     que a inferência usa (na v7 a curvatura do treino era por sklearn com
     idx[:30] arbitrário e a da inferência era pelo Open3D).
  2. Curvatura pelo Open3D (estimate_covariances): sem laço em Python.
  3. Salva has_rgb / has_intensity para o treino saber quais canais são reais.

Uso:  python preprocess_v8.py
"""
import os, time, traceback
from glob import glob
import numpy as np
from common_v8 import (VOXEL_SIZE_M, NORMAL_RADIUS_M, CURV_RADIUS_M, read_ply, to_meters,
                       voxel_downsample, center, geometry, normalize_intensity)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, "../DATA"))
OUT_DIR  = os.path.abspath(os.path.join(SCRIPT_DIR, "../DATA_PREPROCESSED_v8"))


def process_file(fp, out_dir):
    name = os.path.splitext(os.path.basename(fp))[0]
    out = os.path.join(out_dir, f"{name}.npz")
    if os.path.exists(out):
        print(f"  [SKIP] {name}")
        return
    t0 = time.time()

    xyz, labels, rgb, intensity = read_ply(fp, require_labels=True)
    print(f"    Pontos: {len(xyz):,} | flange: {100 * labels.mean():.2f}%")

    xyz, was_mm = to_meters(xyz)
    print(f"    Unidade: {'mm → m' if was_mm else 'metros'}")

    xyz, labels, rgb, intensity, _ = voxel_downsample(xyz, VOXEL_SIZE_M, labels, rgb, intensity)
    print(f"    Voxel {VOXEL_SIZE_M * 1000:.0f} mm: {len(xyz):,} | flange: {100 * labels.mean():.2f}%")

    xyz = center(xyz)
    normals, curv = geometry(xyz)

    has_rgb, has_int = rgb is not None, intensity is not None
    rgb = rgb if has_rgb else np.zeros((len(xyz), 3), np.float32)
    intensity = normalize_intensity(intensity) if has_int else np.zeros((len(xyz), 1), np.float32)

    np.savez_compressed(
        out,
        xyz=xyz.astype(np.float32), labels=labels.astype(np.int64),
        normals=normals, curvature=curv, rgb=rgb.astype(np.float32),
        intensity=intensity.astype(np.float32),
        was_mm=np.array([was_mm]), has_rgb=np.array([has_rgb]), has_intensity=np.array([has_int]),
    )
    print(f"    Salvo {out} ({os.path.getsize(out) / 1e6:.1f} MB) em {time.time() - t0:.1f}s"
          f" | rgb={has_rgb} intensidade={has_int}")


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    files = sorted(glob(os.path.join(DATA_DIR, "*.ply")))
    print(f"PREPROCESS v8 — {len(files)} arquivos\n  IN:  {DATA_DIR}\n  OUT: {OUT_DIR}")
    print(f"  Voxel {VOXEL_SIZE_M * 1000:.0f} mm | normal r={NORMAL_RADIUS_M} m | curvatura r={CURV_RADIUS_M} m")
    for i, fp in enumerate(files, 1):
        print(f"\n[{i}/{len(files)}] {os.path.basename(fp)}")
        try:
            process_file(fp, OUT_DIR)
        except Exception as e:
            print(f"  ERRO: {e}")
            traceback.print_exc()


if __name__ == "__main__":
    main()

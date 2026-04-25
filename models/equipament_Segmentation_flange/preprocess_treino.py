"""
PREPROCESSAMENTO DE NUVENS DE PONTOS INDUSTRIAIS (v5 — VOXEL BIDIRECIONAL)
===========================================================================
Correção crítica: o loop de voxel agora ajusta em AMBAS as direções.

Bug das versões anteriores:
  - Se o primeiro voxel produzisse MENOS pontos que o target, o loop parava
  - Resultado: target=742k mas obtinha 45k pontos (16x abaixo!)
  - Isso fazia flange ficar com 70-95% mesmo com ratio=6

Correção v5:
  - Se n_result < target: DIMINUI o voxel (mais pontos)
  - Se n_result > target: AUMENTA o voxel (menos pontos)
  - Busca binária para convergir rápido
"""

import os
import sys
import time
from glob import glob

import numpy as np
from plyfile import PlyData


# ============================================================
# CONFIGURAÇÃO
# ============================================================

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

DATA_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, "../../DATA"))
PREPROCESSED_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, "../../DATA_PREPROCESSED_v5"))

USE_RGB = True
USE_INTENSITY = True

KNN_NORMALS = 16
CURVATURE_BLOCK_SIZE = 50_000

# Downsample balanceado
EQUIP_FLANGE_RATIO = 4       # equipamento = 4x flange → ~20% flange
MIN_EQUIP_POINTS = 50_000
MAX_EQUIP_POINTS = 800_000
VOXEL_SIZE_INITIAL = 0.01

# Tolerância: aceita resultado entre 70% e 100% do target
DOWNSAMPLE_TOLERANCE_LOW = 0.70   # aceita até 30% abaixo do target
DOWNSAMPLE_MAX_ITER = 15

SHUFFLE_BEFORE_SAVE = True
SAVE_XYZ_CENTERED = True


# ============================================================
# LEITURA DE DADOS
# ============================================================

CLASS_FIELD_CANDIDATES = [
    "scalar_Classification", "scalar_classification",
    "Classification", "classification",
    "label", "scalar_Label", "class"
]


def find_class_field(vertex_properties):
    prop_names = [p.name for p in vertex_properties]
    for name in CLASS_FIELD_CANDIDATES:
        if name in prop_names:
            return name
    raise ValueError(f"Campo de classificação não encontrado. Campos: {prop_names}")


def read_ply_with_features(filepath):
    plydata = PlyData.read(filepath)
    vertex = plydata["vertex"].data
    prop_names = [p.name for p in plydata["vertex"].properties]

    x = np.asarray(vertex["x"], dtype=np.float32)
    y = np.asarray(vertex["y"], dtype=np.float32)
    z = np.asarray(vertex["z"], dtype=np.float32)
    xyz = np.column_stack([x, y, z]).astype(np.float32)

    class_field = find_class_field(plydata["vertex"].properties)
    labels_raw = np.asarray(vertex[class_field], dtype=np.int32)
    labels = np.where(labels_raw == 1, 1, 0).astype(np.int64)

    rgb = None
    if USE_RGB and all(c in prop_names for c in ["red", "green", "blue"]):
        r = np.asarray(vertex["red"], dtype=np.float32)
        g = np.asarray(vertex["green"], dtype=np.float32)
        b = np.asarray(vertex["blue"], dtype=np.float32)
        rgb = np.column_stack([r, g, b]).astype(np.float32)

    intensity = None
    if USE_INTENSITY and "intensity" in prop_names:
        intensity = np.asarray(vertex["intensity"], dtype=np.float32).reshape(-1, 1)

    return xyz, labels, rgb, intensity


# ============================================================
# NORMAIS E CURVATURA
# ============================================================

def estimate_normals_open3d(xyz, k=16):
    import open3d as o3d
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz.astype(np.float64))
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamKNN(knn=k))
    try:
        pcd.orient_normals_consistent_tangent_plane(k=k)
    except RuntimeError:
        pass
    return np.asarray(pcd.normals, dtype=np.float32)


def compute_curvature_vectorized(xyz, k=16, block_size=50_000):
    from sklearn.neighbors import KDTree
    n = xyz.shape[0]
    curvature = np.zeros(n, dtype=np.float32)
    print(f"    Construindo KDTree para {n:,} pontos...")
    tree = KDTree(xyz)
    n_blocks = (n + block_size - 1) // block_size
    for b_idx in range(n_blocks):
        start = b_idx * block_size
        end = min(start + block_size, n)
        if (b_idx % 5 == 0) or (b_idx == n_blocks - 1):
            print(f"    Curvatura: bloco {b_idx+1}/{n_blocks} ({100*end/n:.0f}%)")
        idx = tree.query(xyz[start:end], k=k, return_distance=False)
        neighbors = xyz[idx]
        centered = neighbors - neighbors.mean(axis=1, keepdims=True)
        cov = np.einsum("bki,bkj->bij", centered, centered)
        eigvals = np.linalg.eigvalsh(cov)
        curvature[start:end] = eigvals[:, 0] / (eigvals.sum(axis=1) + 1e-8)
    return curvature.reshape(-1, 1).astype(np.float32)


# ============================================================
# NORMALIZAÇÃO
# ============================================================

def normalize_xyz_unit_sphere(xyz):
    xyz = xyz.astype(np.float32)
    center = np.mean(xyz, axis=0, keepdims=True)
    xyz = xyz - center
    radius = np.max(np.linalg.norm(xyz, axis=1))
    if radius < 1e-8:
        radius = 1.0
    return (xyz / radius).astype(np.float32)


def normalize_scalar_channel(x):
    x = x.astype(np.float32)
    mean = np.mean(x, axis=0, keepdims=True)
    std = np.std(x, axis=0, keepdims=True) + 1e-8
    return ((x - mean) / std).astype(np.float32)


def build_features(xyz_centered, normals, curvature, rgb, intensity):
    xyz_geom = normalize_xyz_unit_sphere(xyz_centered)
    normals = normals.astype(np.float32)
    normals = normals / (np.linalg.norm(normals, axis=1, keepdims=True) + 1e-8)
    curvature = normalize_scalar_channel(curvature)
    if rgb is None:
        rgb = np.zeros((len(xyz_centered), 3), dtype=np.float32)
    else:
        rgb = rgb.astype(np.float32)
        if np.max(rgb) > 1.5:
            rgb = rgb / 255.0
    if intensity is None:
        intensity = np.zeros((len(xyz_centered), 1), dtype=np.float32)
    else:
        intensity = normalize_scalar_channel(intensity)
    return np.concatenate(
        [xyz_geom, normals, curvature, rgb, intensity], axis=1
    ).astype(np.float32)


# ============================================================
# DOWNSAMPLE (CORRIGIDO — BUSCA BIDIRECIONAL)
# ============================================================

def _voxel_downsample_arrays(xyz, rgb, intensity, voxel_size):
    voxel_coords = np.floor(xyz / voxel_size).astype(np.int64)
    mins = voxel_coords.min(axis=0)
    voxel_coords -= mins
    maxes = voxel_coords.max(axis=0) + 1
    hash_keys = (
        voxel_coords[:, 0] * (maxes[1] * maxes[2]) +
        voxel_coords[:, 1] * maxes[2] +
        voxel_coords[:, 2]
    )
    unique_keys, inverse = np.unique(hash_keys, return_inverse=True)
    n_voxels = len(unique_keys)
    counts = np.bincount(inverse, minlength=n_voxels).astype(np.float32)
    counts_safe = np.maximum(counts, 1.0)

    new_xyz = np.zeros((n_voxels, 3), dtype=np.float32)
    for dim in range(3):
        new_xyz[:, dim] = np.bincount(inverse, weights=xyz[:, dim], minlength=n_voxels) / counts_safe

    new_rgb = None
    if rgb is not None:
        new_rgb = np.zeros((n_voxels, 3), dtype=np.float32)
        for dim in range(3):
            new_rgb[:, dim] = np.bincount(inverse, weights=rgb[:, dim], minlength=n_voxels) / counts_safe

    new_intensity = None
    if intensity is not None:
        new_intensity = (
            np.bincount(inverse, weights=intensity.ravel(), minlength=n_voxels) / counts_safe
        ).reshape(-1, 1).astype(np.float32)

    return new_xyz, new_rgb, new_intensity, n_voxels


def _random_subsample_arrays(xyz, rgb, intensity, target_n, rng=None):
    """Fallback: amostragem aleatória quando voxel não converge."""
    if rng is None:
        rng = np.random.default_rng(42)
    N = len(xyz)
    idx = rng.choice(N, size=min(target_n, N), replace=False)
    new_xyz = xyz[idx]
    new_rgb = rgb[idx] if rgb is not None else None
    new_int = intensity[idx] if intensity is not None else None
    return new_xyz, new_rgb, new_int, len(idx)


def class_aware_downsample(
    xyz, labels, rgb, intensity,
    equip_flange_ratio=4,
    min_equip=50_000,
    max_equip=800_000,
    initial_voxel_size=0.01,
    tolerance_low=0.70,
    max_iter=15,
):
    """
    Downsample sensível à classe com BUSCA BIDIRECIONAL do voxel.

    Bug corrigido: versões anteriores só aumentavam o voxel (quando
    tinha pontos demais), mas nunca diminuíam (quando tinha de menos).
    Agora faz busca binária: se resultado < target, diminui o voxel.
    """
    t0 = time.time()

    flange_mask = labels == 1
    equip_mask = labels == 0
    n_flange = int(flange_mask.sum())
    n_equip = int(equip_mask.sum())
    n_total = len(labels)

    target_equip = int(n_flange * equip_flange_ratio)
    target_equip = max(target_equip, min_equip)
    target_equip = min(target_equip, max_equip)

    # Se target >= n_equip, não precisa downsample
    if n_equip <= target_equip:
        print(f"    Equipamento ({n_equip:,}) já <= target ({target_equip:,}), sem downsample.")
        elapsed = time.time() - t0
        flange_pct = 100 * n_flange / n_total
        print(f"    % flange: {flange_pct:.1f}% (sem mudança)")
        print(f"    Downsample: {elapsed:.1f}s")
        return xyz, labels, rgb, intensity

    print(f"    Downsample sensível à classe...")
    print(f"    Total: {n_total:,} (flange: {n_flange:,}, equip: {n_equip:,})")
    print(f"    Target equip: {target_equip:,} (ratio={equip_flange_ratio}x flange)")

    # Separar classes
    f_xyz = xyz[flange_mask]
    f_rgb = rgb[flange_mask] if rgb is not None else None
    f_int = intensity[flange_mask] if intensity is not None else None

    e_xyz = xyz[equip_mask]
    e_rgb = rgb[equip_mask] if rgb is not None else None
    e_int = intensity[equip_mask] if intensity is not None else None

    # Estimar voxel inicial
    bbox_range = e_xyz.max(axis=0) - e_xyz.min(axis=0)
    bbox_volume = np.prod(bbox_range + 1e-8)
    volume_per_voxel = bbox_volume / max(target_equip, 1)
    estimated_voxel = max(volume_per_voxel ** (1.0 / 3.0), initial_voxel_size)

    print(f"    Equip bbox: {bbox_range[0]:.2f} x {bbox_range[1]:.2f} x {bbox_range[2]:.2f}")
    print(f"    Voxel estimado: {estimated_voxel:.6f}")

    # ── BUSCA BIDIRECIONAL ──
    # Limites para busca binária
    voxel_low = estimated_voxel * 0.001   # voxel mínimo (muitos pontos)
    voxel_high = estimated_voxel * 100.0  # voxel máximo (poucos pontos)
    voxel_size = estimated_voxel

    best_result = None
    best_voxel = None
    best_n = 0
    min_acceptable = int(target_equip * tolerance_low)

    for attempt in range(max_iter):
        ds_xyz, ds_rgb, ds_int, n_result = _voxel_downsample_arrays(
            e_xyz, e_rgb, e_int, voxel_size
        )

        ratio_achieved = n_result / max(target_equip, 1)
        print(f"    Tentativa {attempt+1}: voxel={voxel_size:.6f} → {n_result:,} pontos "
              f"({100*ratio_achieved:.0f}% do target)")

        # Guardar melhor resultado que está dentro da faixa aceitável
        if min_acceptable <= n_result <= target_equip:
            if n_result > best_n:
                best_result = (ds_xyz, ds_rgb, ds_int, n_result)
                best_voxel = voxel_size
                best_n = n_result
            # Se está perto o suficiente (>85% do target), para
            if n_result >= target_equip * 0.85:
                break

        # Também guardar se está acima do target mas é o mais próximo
        if n_result > target_equip and (best_result is None or abs(n_result - target_equip) < abs(best_n - target_equip)):
            best_result = (ds_xyz, ds_rgb, ds_int, n_result)
            best_voxel = voxel_size
            best_n = n_result

        # Busca binária
        if n_result < min_acceptable:
            # Poucos pontos demais → diminuir voxel
            voxel_high = voxel_size
            voxel_size = (voxel_low + voxel_size) / 2.0
        elif n_result > target_equip:
            # Muitos pontos → aumentar voxel
            voxel_low = voxel_size
            voxel_size = (voxel_size + voxel_high) / 2.0
        else:
            # Dentro da faixa aceitável, mas vamos tentar melhorar
            if n_result < target_equip * 0.85:
                voxel_high = voxel_size
                voxel_size = (voxel_low + voxel_size) / 2.0
            else:
                break

    # Se nenhum resultado bom, usar o melhor que temos
    if best_result is None:
        print(f"    ⚠ Busca voxel não convergiu, usando random subsample")
        ds_xyz, ds_rgb, ds_int, n_result = _random_subsample_arrays(
            e_xyz, e_rgb, e_int, target_equip
        )
    else:
        ds_xyz, ds_rgb, ds_int, n_result = best_result
        print(f"    ✓ Melhor voxel: {best_voxel:.6f} → {best_n:,} pontos")

    n_equip_final = len(ds_xyz)

    # Juntar
    final_xyz = np.concatenate([f_xyz, ds_xyz], axis=0).astype(np.float32)
    final_labels = np.concatenate([
        np.ones(n_flange, dtype=np.int64),
        np.zeros(n_equip_final, dtype=np.int64)
    ])
    final_rgb = None
    if rgb is not None:
        final_rgb = np.concatenate([f_rgb, ds_rgb], axis=0).astype(np.float32)
    final_intensity = None
    if intensity is not None:
        final_intensity = np.concatenate([f_int, ds_int], axis=0).astype(np.float32)

    n_final = len(final_labels)
    flange_pct_before = 100 * n_flange / max(n_total, 1)
    flange_pct_after = 100 * n_flange / max(n_final, 1)
    elapsed = time.time() - t0

    print(f"    ── Resultado ──")
    print(f"    Flange:      {n_flange:,} → {n_flange:,} (100% preservado)")
    print(f"    Equipamento: {n_equip:,} → {n_equip_final:,} ({100*n_equip_final/max(n_equip,1):.1f}%)")
    print(f"    Total:       {n_total:,} → {n_final:,} ({100*n_final/max(n_total,1):.1f}%)")
    print(f"    % flange:    {flange_pct_before:.1f}% → {flange_pct_after:.1f}%")
    if flange_pct_after > 35:
        print(f"    ⚠ AVISO: flange > 35%")
    print(f"    Downsample:  {elapsed:.1f}s")

    return final_xyz, final_labels, final_rgb, final_intensity


# ============================================================
# PREPROCESS
# ============================================================

def preprocess_single_file(filepath, output_dir):
    basename = os.path.splitext(os.path.basename(filepath))[0]
    output_path = os.path.join(output_dir, f"{basename}.npz")

    if os.path.exists(output_path):
        print(f"  [SKIP] {basename}.npz já existe")
        return output_path

    t0 = time.time()
    print(f"\n  Processando: {basename}")

    print("    Lendo PLY...")
    t_read = time.time()
    xyz, labels, rgb, intensity = read_ply_with_features(filepath)
    n_original = len(xyz)
    n_flange_orig = int(np.sum(labels == 1))
    print(f"    Pontos originais: {n_original:,}")
    print(f"    Flange original: {n_flange_orig:,} ({100*n_flange_orig/n_original:.1f}%)")
    print(f"    Leitura: {time.time()-t_read:.1f}s")

    xyz, labels, rgb, intensity = class_aware_downsample(
        xyz, labels, rgb, intensity,
        equip_flange_ratio=EQUIP_FLANGE_RATIO,
        min_equip=MIN_EQUIP_POINTS,
        max_equip=MAX_EQUIP_POINTS,
        initial_voxel_size=VOXEL_SIZE_INITIAL,
        tolerance_low=DOWNSAMPLE_TOLERANCE_LOW,
        max_iter=DOWNSAMPLE_MAX_ITER,
    )

    # Centralizar
    mean_xyz = np.mean(xyz, axis=0, keepdims=True)
    min_z = np.min(xyz[:, 2:3], axis=0, keepdims=True)
    xyz_centered = xyz.copy()
    xyz_centered[:, 0] -= mean_xyz[0, 0]
    xyz_centered[:, 1] -= mean_xyz[0, 1]
    xyz_centered[:, 2] -= min_z[0, 0]

    print("    Calculando normais...")
    t_n = time.time()
    normals = estimate_normals_open3d(xyz_centered, k=KNN_NORMALS)
    print(f"    Normais: {time.time()-t_n:.1f}s")

    print("    Calculando curvatura...")
    t_c = time.time()
    curvature = compute_curvature_vectorized(xyz_centered, k=KNN_NORMALS, block_size=CURVATURE_BLOCK_SIZE)
    print(f"    Curvatura: {time.time()-t_c:.1f}s")

    print("    Montando features...")
    features = build_features(xyz_centered, normals, curvature, rgb, intensity)

    if SHUFFLE_BEFORE_SAVE:
        rng = np.random.default_rng(42)
        perm = rng.permutation(len(labels))
        features = features[perm]
        labels = labels[perm]
        xyz_centered = xyz_centered[perm]

    flange_idx = np.where(labels == 1)[0]
    equip_idx = np.where(labels == 0)[0]

    save_dict = {
        "features": features.astype(np.float32),
        "labels": labels.astype(np.int64),
        "flange_idx": flange_idx.astype(np.int64),
        "equip_idx": equip_idx.astype(np.int64),
    }
    if SAVE_XYZ_CENTERED:
        save_dict["xyz_centered"] = xyz_centered.astype(np.float32)

    np.savez_compressed(output_path, **save_dict)

    file_size_mb = os.path.getsize(output_path) / (1024 * 1024)
    elapsed = time.time() - t0
    print(f"    Salvo: {output_path}")
    print(f"    Tamanho: {file_size_mb:.1f} MB")
    print(f"    Tempo TOTAL: {elapsed:.1f}s ({elapsed/60:.1f} min)")
    print(f"    Features shape: {features.shape}")
    print(f"    Flange: {len(flange_idx):,}, Equipamento: {len(equip_idx):,}")

    return output_path


def main():
    t_start = time.time()

    ply_files = sorted(glob(os.path.join(DATA_DIR, "*.ply")))
    if not ply_files:
        print(f"ERRO: nenhum PLY encontrado em {DATA_DIR}")
        sys.exit(1)

    os.makedirs(PREPROCESSED_DIR, exist_ok=True)

    print("=" * 70)
    print("PRÉ-PROCESSAMENTO v5 (VOXEL BIDIRECIONAL)")
    print("=" * 70)
    print(f"Entrada:               {DATA_DIR}")
    print(f"Saída:                 {PREPROCESSED_DIR}")
    print(f"Arquivos PLY:          {len(ply_files)}")
    print(f"Equip/Flange ratio:    {EQUIP_FLANGE_RATIO}x")
    print(f"Min equip points:      {MIN_EQUIP_POINTS:,}")
    print(f"Max equip points:      {MAX_EQUIP_POINTS:,}")
    print(f"Tolerância baixa:      {DOWNSAMPLE_TOLERANCE_LOW:.0%}")
    print(f"KNN normais/curvatura: {KNN_NORMALS}")
    print("=" * 70)

    success = 0
    errors = []

    for i, filepath in enumerate(ply_files, start=1):
        print(f"\n{'─'*70}")
        print(f"[{i}/{len(ply_files)}] {os.path.basename(filepath)}")
        try:
            preprocess_single_file(filepath, PREPROCESSED_DIR)
            success += 1
        except Exception as e:
            print(f"  ERRO: {e}")
            errors.append((os.path.basename(filepath), str(e)))
            import traceback
            traceback.print_exc()

    # Verificação global
    print(f"\n{'='*70}")
    print("VERIFICAÇÃO GLOBAL")
    print(f"{'='*70}")

    total_flange = 0
    total_equip = 0
    npz_files = sorted(glob(os.path.join(PREPROCESSED_DIR, "*.npz")))
    for npz_path in npz_files:
        data = np.load(npz_path)
        labels = data["labels"]
        n_f = int(np.sum(labels == 1))
        n_e = int(np.sum(labels == 0))
        total_flange += n_f
        total_equip += n_e
        pct = 100 * n_f / max(n_f + n_e, 1)
        warn = "  ⚠" if pct > 35 else "  ✓"
        print(f"  {os.path.basename(npz_path):30s} flange={n_f:>9,} equip={n_e:>9,} ({pct:5.1f}% flange){warn}")

    total = total_flange + total_equip
    global_pct = 100 * total_flange / max(total, 1)
    print(f"\n  GLOBAL: flange={total_flange:,} equip={total_equip:,} ({global_pct:.1f}% flange)")

    if global_pct > 30:
        print("  ⚠ AVISO: base ainda flange-heavy.")
    else:
        print("  ✓ Balanço OK!")

    t_total = time.time() - t_start
    print(f"\n{'='*70}")
    print("RESUMO")
    print(f"{'='*70}")
    print(f"Processados: {success}/{len(ply_files)}")
    if errors:
        print(f"Erros ({len(errors)}):")
        for name, err in errors:
            print(f"  {name}: {err}")
    print(f"Tempo total: {t_total/60:.1f} min")
    print(f"Arquivos em: {PREPROCESSED_DIR}")


if __name__ == "__main__":
    main()
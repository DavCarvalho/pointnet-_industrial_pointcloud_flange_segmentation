"""
PRÉ-PROCESSAMENTO DE NUVENS DE PONTOS INDUSTRIAIS
===================================================
Calcula normais, curvatura e features UMA VEZ para cada PLY,
salvando como .npz. O treinamento depois carrega instantaneamente.

Uso:
    python preprocess_pointclouds.py

Gera arquivos .npz no diretório PREPROCESSED_DIR com:
    - features: (N, 11) -> xyz(3) + normals(3) + curvature(1) + rgb(3) + intensity(1)
    - labels: (N,) -> 0=equipamento, 1=flange
    - flange_idx: índices dos pontos de flange
    - equip_idx: índices dos pontos de equipamento
"""

import os
import sys
import time
import numpy as np
from glob import glob
from plyfile import PlyData

# ╔══════════════════════════════════════════════════════════════╗
# ║                      CONFIGURAÇÃO                           ║
# ╚══════════════════════════════════════════════════════════════╝

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Diretório com os PLY originais
DATA_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, "../../DATA"))

# Diretório de saída para os .npz pré-processados
PREPROCESSED_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, "../../DATA_PREPROCESSED"))

# Vizinhos para normais e curvatura
KNN_NORMALS = 16

# Voxel downsample antes de calcular curvatura (None = sem downsample)
# Se suas nuvens são muito grandes (>5M pontos), use 0.02-0.05
# Se quiser manter resolução original, use None
VOXEL_SIZE = 0.03  # Ex: 0.02 para reduzir tamanho

USE_RGB = True
USE_INTENSITY = True

# Tamanho do bloco para cálculo vetorizado de curvatura
CURVATURE_BLOCK_SIZE = 50000


# ╔══════════════════════════════════════════════════════════════╗
# ║                   LEITURA DE DADOS                           ║
# ╚══════════════════════════════════════════════════════════════╝

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
    """Lê PLY e retorna coords, labels, rgb, intensity."""
    plydata = PlyData.read(filepath)
    vertex = plydata["vertex"].data
    prop_names = [p.name for p in plydata["vertex"].properties]

    # Coordenadas XYZ
    x = np.array(vertex["x"], dtype=np.float32)
    y = np.array(vertex["y"], dtype=np.float32)
    z = np.array(vertex["z"], dtype=np.float32)
    xyz = np.column_stack([x, y, z])

    # Labels
    class_field = find_class_field(plydata["vertex"].properties)
    labels = np.array(vertex[class_field], dtype=np.int32)
    # Binarizar: 1=flange, resto=0
    labels = np.where(labels == 1, 1, 0).astype(np.int64)

    # RGB
    rgb = None
    if USE_RGB and all(c in prop_names for c in ["red", "green", "blue"]):
        r = np.array(vertex["red"], dtype=np.float32)
        g = np.array(vertex["green"], dtype=np.float32)
        b = np.array(vertex["blue"], dtype=np.float32)
        rgb = np.column_stack([r, g, b])

    # Intensidade
    intensity = None
    if USE_INTENSITY and "intensity" in prop_names:
        intensity = np.array(vertex["intensity"], dtype=np.float32).reshape(-1, 1)

    return xyz, labels, rgb, intensity


# ╔══════════════════════════════════════════════════════════════╗
# ║              NORMAIS E CURVATURA (OTIMIZADOS)                ║
# ╚══════════════════════════════════════════════════════════════╝

def estimate_normals_open3d(xyz, k=16):
    """Estima normais via Open3D (rápido, C++ interno)."""
    import open3d as o3d

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz.astype(np.float64))

    pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamKNN(knn=k)
    )

    try:
        pcd.orient_normals_consistent_tangent_plane(k=k)
    except RuntimeError:
        pass

    normals = np.asarray(pcd.normals, dtype=np.float32)
    return normals


def compute_curvature_vectorized(xyz, k=16, block_size=50000):
    """
    Calcula curvatura local de forma VETORIZADA.
    ~10-50x mais rápido que o loop original.
    """
    from sklearn.neighbors import KDTree

    N = xyz.shape[0]
    curvature = np.zeros(N, dtype=np.float32)

    print(f"    Construindo KDTree para {N:,} pontos...")
    tree = KDTree(xyz)

    n_blocks = (N + block_size - 1) // block_size
    for b_idx in range(n_blocks):
        start = b_idx * block_size
        end = min(start + block_size, N)
        actual_size = end - start

        if b_idx % 5 == 0 or b_idx == n_blocks - 1:
            print(f"    Curvatura: bloco {b_idx+1}/{n_blocks} "
                  f"({100*end/N:.0f}%)")

        # Query KNN para o bloco inteiro
        idx = tree.query(xyz[start:end], k=k, return_distance=False)  # (block, k)

        # Pegar vizinhos de uma vez: (block, k, 3)
        neighbors = xyz[idx]

        # Centralizar: (block, k, 3)
        centered = neighbors - neighbors.mean(axis=1, keepdims=True)

        # Covariância vetorizada: (block, 3, 3) via einsum
        cov = np.einsum('bki,bkj->bij', centered, centered)

        # Eigenvalues em batch: (block, 3)
        eigvals = np.linalg.eigvalsh(cov)

        # Curvatura = menor eigenvalue / soma
        curvature[start:end] = eigvals[:, 0] / (eigvals.sum(axis=1) + 1e-8)

    return curvature.reshape(-1, 1)


# ╔══════════════════════════════════════════════════════════════╗
# ║                   NORMALIZAÇÃO                               ║
# ╚══════════════════════════════════════════════════════════════╝

def normalize_xyz_unit_sphere(xyz):
    """Centraliza e escala para esfera unitária."""
    xyz = xyz.astype(np.float32)
    center = np.mean(xyz, axis=0, keepdims=True)
    xyz = xyz - center
    radius = np.max(np.linalg.norm(xyz, axis=1))
    if radius < 1e-8:
        radius = 1.0
    return (xyz / radius).astype(np.float32)


def normalize_scalar_channel(x):
    """Z-score normalization."""
    x = x.astype(np.float32)
    mean = np.mean(x, axis=0, keepdims=True)
    std = np.std(x, axis=0, keepdims=True) + 1e-8
    return ((x - mean) / std).astype(np.float32)


def build_features(xyz_centered, normals, curvature, rgb, intensity):
    """
    Monta tensor de features: (N, 11)
    [xyz_norm(3), normals(3), curvature(1), rgb(3), intensity(1)]
    """
    # XYZ normalizado (esfera unitária)
    xyz_geom = normalize_xyz_unit_sphere(xyz_centered)

    # Normais: garantir módulo ~1
    normals = normals.astype(np.float32)
    normals_norm = np.linalg.norm(normals, axis=1, keepdims=True) + 1e-8
    normals = normals / normals_norm

    # Curvatura normalizada
    curvature = normalize_scalar_channel(curvature)

    # RGB
    if rgb is None:
        rgb = np.zeros((len(xyz_centered), 3), dtype=np.float32)
    else:
        rgb = rgb.astype(np.float32)
        if np.max(rgb) > 1.5:
            rgb = rgb / 255.0

    # Intensidade
    if intensity is None:
        intensity = np.zeros((len(xyz_centered), 1), dtype=np.float32)
    else:
        intensity = normalize_scalar_channel(intensity)

    features = np.concatenate(
        [xyz_geom, normals, curvature, rgb, intensity],
        axis=1
    ).astype(np.float32)

    return features


# ╔══════════════════════════════════════════════════════════════╗
# ║              VOXEL DOWNSAMPLE (OPCIONAL)                     ║
# ╚══════════════════════════════════════════════════════════════╝

def voxel_downsample_with_labels(xyz, labels, rgb, intensity, voxel_size):
    """
    Voxel downsample que preserva labels por voto majoritário.
    Retorna arrays reduzidos.
    """
    if voxel_size is None or voxel_size <= 0:
        return xyz, labels, rgb, intensity

    print(f"    Voxel downsample (voxel={voxel_size})...")
    print(f"    Pontos antes: {len(xyz):,}")

    # Quantizar coordenadas em voxels
    voxel_coords = np.floor(xyz / voxel_size).astype(np.int64)

    # Hash para agrupar pontos no mesmo voxel
    # Usar string hash para evitar colisões
    voxel_keys = (
        voxel_coords[:, 0].astype(str) + "_" +
        voxel_coords[:, 1].astype(str) + "_" +
        voxel_coords[:, 2].astype(str)
    )

    unique_keys, inverse = np.unique(voxel_keys, return_inverse=True)
    n_voxels = len(unique_keys)

    new_xyz = np.zeros((n_voxels, 3), dtype=np.float32)
    new_labels = np.zeros(n_voxels, dtype=np.int64)
    new_rgb = np.zeros((n_voxels, 3), dtype=np.float32) if rgb is not None else None
    new_intensity = np.zeros((n_voxels, 1), dtype=np.float32) if intensity is not None else None

    for i in range(n_voxels):
        mask = inverse == i
        new_xyz[i] = xyz[mask].mean(axis=0)

        # Label por voto majoritário
        voxel_labels = labels[mask]
        # Se qualquer ponto é flange, marca como flange (conservador)
        new_labels[i] = 1 if np.any(voxel_labels == 1) else 0

        if rgb is not None:
            new_rgb[i] = rgb[mask].mean(axis=0)
        if intensity is not None:
            new_intensity[i] = intensity[mask].mean(axis=0)

    print(f"    Pontos depois: {n_voxels:,} ({100*n_voxels/len(xyz):.1f}%)")

    return new_xyz, new_labels, new_rgb, new_intensity


# ╔══════════════════════════════════════════════════════════════╗
# ║                      MAIN                                   ║
# ╚══════════════════════════════════════════════════════════════╝

def preprocess_single_file(filepath, output_dir):
    """Pré-processa um único arquivo PLY e salva como .npz."""
    basename = os.path.splitext(os.path.basename(filepath))[0]
    output_path = os.path.join(output_dir, f"{basename}.npz")

    # Pular se já existe
    if os.path.exists(output_path):
        print(f"  [SKIP] {basename}.npz já existe")
        return output_path

    t0 = time.time()
    print(f"\n  Processando: {basename}")

    # 1. Ler PLY
    print(f"    Lendo PLY...")
    xyz, labels, rgb, intensity = read_ply_with_features(filepath)
    n_original = len(xyz)
    print(f"    Pontos originais: {n_original:,}")
    print(f"    Flange: {np.sum(labels==1):,} ({100*np.mean(labels==1):.1f}%)")

    # 2. Voxel downsample (opcional)
    if VOXEL_SIZE is not None:
        xyz, labels, rgb, intensity = voxel_downsample_with_labels(
            xyz, labels, rgb, intensity, VOXEL_SIZE
        )

    # 3. Centralizar coordenadas
    mean_xyz = np.mean(xyz, axis=0, keepdims=True)
    min_z = np.min(xyz[:, 2:3], axis=0, keepdims=True)
    xyz_centered = xyz.copy()
    xyz_centered[:, 0] -= mean_xyz[0, 0]
    xyz_centered[:, 1] -= mean_xyz[0, 1]
    xyz_centered[:, 2] -= min_z[0, 0]

    # 4. Calcular normais (Open3D - rápido)
    print(f"    Calculando normais...")
    t_normals = time.time()
    normals = estimate_normals_open3d(xyz_centered, k=KNN_NORMALS)
    print(f"    Normais: {time.time()-t_normals:.1f}s")

    # 5. Calcular curvatura (VETORIZADO - muito mais rápido)
    print(f"    Calculando curvatura (vetorizado)...")
    t_curv = time.time()
    curvature = compute_curvature_vectorized(
        xyz_centered, k=KNN_NORMALS, block_size=CURVATURE_BLOCK_SIZE
    )
    print(f"    Curvatura: {time.time()-t_curv:.1f}s")

    # 6. Montar features
    print(f"    Montando features...")
    features = build_features(xyz_centered, normals, curvature, rgb, intensity)

    # 7. Índices por classe
    flange_idx = np.where(labels == 1)[0]
    equip_idx = np.where(labels != 1)[0]

    # 8. Salvar
    np.savez_compressed(
        output_path,
        features=features,
        labels=labels,
        flange_idx=flange_idx,
        equip_idx=equip_idx
    )

    file_size_mb = os.path.getsize(output_path) / (1024 * 1024)
    elapsed = time.time() - t0
    print(f"    Salvo: {output_path}")
    print(f"    Tamanho: {file_size_mb:.1f} MB")
    print(f"    Tempo: {elapsed:.1f}s ({elapsed/60:.1f} min)")
    print(f"    Features shape: {features.shape}")
    print(f"    Flange: {len(flange_idx):,}, Equipamento: {len(equip_idx):,}")

    return output_path


def main():
    t_start = time.time()

    # Encontrar todos os PLY
    ply_files = sorted(glob(os.path.join(DATA_DIR, "*.ply")))
    if not ply_files:
        print(f"ERRO: Nenhum arquivo PLY encontrado em {DATA_DIR}")
        sys.exit(1)

    print(f"{'='*60}")
    print(f"PRÉ-PROCESSAMENTO DE NUVENS DE PONTOS")
    print(f"{'='*60}")
    print(f"Diretório de entrada: {DATA_DIR}")
    print(f"Diretório de saída:   {PREPROCESSED_DIR}")
    print(f"Arquivos PLY:         {len(ply_files)}")
    print(f"Voxel size:           {VOXEL_SIZE}")
    print(f"KNN normais:          {KNN_NORMALS}")
    print(f"Block size curvatura: {CURVATURE_BLOCK_SIZE:,}")

    # Criar diretório de saída
    os.makedirs(PREPROCESSED_DIR, exist_ok=True)

    # Processar cada arquivo
    success = 0
    errors = []

    for i, filepath in enumerate(ply_files):
        print(f"\n[{i+1}/{len(ply_files)}] {os.path.basename(filepath)}")
        try:
            preprocess_single_file(filepath, PREPROCESSED_DIR)
            success += 1
        except Exception as e:
            print(f"  ERRO: {e}")
            errors.append((os.path.basename(filepath), str(e)))
            import traceback
            traceback.print_exc()

    # Resumo
    t_total = time.time() - t_start
    print(f"\n{'='*60}")
    print(f"RESUMO")
    print(f"{'='*60}")
    print(f"Processados com sucesso: {success}/{len(ply_files)}")
    if errors:
        print(f"Erros ({len(errors)}):")
        for name, err in errors:
            print(f"  {name}: {err}")
    print(f"Tempo total: {t_total/60:.1f} minutos")
    print(f"Arquivos salvos em: {PREPROCESSED_DIR}")


if __name__ == "__main__":
    main()
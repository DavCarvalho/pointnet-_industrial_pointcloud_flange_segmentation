import os
import hashlib
import numpy as np
import pandas as pd
from plyfile import PlyData, PlyElement
import time

# =========================
# CONFIG
# =========================
DATA_DIR = "./DATA"
OUT_DIR = "./FLANGE_DATASET_labels2"

# Estratégia híbrida:
# - flange: mantém todos os pontos
# - non-flange: voxel downsampling
MAX_FLANGE_POINTS = None
MAX_NON_FLANGE_POINTS = 400_000

CLASS_NAMES_CANDIDATES = [
    "scalar_Classification", "scalar_classification",
    "Classification", "classification",
    "label", "scalar_Label", "class"
]


# =========================
# HELPERS
# =========================
def stable_seed(text: str) -> int:
    return int(hashlib.md5(text.encode("utf-8")).hexdigest()[:8], 16)


def deterministic_random_subsample(points: np.ndarray, max_points: int, key: str) -> np.ndarray:
    """
    Subamostragem aleatória, porém determinística por equipamento/chave.
    Boa para flange, preservando a distribuição natural de densidade.
    """
    if max_points is None or len(points) <= max_points:
        return points

    rng = np.random.default_rng(stable_seed(key))
    idx = rng.choice(len(points), size=max_points, replace=False)
    return points[idx]


def voxel_downsample(coords: np.ndarray, max_points: int) -> np.ndarray:
    """
    Reduz nuvem para ~max_points via voxel grid.
    Boa para non-flange, preservando cobertura espacial.
    """
    n = len(coords)
    if max_points is None or n <= max_points:
        return coords

    mins = coords.min(axis=0)
    bbox = coords.max(axis=0) - mins
    volume = np.prod(bbox + 1e-6)

    # chute inicial para voxel size
    voxel_size = (volume / max_points) ** (1.0 / 3.0)

    while True:
        voxel_idx = ((coords - mins) / voxel_size).astype(np.int64)
        mx = voxel_idx.max(axis=0) + 1

        # hash 3D -> 1D
        hashes = voxel_idx[:, 0] + voxel_idx[:, 1] * mx[0] + voxel_idx[:, 2] * mx[0] * mx[1]

        _, unique_idx = np.unique(hashes, return_index=True)
        result = coords[unique_idx]

        # margem pequena para não ficar iterando demais
        if len(result) <= max_points * 1.15:
            return result

        voxel_size *= 1.25


def find_class_field(vertex_properties):
    prop_names = [p.name for p in vertex_properties]
    for name in CLASS_NAMES_CANDIDATES:
        if name in prop_names:
            return name
    raise ValueError(f"Campo de classificação não encontrado. Campos disponíveis: {prop_names}")


def save_points_as_ply(points: np.ndarray, out_path: str):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    vertex_array = np.array(
        [(float(p[0]), float(p[1]), float(p[2])) for p in points],
        dtype=[("x", "f4"), ("y", "f4"), ("z", "f4")]
    )
    el = PlyElement.describe(vertex_array, "vertex")
    PlyData([el], text=True).write(out_path)


# =========================
# MAIN
# =========================
def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(os.path.join(OUT_DIR, "flanges"), exist_ok=True)
    os.makedirs(os.path.join(OUT_DIR, "non_flanges"), exist_ok=True)

    ply_files = sorted([f for f in os.listdir(DATA_DIR) if f.lower().endswith(".ply")])
    print(f"Encontrados {len(ply_files)} arquivos PLY\n")

    rows = []
    t_total = time.time()

    for fname in ply_files:
        t0 = time.time()

        file_path = os.path.join(DATA_DIR, fname)
        equipment_name = os.path.splitext(fname)[0]

        plydata = PlyData.read(file_path)
        vertex = plydata["vertex"].data
        class_field = find_class_field(plydata["vertex"].properties)

        coords = np.column_stack([
            np.array(vertex["x"], dtype=np.float32),
            np.array(vertex["y"], dtype=np.float32),
            np.array(vertex["z"], dtype=np.float32),
        ])
        labels = np.array(vertex[class_field])

        raw_flange_points = coords[labels == 1]
        raw_non_flange_points = coords[labels != 1]

        # flange -> mantém tudo
        flange_points = deterministic_random_subsample(
            raw_flange_points,
            MAX_FLANGE_POINTS,
            f"{equipment_name}_flange"
        )

        # non-flange -> voxel downsampling
        non_flange_points = voxel_downsample(
            raw_non_flange_points,
            MAX_NON_FLANGE_POINTS
        )

        f_tag = f" -> {len(flange_points):,}" if len(flange_points) < len(raw_flange_points) else ""
        n_tag = f" -> {len(non_flange_points):,}" if len(non_flange_points) < len(raw_non_flange_points) else ""

        print(
            f"{equipment_name}: "
            f"{len(raw_flange_points):,} flange{f_tag} + "
            f"{len(raw_non_flange_points):,} equip{n_tag}"
        )

        # arquivo positivo
        if len(flange_points) > 0:
            flange_rel = os.path.join("flanges", f"{equipment_name}_flanges.ply")
            flange_abs = os.path.join(OUT_DIR, flange_rel)
            save_points_as_ply(flange_points, flange_abs)

            centroid = flange_points.mean(axis=0)
            rows.append({
                "Equipamento": equipment_name,
                "Flange_ID": f"{equipment_name}_flanges",
                "Num_Pontos": int(len(flange_points)),
                "Centroid_X": float(centroid[0]),
                "Centroid_Y": float(centroid[1]),
                "Centroid_Z": float(centroid[2]),
                "Arquivo_PLY": flange_rel,
                "Label": 1,
            })

        # arquivo negativo
        if len(non_flange_points) > 0:
            non_flange_rel = os.path.join("non_flanges", f"{equipment_name}_equipamento.ply")
            non_flange_abs = os.path.join(OUT_DIR, non_flange_rel)
            save_points_as_ply(non_flange_points, non_flange_abs)

            centroid = non_flange_points.mean(axis=0)
            rows.append({
                "Equipamento": equipment_name,
                "Flange_ID": f"{equipment_name}_equipamento",
                "Num_Pontos": int(len(non_flange_points)),
                "Centroid_X": float(centroid[0]),
                "Centroid_Y": float(centroid[1]),
                "Centroid_Z": float(centroid[2]),
                "Arquivo_PLY": non_flange_rel,
                "Label": 0,
            })

        dt = time.time() - t0
        print(f"  tempo: {dt:.1f}s\n")

    df = pd.DataFrame(rows)
    csv_path = os.path.join(OUT_DIR, "flange_info_labels2.csv")
    df.to_csv(csv_path, index=False)

    total_dt = time.time() - t_total

    print("=" * 60)
    print("DATASET GERADO")
    print("=" * 60)
    print(f"Tempo total: {total_dt:.1f}s")
    print(f"CSV: {csv_path}")
    print(f"Arquivos: {len(df)}")
    print("\nDistribuição:")
    print(df["Label"].value_counts().to_string())
    print(f"\nEquipamentos: {df['Equipamento'].nunique()}")


if __name__ == "__main__":
    main()
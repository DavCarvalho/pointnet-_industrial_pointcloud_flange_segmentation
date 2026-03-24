import os
import hashlib
import numpy as np
import pandas as pd
from plyfile import PlyData, PlyElement

DATA_DIR = "./DATA"
OUT_DIR = "./FLANGE_DATASET_labels"

CLASS_NAMES_CANDIDATES = [
    "scalar_Classification", "scalar_classification",
    "Classification", "classification",
    "label", "scalar_Label", "class"
]

MAX_NON_FLANGE_POINTS = 300_000
MAX_FLANGE_POINTS = None   # pode deixar None


def stable_seed(text):
    return int(hashlib.md5(text.encode("utf-8")).hexdigest()[:8], 16)


def deterministic_subsample(points, max_points, key):
    if max_points is None or len(points) <= max_points:
        return points

    rng = np.random.default_rng(stable_seed(key))
    idx = rng.choice(len(points), size=max_points, replace=False)
    return points[idx]


def find_class_field(vertex_properties):
    prop_names = [p.name for p in vertex_properties]
    for name in CLASS_NAMES_CANDIDATES:
        if name in prop_names:
            return name
    raise ValueError(f"Nenhum campo de classificação encontrado. Campos disponíveis: {prop_names}")


def save_points_as_ply(points, out_path):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    vertex_array = np.array(
        [(float(p[0]), float(p[1]), float(p[2])) for p in points],
        dtype=[("x", "f4"), ("y", "f4"), ("z", "f4")]
    )
    el = PlyElement.describe(vertex_array, "vertex")
    PlyData([el], text=True).write(out_path)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(os.path.join(OUT_DIR, "flanges"), exist_ok=True)
    os.makedirs(os.path.join(OUT_DIR, "non_flanges"), exist_ok=True)

    ply_files = sorted([f for f in os.listdir(DATA_DIR) if f.lower().endswith(".ply")])
    print(f"Encontrados {len(ply_files)} arquivos PLY\n")

    rows = []

    for fname in ply_files:
        file_path = os.path.join(DATA_DIR, fname)
        equipment_name = os.path.splitext(fname)[0]

        plydata = PlyData.read(file_path)
        vertex = plydata["vertex"].data
        class_field = find_class_field(plydata["vertex"].properties)

        coords = np.column_stack([
            np.array(vertex["x"], dtype=np.float64),
            np.array(vertex["y"], dtype=np.float64),
            np.array(vertex["z"], dtype=np.float64),
        ])
        labels = np.array(vertex[class_field])

        raw_flange_points = coords[labels == 1]
        raw_non_flange_points = coords[labels != 1]

        flange_points = deterministic_subsample(
            raw_flange_points,
            MAX_FLANGE_POINTS,
            f"{equipment_name}_flange"
        )

        non_flange_points = deterministic_subsample(
            raw_non_flange_points,
            MAX_NON_FLANGE_POINTS,
            f"{equipment_name}_non_flange"
        )

        print(
            f"{equipment_name}: "
            f"{len(raw_flange_points)} flange -> {len(flange_points)} salvo | "
            f"{len(raw_non_flange_points)} equipamento -> {len(non_flange_points)} salvo"
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
                "Num_Pontos": len(flange_points),
                "Centroid_X": centroid[0],
                "Centroid_Y": centroid[1],
                "Centroid_Z": centroid[2],
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
                "Num_Pontos": len(non_flange_points),
                "Centroid_X": centroid[0],
                "Centroid_Y": centroid[1],
                "Centroid_Z": centroid[2],
                "Arquivo_PLY": non_flange_rel,
                "Label": 0,
            })

    df = pd.DataFrame(rows)
    csv_path = os.path.join(OUT_DIR, "flange_info_labels.csv")
    df.to_csv(csv_path, index=False)

    print("\n==============================")
    print("DATASET GERADO")
    print("==============================")
    print(f"CSV: {csv_path}")
    print(f"Total de arquivos-amostra: {len(df)}")
    print("\nDistribuição por classe:")
    print(df["Label"].value_counts())
    print("\nEquipamentos únicos:")
    print(df["Equipamento"].nunique())


if __name__ == "__main__":
    main()
import os
import hashlib
import argparse
import numpy as np
import pandas as pd
from plyfile import PlyData, PlyElement
from sklearn.neighbors import KDTree

CLASS_FIELD_CANDIDATES = [
    "scalar_Classification", "scalar_classification",
    "Classification", "classification",
    "label", "scalar_Label", "class"
]


def stable_seed(text: str) -> int:
    return int(hashlib.md5(text.encode("utf-8")).hexdigest()[:8], 16)


def make_rng(*parts) -> np.random.Generator:
    key = "|".join(str(p) for p in parts)
    return np.random.default_rng(stable_seed(key))


def find_class_field(vertex_properties):
    prop_names = [p.name for p in vertex_properties]
    for name in CLASS_FIELD_CANDIDATES:
        if name in prop_names:
            return name
    raise ValueError(f"Campo de classificação não encontrado. Campos disponíveis: {prop_names}")


def read_ply_xyz_labels(path: str):
    plydata = PlyData.read(path)
    vertex = plydata["vertex"].data
    class_field = find_class_field(plydata["vertex"].properties)

    xyz = np.column_stack([
        np.asarray(vertex["x"], dtype=np.float64),
        np.asarray(vertex["y"], dtype=np.float64),
        np.asarray(vertex["z"], dtype=np.float64),
    ])
    raw_labels = np.asarray(vertex[class_field], dtype=np.int32)
    labels = np.where(raw_labels == 1, 1, 0).astype(np.int64)
    return xyz, labels


def save_points_as_ply(points: np.ndarray, out_path: str):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    vertex_array = np.array(
        [(float(p[0]), float(p[1]), float(p[2])) for p in points],
        dtype=[("x", "f4"), ("y", "f4"), ("z", "f4")]
    )
    el = PlyElement.describe(vertex_array, "vertex")
    PlyData([el], text=True).write(out_path)


def sample_without_replacement_or_repeat(indices: np.ndarray, n: int, rng: np.random.Generator):
    if len(indices) == 0:
        return np.array([], dtype=np.int64)
    if len(indices) >= n:
        return rng.choice(indices, size=n, replace=False)
    extra = rng.choice(indices, size=(n - len(indices)), replace=True)
    return np.concatenate([indices, extra])


def extract_knn_patch(coords: np.ndarray, tree: KDTree, center_idx: int, num_points: int):
    k = min(num_points, len(coords))
    _, idx = tree.query(coords[center_idx:center_idx + 1], k=k)
    idx = idx[0]
    patch = coords[idx]
    if len(patch) < num_points:
        rng = make_rng("patch_pad", center_idx, len(coords), num_points)
        extra = rng.choice(len(patch), size=(num_points - len(patch)), replace=True)
        patch = np.concatenate([patch, patch[extra]], axis=0)
        idx = np.concatenate([idx, idx[extra]], axis=0)
    return patch, idx


def build_negative_center_pool(
    coords: np.ndarray,
    labels: np.ndarray,
    flange_idx: np.ndarray,
    non_flange_idx: np.ndarray,
    neg_per_equipment: int,
    oversample_factor: int,
    hard_neg_ratio: float,
    hard_neg_max_dist: float,
    easy_neg_min_dist: float,
    seed_key: str,
):
    rng = make_rng(seed_key, "neg_pool")

    if len(non_flange_idx) == 0:
        return np.array([], dtype=np.int64), np.array([], dtype=np.float64)

    n_candidates = min(len(non_flange_idx), max(neg_per_equipment * oversample_factor, neg_per_equipment))
    candidate_idx = rng.choice(non_flange_idx, size=n_candidates, replace=False)

    if len(flange_idx) == 0:
        chosen = sample_without_replacement_or_repeat(candidate_idx, neg_per_equipment, rng)
        return chosen, np.full(len(chosen), np.inf, dtype=np.float64)

    flange_tree = KDTree(coords[flange_idx])
    dists, _ = flange_tree.query(coords[candidate_idx], k=1)
    dists = dists.reshape(-1)

    hard_mask = dists <= hard_neg_max_dist
    easy_mask = dists >= easy_neg_min_dist

    hard_candidates = candidate_idx[hard_mask]
    easy_candidates = candidate_idx[easy_mask]
    hard_dists = dists[hard_mask]
    easy_dists = dists[easy_mask]

    n_hard = int(round(neg_per_equipment * hard_neg_ratio))
    n_easy = neg_per_equipment - n_hard

    chosen_idx = []
    chosen_dists = []

    if len(hard_candidates) > 0 and n_hard > 0:
        take = min(n_hard, len(hard_candidates))
        order = rng.choice(len(hard_candidates), size=take, replace=False)
        chosen_idx.append(hard_candidates[order])
        chosen_dists.append(hard_dists[order])

    if len(easy_candidates) > 0 and n_easy > 0:
        take = min(n_easy, len(easy_candidates))
        order = rng.choice(len(easy_candidates), size=take, replace=False)
        chosen_idx.append(easy_candidates[order])
        chosen_dists.append(easy_dists[order])

    if chosen_idx:
        chosen_idx = np.concatenate(chosen_idx)
        chosen_dists = np.concatenate(chosen_dists)
    else:
        chosen_idx = np.array([], dtype=np.int64)
        chosen_dists = np.array([], dtype=np.float64)

    if len(chosen_idx) < neg_per_equipment:
        already = set(chosen_idx.tolist())
        remaining_mask = np.array([i not in already for i in candidate_idx], dtype=bool)
        remaining_idx = candidate_idx[remaining_mask]
        remaining_dists = dists[remaining_mask]
        need = neg_per_equipment - len(chosen_idx)
        if len(remaining_idx) > 0:
            take = min(need, len(remaining_idx))
            order = rng.choice(len(remaining_idx), size=take, replace=False)
            chosen_idx = np.concatenate([chosen_idx, remaining_idx[order]])
            chosen_dists = np.concatenate([chosen_dists, remaining_dists[order]])

    if len(chosen_idx) < neg_per_equipment:
        filler = sample_without_replacement_or_repeat(non_flange_idx, neg_per_equipment - len(chosen_idx), rng)
        if len(flange_idx) > 0:
            filler_d, _ = flange_tree.query(coords[filler], k=1)
            filler_d = filler_d.reshape(-1)
        else:
            filler_d = np.full(len(filler), np.inf, dtype=np.float64)
        chosen_idx = np.concatenate([chosen_idx, filler])
        chosen_dists = np.concatenate([chosen_dists, filler_d])

    return chosen_idx[:neg_per_equipment], chosen_dists[:neg_per_equipment]


def build_patch_dataset(
    data_dir: str,
    out_dir: str,
    num_points: int,
    pos_per_equipment: int,
    neg_per_equipment: int,
    pos_min_flange_ratio: float,
    neg_max_flange_ratio: float,
    hard_neg_ratio: float,
    hard_neg_max_dist: float,
    easy_neg_min_dist: float,
    oversample_factor: int,
    max_cloud_points: int,
    seed: int,
):
    patch_root = os.path.join(out_dir, "patches")
    pos_root = os.path.join(patch_root, "flange")
    neg_root = os.path.join(patch_root, "non_flange")
    os.makedirs(pos_root, exist_ok=True)
    os.makedirs(neg_root, exist_ok=True)

    ply_files = sorted([f for f in os.listdir(data_dir) if f.lower().endswith(".ply")])
    if not ply_files:
        raise FileNotFoundError(f"Nenhum PLY encontrado em {data_dir}")

    all_rows = []
    summary_rows = []

    global_rng = np.random.default_rng(seed)

    print(f"Arquivos PLY encontrados: {len(ply_files)}")
    print(f"Saída: {out_dir}")
    print(f"NUM_POINTS por patch: {num_points}")
    print(f"Patches positivos por equipamento: {pos_per_equipment}")
    print(f"Patches negativos por equipamento: {neg_per_equipment}\n")

    for file_idx, fname in enumerate(ply_files, start=1):
        path = os.path.join(data_dir, fname)
        equipment_name = os.path.splitext(fname)[0]

        print(f"[{file_idx}/{len(ply_files)}] Lendo {fname}...")
        coords, labels = read_ply_xyz_labels(path)

        if max_cloud_points is not None and len(coords) > max_cloud_points:
            rng_cloud = make_rng(seed, equipment_name, "cloud_subsample")
            keep_idx = rng_cloud.choice(len(coords), size=max_cloud_points, replace=False)
            coords = coords[keep_idx]
            labels = labels[keep_idx]
            print(f"  Cloud subsample aplicado: {max_cloud_points:,} pontos")

        flange_idx = np.where(labels == 1)[0]
        non_flange_idx = np.where(labels == 0)[0]

        print(
            f"  Pontos: total={len(coords):,} | flange={len(flange_idx):,} | non_flange={len(non_flange_idx):,}"
        )

        if len(coords) == 0 or len(non_flange_idx) == 0:
            print("  [SKIP] Arquivo sem pontos suficientes.\n")
            continue

        tree = KDTree(coords)
        rng_pos = make_rng(seed, equipment_name, "positive_centers")

        if len(flange_idx) > 0:
            pos_centers = sample_without_replacement_or_repeat(flange_idx, pos_per_equipment, rng_pos)
        else:
            pos_centers = np.array([], dtype=np.int64)

        neg_centers, neg_center_dists = build_negative_center_pool(
            coords=coords,
            labels=labels,
            flange_idx=flange_idx,
            non_flange_idx=non_flange_idx,
            neg_per_equipment=neg_per_equipment,
            oversample_factor=oversample_factor,
            hard_neg_ratio=hard_neg_ratio,
            hard_neg_max_dist=hard_neg_max_dist,
            easy_neg_min_dist=easy_neg_min_dist,
            seed_key=f"{seed}|{equipment_name}",
        )

        kept_pos = 0
        kept_neg = 0
        rejected_pos = 0
        rejected_neg = 0

        # Positivos
        for j, center_idx in enumerate(pos_centers):
            patch_xyz, patch_idx = extract_knn_patch(coords, tree, int(center_idx), num_points)
            patch_labels = labels[patch_idx]
            flange_ratio = float(np.mean(patch_labels == 1))

            if flange_ratio < pos_min_flange_ratio:
                rejected_pos += 1
                continue

            rel_path = os.path.join("patches", "flange", f"{equipment_name}_pos_{kept_pos:04d}.ply")
            abs_path = os.path.join(out_dir, rel_path)
            save_points_as_ply(patch_xyz, abs_path)

            center_xyz = coords[int(center_idx)]
            all_rows.append({
                "Equipamento": equipment_name,
                "Source_PLY": fname,
                "Patch_ID": f"{equipment_name}_pos_{kept_pos:04d}",
                "Arquivo_PLY": rel_path,
                "Label": 1,
                "Center_Index": int(center_idx),
                "Center_X": float(center_xyz[0]),
                "Center_Y": float(center_xyz[1]),
                "Center_Z": float(center_xyz[2]),
                "Num_Points": int(num_points),
                "Patch_Flange_Ratio": flange_ratio,
                "Nearest_Flange_Dist": 0.0,
                "Center_Point_Label": 1,
            })
            kept_pos += 1

        # Negativos
        for j, center_idx in enumerate(neg_centers):
            patch_xyz, patch_idx = extract_knn_patch(coords, tree, int(center_idx), num_points)
            patch_labels = labels[patch_idx]
            flange_ratio = float(np.mean(patch_labels == 1))

            if flange_ratio > neg_max_flange_ratio:
                rejected_neg += 1
                continue

            rel_path = os.path.join("patches", "non_flange", f"{equipment_name}_neg_{kept_neg:04d}.ply")
            abs_path = os.path.join(out_dir, rel_path)
            save_points_as_ply(patch_xyz, abs_path)

            center_xyz = coords[int(center_idx)]
            center_dist = float(neg_center_dists[j]) if j < len(neg_center_dists) else np.nan
            all_rows.append({
                "Equipamento": equipment_name,
                "Source_PLY": fname,
                "Patch_ID": f"{equipment_name}_neg_{kept_neg:04d}",
                "Arquivo_PLY": rel_path,
                "Label": 0,
                "Center_Index": int(center_idx),
                "Center_X": float(center_xyz[0]),
                "Center_Y": float(center_xyz[1]),
                "Center_Z": float(center_xyz[2]),
                "Num_Points": int(num_points),
                "Patch_Flange_Ratio": flange_ratio,
                "Nearest_Flange_Dist": center_dist,
                "Center_Point_Label": 0,
            })
            kept_neg += 1

        summary_rows.append({
            "Equipamento": equipment_name,
            "Source_PLY": fname,
            "Total_Points": int(len(coords)),
            "Flange_Points": int(len(flange_idx)),
            "Non_Flange_Points": int(len(non_flange_idx)),
            "Requested_Pos": int(pos_per_equipment),
            "Requested_Neg": int(neg_per_equipment),
            "Kept_Pos": int(kept_pos),
            "Kept_Neg": int(kept_neg),
            "Rejected_Pos": int(rejected_pos),
            "Rejected_Neg": int(rejected_neg),
        })

        print(
            f"  Patches salvos: pos={kept_pos} (rejeitados={rejected_pos}) | "
            f"neg={kept_neg} (rejeitados={rejected_neg})\n"
        )

    if not all_rows:
        raise RuntimeError("Nenhum patch foi salvo. Relaxe os thresholds ou aumente os pontos disponíveis.")

    df = pd.DataFrame(all_rows)
    summary_df = pd.DataFrame(summary_rows)

    csv_path = os.path.join(out_dir, "classification_patch_info.csv")
    summary_path = os.path.join(out_dir, "classification_patch_summary.csv")
    df.to_csv(csv_path, index=False)
    summary_df.to_csv(summary_path, index=False)

    print("=" * 60)
    print("DATASET DE CLASSIFICAÇÃO GERADO")
    print("=" * 60)
    print(f"Metadata: {csv_path}")
    print(f"Resumo   : {summary_path}")
    print(f"Total de patches: {len(df)}")
    print("\nDistribuição por classe:")
    print(df["Label"].value_counts())
    print("\nEquipamentos únicos:")
    print(df["Equipamento"].nunique())
    print("\nPatch_Flange_Ratio por classe:")
    print(df.groupby("Label")["Patch_Flange_Ratio"].describe())


def parse_args():
    parser = argparse.ArgumentParser(
        description="Constrói dataset de classificação a partir da nuvem completa, usando patches candidatos."
    )
    parser.add_argument("--data_dir", type=str, default="../DATA")
    parser.add_argument("--out_dir", type=str, default="./FLANGE_CLASSIFICATION_PATCH_DATASET")
    parser.add_argument("--num_points", type=int, default=2048)
    parser.add_argument("--pos_per_equipment", type=int, default=120)
    parser.add_argument("--neg_per_equipment", type=int, default=120)
    parser.add_argument("--pos_min_flange_ratio", type=float, default=0.03)
    parser.add_argument("--neg_max_flange_ratio", type=float, default=0.005)
    parser.add_argument("--hard_neg_ratio", type=float, default=0.50)
    parser.add_argument("--hard_neg_max_dist", type=float, default=120.0)
    parser.add_argument("--easy_neg_min_dist", type=float, default=250.0)
    parser.add_argument("--oversample_factor", type=int, default=10)
    parser.add_argument("--max_cloud_points", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main():
    args = parse_args()
    build_patch_dataset(
        data_dir=args.data_dir,
        out_dir=args.out_dir,
        num_points=args.num_points,
        pos_per_equipment=args.pos_per_equipment,
        neg_per_equipment=args.neg_per_equipment,
        pos_min_flange_ratio=args.pos_min_flange_ratio,
        neg_max_flange_ratio=args.neg_max_flange_ratio,
        hard_neg_ratio=args.hard_neg_ratio,
        hard_neg_max_dist=args.hard_neg_max_dist,
        easy_neg_min_dist=args.easy_neg_min_dist,
        oversample_factor=args.oversample_factor,
        max_cloud_points=args.max_cloud_points,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()

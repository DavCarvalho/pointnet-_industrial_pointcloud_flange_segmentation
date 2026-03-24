import os
import numpy as np
import pandas as pd
from plyfile import PlyData, PlyElement
import hdbscan


# =========================
# CONFIG
# =========================
DATA_DIR = "./DATA"
OUT_DIR = "./FLANGE_DATASET"

MIN_POINTS_FLANGE = 500
MIN_POINTS_NON_FLANGE = 15000
MAX_NON_FLANGE_INSTANCES = 5
RNG = np.random.default_rng(42)


# =========================
# LEITURA DO PLY
# =========================
def read_ply(file_path):
    plydata = PlyData.read(file_path)
    vertex = plydata["vertex"].data
    prop_names = [p.name for p in plydata["vertex"].properties]

    coords = np.column_stack([
        np.array(vertex["x"], dtype=np.float64),
        np.array(vertex["y"], dtype=np.float64),
        np.array(vertex["z"], dtype=np.float64),
    ])

    class_field = None
    for name in [
        "scalar_Classification", "scalar_classification",
        "Classification", "classification",
        "label", "scalar_Label", "class"
    ]:
        if name in prop_names:
            class_field = name
            break

    if class_field is None:
        raise ValueError(f"Campo de classificação não encontrado em {file_path}")

    labels = np.array(vertex[class_field])
    return coords, labels, class_field


# =========================
# SALVAR PATCH COMO PLY
# =========================
def save_points_as_ply(points, out_path):
    vertex_array = np.array(
        [(float(p[0]), float(p[1]), float(p[2])) for p in points],
        dtype=[("x", "f4"), ("y", "f4"), ("z", "f4")]
    )
    el = PlyElement.describe(vertex_array, "vertex")
    PlyData([el], text=True).write(out_path)


# =========================
# EXTRAÇÃO ROBUSTA DE FLANGES
# =========================
def detectar_flanges(coords, min_cluster_size=50):
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=10,
        cluster_selection_method="eom",
        allow_single_cluster=False,
    )
    labels = clusterer.fit_predict(coords)
    return labels


def merge_clusters_proximos(coords, labels, fator_merge=0.8):
    unique_labels = sorted(set(labels) - {-1})
    if len(unique_labels) <= 1:
        return labels

    info = {}
    for cid in unique_labels:
        pts = coords[labels == cid]
        centroid = pts.mean(axis=0)
        extent = pts.max(axis=0) - pts.min(axis=0)
        diameter = extent.max()
        info[cid] = {"centroid": centroid, "diameter": diameter}

    merge_map = {cid: cid for cid in unique_labels}

    def find_root(x):
        while merge_map[x] != x:
            merge_map[x] = merge_map[merge_map[x]]
            x = merge_map[x]
        return x

    for i, cid_a in enumerate(unique_labels):
        for cid_b in unique_labels[i + 1:]:
            ca = info[cid_a]["centroid"]
            cb = info[cid_b]["centroid"]
            dist = np.linalg.norm(ca - cb)
            diam_medio = (info[cid_a]["diameter"] + info[cid_b]["diameter"]) / 2.0

            if diam_medio > 0 and dist < fator_merge * diam_medio:
                root_a = find_root(cid_a)
                root_b = find_root(cid_b)
                if root_a != root_b:
                    merge_map[root_b] = root_a

    new_labels = labels.copy()
    for cid in unique_labels:
        root = find_root(cid)
        new_labels[labels == cid] = root

    return new_labels


def validar_geometria(coords, labels, min_pontos=30, min_diametro=0.01, max_aspect_ratio=8.0):
    unique_labels = sorted(set(labels) - {-1})
    valid_labels = labels.copy()

    for cid in unique_labels:
        mask = labels == cid
        pts = coords[mask]
        n = pts.shape[0]

        if n < min_pontos:
            valid_labels[mask] = -1
            continue

        extent = pts.max(axis=0) - pts.min(axis=0)
        extent_sorted = np.sort(extent)[::-1]

        if extent_sorted[0] < min_diametro:
            valid_labels[mask] = -1
            continue

        if extent_sorted[2] > 1e-6:
            ar = extent_sorted[0] / extent_sorted[2]
        else:
            ar = extent_sorted[0] / max(extent_sorted[1], 1e-6)

        if ar > max_aspect_ratio:
            valid_labels[mask] = -1
            continue

    return valid_labels


def renumerar_labels(labels):
    unique = sorted(set(labels) - {-1})
    mapping = {old: new for new, old in enumerate(unique)}
    new_labels = labels.copy()
    for old, new in mapping.items():
        new_labels[labels == old] = new
    new_labels[labels == -1] = -1
    return new_labels


def extract_flange_clusters(file_path):
    coords, labels, _ = read_ply(file_path)

    flange_mask = labels == 1
    flange_points = coords[flange_mask]

    if len(flange_points) < 30:
        return []

    mcs = max(30, min(500, int(len(flange_points) * 0.01)))

    cluster_labels = detectar_flanges(flange_points, min_cluster_size=mcs)
    cluster_labels = merge_clusters_proximos(flange_points, cluster_labels)
    cluster_labels = validar_geometria(flange_points, cluster_labels)
    cluster_labels = renumerar_labels(cluster_labels)

    clusters = []
    for cid in sorted(set(cluster_labels) - {-1}):
        pts = flange_points[cluster_labels == cid]
        if len(pts) >= MIN_POINTS_FLANGE:
            clusters.append(pts)

    return clusters


# =========================
# PATCHES NÃO-FLANGE
# =========================
def extract_non_flange_patches(file_path):
    coords, labels, _ = read_ply(file_path)
    non_flange_points = coords[labels != 1]

    patches = []

    if len(non_flange_points) < MIN_POINTS_NON_FLANGE:
        return patches

    num_instances = min(
        MAX_NON_FLANGE_INSTANCES,
        len(non_flange_points) // MIN_POINTS_NON_FLANGE
    )

    for _ in range(num_instances):
        idx = RNG.choice(len(non_flange_points), MIN_POINTS_NON_FLANGE, replace=False)
        patch = non_flange_points[idx]
        patches.append(patch)

    return patches


# =========================
# MAIN
# =========================
def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(os.path.join(OUT_DIR, "flanges"), exist_ok=True)
    os.makedirs(os.path.join(OUT_DIR, "non_flanges"), exist_ok=True)

    ply_files = sorted([f for f in os.listdir(DATA_DIR) if f.lower().endswith(".ply")])

    all_rows = []

    for fname in ply_files:
        equipment_name = os.path.splitext(fname)[0]
        file_path = os.path.join(DATA_DIR, fname)

        print(f"\n=== {equipment_name} ===")

        # FLANGES
        flange_clusters = extract_flange_clusters(file_path)
        print(f"Flanges salvos: {len(flange_clusters)}")

        for i, pts in enumerate(flange_clusters, start=1):
            patch_name = f"{equipment_name}_flange_{i}"
            rel_path = os.path.join("flanges", patch_name + ".ply")
            abs_path = os.path.join(OUT_DIR, rel_path)

            save_points_as_ply(pts, abs_path)

            centroid = pts.mean(axis=0)

            all_rows.append({
                "Equipamento": equipment_name,
                "Flange_ID": patch_name,
                "Num_Pontos": len(pts),
                "Centroid_X": centroid[0],
                "Centroid_Y": centroid[1],
                "Centroid_Z": centroid[2],
                "Arquivo_PLY": rel_path,
                "Label": 1,
            })

        # NON-FLANGES
        non_flange_patches = extract_non_flange_patches(file_path)
        print(f"Non-flanges salvos: {len(non_flange_patches)}")

        for i, pts in enumerate(non_flange_patches, start=1):
            patch_name = f"{equipment_name}_non_flange_{i}"
            rel_path = os.path.join("non_flanges", patch_name + ".ply")
            abs_path = os.path.join(OUT_DIR, rel_path)

            save_points_as_ply(pts, abs_path)

            centroid = pts.mean(axis=0)

            all_rows.append({
                "Equipamento": equipment_name,
                "Flange_ID": patch_name,
                "Num_Pontos": len(pts),
                "Centroid_X": centroid[0],
                "Centroid_Y": centroid[1],
                "Centroid_Z": centroid[2],
                "Arquivo_PLY": rel_path,
                "Label": 0,
            })

    df = pd.DataFrame(all_rows)
    csv_path = os.path.join(OUT_DIR, "flange_info.csv")
    df.to_csv(csv_path, index=False)

    print("\n==============================")
    print("DATASET GERADO")
    print("==============================")
    print(f"Arquivo CSV: {csv_path}")
    print(f"Total de instâncias: {len(df)}")
    print("\nDistribuição por classe:")
    print(df["Label"].value_counts())
    print("\nEquipamentos únicos:")
    print(df["Equipamento"].nunique())


if __name__ == "__main__":
    main()


# # Function to load data from a PLY file
# def load_ply(file_path):
#     plydata = PlyData.read(file_path)
#     vertex = plydata['vertex']
#     x = vertex['x']
#     y = vertex['y']
#     z = vertex['z']
#     if 'scalar_Classification' in vertex.data.dtype.names:
#         labels = vertex['scalar_Classification']
#     else:
#         labels = None
#     coords = np.vstack((x, y, z)).astype(np.float32).T  # Shape (N, 3)
#     return coords, labels

# # Function to extract flanges and non-flanges
# def extract_flanges(coords, labels):
#     if labels is not None:
#         flange_points = coords[labels == 1]
#         non_flange_points = coords[labels != 1]
#     else:
#         # If no labels, consider all points as non-flanges
#         flange_points = np.array([])
#         non_flange_points = coords
#     return flange_points, non_flange_points

# # Function to cluster flanges using HDBSCAN
# def cluster_flanges_hdbscan(flange_points, min_cluster_size=5):
#     clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size)
#     clusters = clusterer.fit_predict(flange_points)
#     return clusters

# # Function to save points to PLY files
# def save_ply(equipment_name, category, points, output_dir, flange_info_list=None, min_points_flange=500, min_points_non_flange=15000, max_instances_non_flange=15):
#     category_dir = os.path.join(output_dir, category)
#     os.makedirs(category_dir, exist_ok=True)

#     if category == 'flanges' and isinstance(points, dict) and 'clusters' in points:
#         unique_clusters = np.unique(points['clusters'])
#         for cluster in unique_clusters:
#             if cluster == -1:
#                 continue
#             cluster_points = points['coords'][points['clusters'] == cluster]
#             num_points = len(cluster_points)

#             # Apply minimum size filter for flanges
#             if num_points < min_points_flange:
#                 print(f"Cluster {cluster} ignored for having only {num_points} points")
#                 continue

#             centroid = cluster_points.mean(axis=0)
#             flange_id = f"{equipment_name}_flange_{cluster+1}"
#             ply_filename = f"{flange_id}.ply"
#             ply_output_path = os.path.join(category_dir, ply_filename)

#             # Save flange points to PLY file
#             vertex_element = np.array(
#                 [(p[0], p[1], p[2]) for p in cluster_points],
#                 dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4')]
#             )
#             el = PlyElement.describe(vertex_element, 'vertex')
#             PlyData([el], text=True).write(ply_output_path)

#             # Add information to dataframe
#             flange_info = {
#                 'Equipamento': equipment_name,
#                 'Flange_ID': flange_id,
#                 'Num_Pontos': num_points,
#                 'Centroid_X': centroid[0],
#                 'Centroid_Y': centroid[1],
#                 'Centroid_Z': centroid[2],
#                 'Arquivo_PLY': os.path.join('flanges', ply_filename),
#                 'Label': 1  # Flange
#             }
#             flange_info_list.append(flange_info)
#             print(f"Saved {flange_id} with {num_points} points")
#     else:
#         # For non-flanges, save a limited number of instances with random sampling
#         if isinstance(points, np.ndarray):
#             num_points_total = len(points)
#             if num_points_total < min_points_non_flange:
#                 print(f"Equipment {equipment_name} ignored for having only {num_points_total} points as non-flange")
#                 return

#             # Limit the number of non-flange instances
#             num_instances = min(max_instances_non_flange, num_points_total // min_points_non_flange)
#             for i in range(num_instances):
#                 # Randomly sample non-flange points
#                 sample_indices = np.random.choice(num_points_total, min_points_non_flange, replace=False)
#                 instance_points = points[sample_indices]

#                 instance_id = f"{equipment_name}_non_flange_{i+1}"
#                 ply_filename = f"{instance_id}.ply"
#                 ply_output_path = os.path.join(category_dir, ply_filename)

#                 # Save non-flange points to PLY file
#                 vertex_element = np.array(
#                     [(p[0], p[1], p[2]) for p in instance_points],
#                     dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4')]
#                 )
#                 el = PlyElement.describe(vertex_element, 'vertex')
#                 PlyData([el], text=True).write(ply_output_path)

#                 # Add information to dataframe
#                 centroid = instance_points.mean(axis=0)
#                 flange_info = {
#                     'Equipamento': equipment_name,
#                     'Flange_ID': instance_id,
#                     'Num_Pontos': len(instance_points),
#                     'Centroid_X': centroid[0],
#                     'Centroid_Y': centroid[1],
#                     'Centroid_Z': centroid[2],
#                     'Arquivo_PLY': os.path.join('non_flanges', ply_filename),
#                     'Label': 0  # Non-flange
#                 }
#                 flange_info_list.append(flange_info)
#                 print(f"Saved {instance_id} with {len(instance_points)} points")
#         else:
#             print("Unrecognized 'points' format for saving PLY.")

# # Function to process each equipment (PLY file)
# def process_equipment(ply_file, project_dir, output_dir, flange_info_list, min_points_flange=500, min_points_non_flange=15000, max_non_flange_instances=15):
#     ply_path = os.path.join(project_dir, ply_file)
#     equipment_name = os.path.splitext(ply_file)[0]

#     coords, labels = load_ply(ply_path)
#     flange_points, non_flange_points = extract_flanges(coords, labels)

#     # Process flanges
#     if len(flange_points) > 0:
#         clusters = cluster_flanges_hdbscan(flange_points, min_cluster_size=5)
#         num_clusters = len(np.unique(clusters)) - (1 if -1 in clusters else 0)
#         print(f"HDBSCAN found {num_clusters} flanges in equipment {equipment_name}")

#         if num_clusters > 0:
#             # Prepare data to save flanges
#             flange_data = {
#                 'coords': flange_points,
#                 'clusters': clusters
#             }

#             # Save flanges (applying min_points_flange filter)
#             save_ply(
#                 equipment_name,
#                 'flanges',
#                 flange_data,
#                 output_dir,
#                 flange_info_list,
#                 min_points_flange=min_points_flange
#             )
#     else:
#         print(f"Warning: No flanges found in equipment {equipment_name}")

#     # Process non-flanges
#     if len(non_flange_points) >= min_points_non_flange:
#         # Save a limited number of non-flange instances
#         save_ply(
#             equipment_name,
#             'non_flanges',
#             non_flange_points,
#             output_dir,
#             flange_info_list,
#             min_points_flange=min_points_flange,
#             min_points_non_flange=min_points_non_flange,
#             max_instances_non_flange=max_non_flange_instances
#         )
#     else:
#         print(f"Warning: Not enough points to split non-flanges in equipment {equipment_name}")

# # Main function to process the data
# def main_data_processing():
#     project_dir = "./DATA/"
#     output_dir = "./FLANGE_DATASET/"
#     os.makedirs(output_dir, exist_ok=True)
#     ply_files = [f for f in os.listdir(project_dir) if f.endswith('.ply')]

#     flange_info_list = []

#     for ply_file in ply_files:
#         process_equipment(
#             ply_file,
#             project_dir,
#             output_dir,
#             flange_info_list,
#             min_points_flange=500,
#             min_points_non_flange=15000,
#             max_non_flange_instances=15
#         )

#     # Create the dataframe
#     flange_df = pd.DataFrame(flange_info_list)

#     # Check class distribution
#     print("Class Distribution:")
#     print(flange_df['Label'].value_counts())

#     # Save dataframe to CSV file
#     flange_df.to_csv(os.path.join(output_dir, 'flange_info.csv'), index=False)
#     print("Flange and non-flange information saved to flange_info.csv")
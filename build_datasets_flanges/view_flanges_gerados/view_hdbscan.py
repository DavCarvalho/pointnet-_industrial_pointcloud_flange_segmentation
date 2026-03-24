"""
Visualização de flanges — Dataset HDBSCAN (FLANGE_DATASET).
Cada flange é um PLY separado, cada non-flange é um PLY separado.

Uso: python visualizar_hdbscan.py
Dependências: pip install open3d plyfile numpy pandas
"""

import os
import numpy as np
import pandas as pd
from plyfile import PlyData
import open3d as o3d

DATASET_DIR = "./FLANGE_DATASET"
CSV_PATH = os.path.join(DATASET_DIR, "flange_info.csv")
MAX_PONTOS = 500_000


def gerar_cores(n):
    cores = [
        [1, 0, 0], [0, 0.8, 0], [0, 0.4, 1], [1, 0.8, 0],
        [1, 0, 1], [0, 1, 1], [1, 0.5, 0], [0.5, 0, 1],
        [0, 1, 0.5], [1, 0.3, 0.5], [0.6, 0.8, 0], [0, 0.6, 0.8],
    ]
    while len(cores) < n:
        cores.append(list(np.random.rand(3) * 0.7 + 0.3))
    return cores[:n]


def load_ply(path):
    v = PlyData.read(path)["vertex"]
    return np.column_stack([v["x"], v["y"], v["z"]]).astype(np.float64)


def visualizar_equipamento(equip, df_equip):
    flanges = df_equip[df_equip["Label"] == 1].reset_index(drop=True)
    non_flanges = df_equip[df_equip["Label"] == 0].reset_index(drop=True)

    n_flanges = len(flanges)
    cores = gerar_cores(n_flanges)
    geometrias = []

    # Non-flanges (cinza)
    for _, row in non_flanges.iterrows():
        path = os.path.join(DATASET_DIR, row["Arquivo_PLY"])
        if not os.path.exists(path):
            print(f"    SKIP: {path}")
            continue
        pts = load_ply(path)
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pts)
        pcd.paint_uniform_color([0.4, 0.4, 0.4])
        if len(pts) > MAX_PONTOS:
            pcd = pcd.random_down_sample(MAX_PONTOS / len(pts))
        geometrias.append(pcd)
        print(f"    Equip: {len(pcd.points):,} pts")

    # Flanges (coloridos)
    for i, (_, row) in enumerate(flanges.iterrows()):
        path = os.path.join(DATASET_DIR, row["Arquivo_PLY"])
        if not os.path.exists(path):
            print(f"    SKIP: {path}")
            continue
        pts = load_ply(path)
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pts)
        pcd.paint_uniform_color(cores[i])
        geometrias.append(pcd)

        # Bounding box
        cx, cy, cz = row["Centroid_X"], row["Centroid_Y"], row["Centroid_Z"]
        ext = pts.max(axis=0) - pts.min(axis=0)
        half = ext / 2.0
        center = np.array([cx, cy, cz])
        bbox = o3d.geometry.AxisAlignedBoundingBox(
            min_bound=center - half, max_bound=center + half
        )
        bbox.color = cores[i]
        geometrias.append(bbox)

        print(f"    Flange {i+1}: {len(pts):,} pts | cor={cores[i]}")

    titulo = f"{equip} — {n_flanges} flanges (HDBSCAN)"
    print(f"\n  Abrindo: {titulo}")

    try:
        o3d.visualization.draw(
            geometrias, title=titulo,
            width=1600, height=1000, show_ui=True,
            bg_color=(0.05, 0.05, 0.1, 1.0), point_size=2,
        )
    except (AttributeError, TypeError):
        vis = o3d.visualization.Visualizer()
        vis.create_window(window_name=titulo, width=1600, height=1000)
        for g in geometrias:
            vis.add_geometry(g)
        opt = vis.get_render_option()
        opt.point_size = 2.0
        opt.background_color = np.array([0.05, 0.05, 0.1])
        vis.run()
        vis.destroy_window()


def main():
    df = pd.read_csv(CSV_PATH)
    equipamentos = sorted(df["Equipamento"].unique())
    print(f"Equipamentos: {len(equipamentos)}\n")

    print("  0 — Ver TODOS")
    for i, eq in enumerate(equipamentos, 1):
        nf = (df[(df["Equipamento"] == eq) & (df["Label"] == 1)]).shape[0]
        print(f"  {i} — {eq} ({nf} flanges)")

    escolha = input("\nEscolha: ").strip()

    if escolha == "0":
        lista = equipamentos
    elif escolha.isdigit() and 1 <= int(escolha) <= len(equipamentos):
        lista = [equipamentos[int(escolha) - 1]]
    else:
        print("Inválido.")
        return

    for equip in lista:
        df_eq = df[df["Equipamento"] == equip]
        print(f"\n{'='*50}")
        print(f"  {equip}")
        print(f"{'='*50}")
        visualizar_equipamento(equip, df_eq)


if __name__ == "__main__":
    main()
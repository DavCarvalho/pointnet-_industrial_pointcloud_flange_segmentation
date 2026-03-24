"""
Visualização de flanges — Dataset Labels (FLANGE_DATASET_labels).
1 PLY com todos os flanges + 1 PLY com o equipamento por equipamento.

Uso: python visualizar_labels.py
Dependências: pip install open3d plyfile numpy pandas
"""

import os
import numpy as np
import pandas as pd
from plyfile import PlyData
import open3d as o3d

DATASET_DIR = "./FLANGE_DATASET_labels"
CSV_PATH = os.path.join(DATASET_DIR, "flange_info_labels.csv")
MAX_PONTOS = 500_000


def load_ply(path):
    v = PlyData.read(path)["vertex"]
    return np.column_stack([v["x"], v["y"], v["z"]]).astype(np.float64)


def visualizar_equipamento(equip, df_equip):
    geometrias = []

    # Equipamento (cinza)
    row_equip = df_equip[df_equip["Label"] == 0]
    if len(row_equip) > 0:
        path = os.path.join(DATASET_DIR, row_equip.iloc[0]["Arquivo_PLY"])
        if os.path.exists(path):
            pts = load_ply(path)
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(pts)
            pcd.paint_uniform_color([0.4, 0.4, 0.4])
            if len(pts) > MAX_PONTOS:
                pcd = pcd.random_down_sample(MAX_PONTOS / len(pts))
            geometrias.append(pcd)
            print(f"    Equipamento: {len(pcd.points):,} pts")

    # Flanges (vermelho)
    row_flange = df_equip[df_equip["Label"] == 1]
    if len(row_flange) > 0:
        path = os.path.join(DATASET_DIR, row_flange.iloc[0]["Arquivo_PLY"])
        if os.path.exists(path):
            pts = load_ply(path)
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(pts)
            pcd.paint_uniform_color([1, 0, 0])
            if len(pts) > MAX_PONTOS:
                pcd = pcd.random_down_sample(MAX_PONTOS / len(pts))
            geometrias.append(pcd)
            print(f"    Flanges: {len(pcd.points):,} pts (vermelho)")

    titulo = f"{equip} — flanges vs equipamento (Labels)"
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
        row_f = df[(df["Equipamento"] == eq) & (df["Label"] == 1)]
        n_pts = row_f.iloc[0]["Num_Pontos"] if len(row_f) > 0 else 0
        print(f"  {i} — {eq} ({n_pts:,} pts flange)")

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
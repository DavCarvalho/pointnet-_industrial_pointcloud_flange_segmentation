# """
# Visualização rápida de flanges — otimizado pra GPU e equipamentos grandes.

# Otimizações:
# - Voxel downsampling pra display (reduz pontos 5-10x, visual igual)
# - KDTree pra atribuição de pontos (O(n log n) vs O(n*k) do broadcasting)
# - Open3D GPU rendering com o3d.visualization.draw() (novo, mais rápido)
# - Tudo batched, sem loop por ponto

# Uso: python visualizar_flanges_gpu.py
# Dependências: pip install open3d plyfile numpy pandas
# """

# import os
# import numpy as np
# import pandas as pd
# from plyfile import PlyData
# import open3d as o3d
# from scipy.spatial import cKDTree


# # ==============================
# # CONFIGURAÇÃO
# # ==============================
# PROJECT_DIR = "./DATA/"
# CSV_PATH = "flanges_detalhe_gemini.csv"
# VOXEL_DOWNSAMPLE = True   # True = muito mais rápido, False = todos os pontos
# MAX_PONTOS_DISPLAY = 500_000  # se tiver mais que isso, downsamplea


# def gerar_cores(n):
#     cores = [
#         [1, 0, 0], [0, 0.8, 0], [0, 0.4, 1], [1, 0.8, 0],
#         [1, 0, 1], [0, 1, 1], [1, 0.5, 0], [0.5, 0, 1],
#         [0, 1, 0.5], [1, 0.3, 0.5], [0.6, 0.8, 0], [0, 0.6, 0.8],
#         [0.8, 0.4, 0.2], [0.4, 0.6, 1], [1, 0.6, 0.6], [0.2, 0.8, 0.8],
#         [0.9, 0.3, 0.1], [0.3, 0.9, 0.3], [0.1, 0.3, 0.9], [0.9, 0.9, 0.1],
#         [0.7, 0.1, 0.7], [0.1, 0.7, 0.4], [0.8, 0.6, 0.1], [0.4, 0.2, 0.8],
#         [0.2, 0.5, 0.3], [0.9, 0.4, 0.6], [0.5, 0.7, 0.2], [0.3, 0.3, 0.7],
#         [0.7, 0.5, 0.5], [0.5, 0.8, 0.7], [0.8, 0.2, 0.4], [0.4, 0.9, 0.6],
#         [0.6, 0.3, 0.2], [0.2, 0.6, 0.9], [0.9, 0.7, 0.3], [0.3, 0.4, 0.5],
#         [0.7, 0.8, 0.4], [0.5, 0.2, 0.6], [0.8, 0.5, 0.8], [0.4, 0.7, 0.3],
#         [0.6, 0.6, 0.1], [0.1, 0.5, 0.7], [0.9, 0.2, 0.8], [0.2, 0.9, 0.2],
#         [0.7, 0.3, 0.9], [0.3, 0.7, 0.1], [0.8, 0.8, 0.6], [0.6, 0.1, 0.5],
#         [0.1, 0.8, 0.6], [0.5, 0.5, 0.9],
#     ]
#     while len(cores) < n:
#         cores.append(list(np.random.rand(3) * 0.7 + 0.3))
#     return cores[:n]


# def carregar_ply_rapido(ply_path):
#     """Carrega PLY direto como arrays numpy."""
#     plydata = PlyData.read(ply_path)
#     v = plydata['vertex'].data
#     props = [p.name for p in plydata['vertex'].properties]

#     coords = np.column_stack([v['x'], v['y'], v['z']]).astype(np.float64)

#     class_field = None
#     for name in ['scalar_Classification', 'scalar_classification',
#                  'Classification', 'classification', 'label', 'scalar_Label']:
#         if name in props:
#             class_field = name
#             break

#     labels = np.array(v[class_field]) if class_field else np.zeros(len(coords))
#     return coords, labels


# def auto_voxel_size(coords, target_points):
#     """Calcula voxel size pra atingir ~target_points após downsample."""
#     extent = coords.max(axis=0) - coords.min(axis=0)
#     volume = np.prod(extent[extent > 0])
#     if volume == 0:
#         return 0.01
#     density = len(coords) / volume
#     target_density = target_points / volume
#     ratio = (density / target_density) ** (1.0 / 3.0)
#     # voxel_size ≈ espaçamento médio atual × ratio
#     avg_spacing = (volume / len(coords)) ** (1.0 / 3.0)
#     return avg_spacing * ratio


# def visualizar_equipamento(ply_path, df_flanges, equip_name):
#     coords, labels = carregar_ply_rapido(ply_path)
#     n_total = len(coords)

#     mask_corpo = labels != 1
#     mask_flange = labels == 1
#     n_flanges = len(df_flanges)
#     cores = gerar_cores(n_flanges)

#     print(f"  Pontos total: {n_total:,} | Corpo: {mask_corpo.sum():,} | Flange: {mask_flange.sum():,}")

#     geometrias = []

#     # === CORPO (cinza) ===
#     pcd_corpo = o3d.geometry.PointCloud()
#     pcd_corpo.points = o3d.utility.Vector3dVector(coords[mask_corpo])
#     pcd_corpo.paint_uniform_color([0.4, 0.4, 0.4])

#     # Downsample corpo se muito grande
#     if VOXEL_DOWNSAMPLE and mask_corpo.sum() > MAX_PONTOS_DISPLAY:
#         vs = auto_voxel_size(coords[mask_corpo], MAX_PONTOS_DISPLAY)
#         pcd_corpo = pcd_corpo.voxel_down_sample(vs)
#         print(f"  Corpo downsampled: {len(pcd_corpo.points):,} pontos (voxel={vs:.4f})")

#     geometrias.append(pcd_corpo)

#     # === FLANGES (coloridos) ===
#     flange_coords = coords[mask_flange]

#     if n_flanges > 0 and len(flange_coords) > 0:
#         # KDTree nos centróides — atribuição O(n log k) em vez de O(n*k)
#         centroides = df_flanges[['centroid_x', 'centroid_y', 'centroid_z']].values
#         tree = cKDTree(centroides)
#         _, atribuicoes = tree.query(flange_coords, k=1, workers=-1)  # paralelo

#         # Montar array de cores de uma vez (vetorizado)
#         cores_array = np.array(cores)
#         cores_por_ponto = cores_array[atribuicoes]

#         pcd_flanges = o3d.geometry.PointCloud()
#         pcd_flanges.points = o3d.utility.Vector3dVector(flange_coords)
#         pcd_flanges.colors = o3d.utility.Vector3dVector(cores_por_ponto)

#         # Downsample flanges se necessário (mantém cores)
#         if VOXEL_DOWNSAMPLE and len(flange_coords) > MAX_PONTOS_DISPLAY // 2:
#             vs = auto_voxel_size(flange_coords, MAX_PONTOS_DISPLAY // 2)
#             pcd_flanges = pcd_flanges.voxel_down_sample(vs)
#             print(f"  Flanges downsampled: {len(pcd_flanges.points):,} pontos")

#         geometrias.append(pcd_flanges)

#         # === Esferas nos centróides ===
#         extent = coords.max(axis=0) - coords.min(axis=0)
#         sphere_r = extent.max() * 0.006

#         for i, row in df_flanges.iterrows():
#             esfera = o3d.geometry.TriangleMesh.create_sphere(
#                 radius=sphere_r, resolution=8  # resolução baixa = mais rápido
#             )
#             esfera.translate([row['centroid_x'], row['centroid_y'], row['centroid_z']])
#             esfera.paint_uniform_color(cores[i % n_flanges])
#             esfera.compute_vertex_normals()
#             geometrias.append(esfera)

#         # === Bounding boxes ===
#         for i, row in df_flanges.iterrows():
#             cx, cy, cz = row['centroid_x'], row['centroid_y'], row['centroid_z']
#             ex, ey, ez = row['extensao_x'], row['extensao_y'], row['extensao_z']
#             half = np.array([ex, ey, ez]) / 2.0
#             mn = np.array([cx, cy, cz]) - half
#             mx = np.array([cx, cy, cz]) + half

#             bbox = o3d.geometry.AxisAlignedBoundingBox(min_bound=mn, max_bound=mx)
#             bbox.color = cores[i % n_flanges]
#             geometrias.append(bbox)

#     # === Render ===
#     titulo = f"{equip_name} — {n_flanges} flanges"
#     print(f"\n  Abrindo: {titulo}")
#     for i, row in df_flanges.iterrows():
#         print(f"    Flange {row['flange_id']:>2d}: {row['n_pontos']:>6d} pts | "
#               f"({row['centroid_x']:.1f}, {row['centroid_y']:.1f}, {row['centroid_z']:.1f})")

#     # Tentar o visualizador novo (GPU-accelerated), fallback pro legado
#     try:
#         o3d.visualization.draw(
#             geometrias,
#             title=titulo,
#             width=1600,
#             height=1000,
#             show_ui=True,
#             bg_color=(0.05, 0.05, 0.1, 1.0),
#             point_size=2,
#         )
#     except (AttributeError, TypeError):
#         # Fallback pro visualizador legado se versão do o3d não suporta draw()
#         vis = o3d.visualization.Visualizer()
#         vis.create_window(window_name=titulo, width=1600, height=1000)
#         for g in geometrias:
#             vis.add_geometry(g)
#         opt = vis.get_render_option()
#         opt.point_size = 2.0
#         opt.background_color = np.array([0.05, 0.05, 0.1])
#         vis.run()
#         vis.destroy_window()


# def main():
#     if not os.path.isfile(CSV_PATH):
#         print(f"ERRO: '{CSV_PATH}' não encontrado.")
#         return

#     df = pd.read_csv(CSV_PATH)
#     equipamentos = sorted(df['equipamento'].unique())
#     print(f"Equipamentos: {len(equipamentos)}\n")

#     # Menu pra escolher qual ver (útil com 28 equipamentos)
#     print("Escolha:")
#     print("  0 — Ver TODOS (um por um)")
#     for i, eq in enumerate(equipamentos, 1):
#         n = len(df[df['equipamento'] == eq])
#         print(f"  {i} — {eq} ({n} flanges)")

#     escolha = input("\nDigite o número: ").strip()

#     if escolha == '0':
#         lista = equipamentos
#     elif escolha.isdigit() and 1 <= int(escolha) <= len(equipamentos):
#         lista = [equipamentos[int(escolha) - 1]]
#     else:
#         print("Opção inválida.")
#         return

#     for equip in lista:
#         ply_path = os.path.join(PROJECT_DIR, equip + '.ply')
#         if not os.path.isfile(ply_path):
#             print(f"PLY não encontrado: {ply_path}")
#             continue

#         df_eq = df[df['equipamento'] == equip].reset_index(drop=True)
#         print(f"\n{'=' * 50}")
#         print(f" {equip}: {len(df_eq)} flanges")
#         print(f"{'=' * 50}")

#         visualizar_equipamento(ply_path, df_eq, equip)


# if __name__ == '__main__':
#     main()

"""
Visualização RÁPIDA de flanges — Modo Turbo (GPU + CPU Otimizados).

Otimizações aplicadas:
- Random downsampling (MUITO mais rápido que Voxel, visualmente idêntico)
- KDTree para atribuição de pontos (O(n log n))
- Remoção de geometrias redundantes (Esferas) para aliviar o Scene Graph
- Open3D GPU rendering com o3d.visualization.draw()

Uso: python visualizar_flanges_gpu.py
"""

import os
import numpy as np
import pandas as pd
from plyfile import PlyData
import open3d as o3d
from scipy.spatial import cKDTree

# ==============================
# CONFIGURAÇÃO
# ==============================
PROJECT_DIR = "./DATA/"
CSV_PATH = "resultado_flanges_detalhado_hdbscan_geom.csv"
MAX_PONTOS_DISPLAY = 500_000  # Limita os pontos na tela para manter o FPS alto

def gerar_cores(n):
    cores = [
        [1, 0, 0], [0, 0.8, 0], [0, 0.4, 1], [1, 0.8, 0],
        [1, 0, 1], [0, 1, 1], [1, 0.5, 0], [0.5, 0, 1],
        [0, 1, 0.5], [1, 0.3, 0.5], [0.6, 0.8, 0], [0, 0.6, 0.8],
        [0.8, 0.4, 0.2], [0.4, 0.6, 1], [1, 0.6, 0.6], [0.2, 0.8, 0.8],
    ]
    while len(cores) < n:
        cores.append(list(np.random.rand(3) * 0.7 + 0.3))
    return cores[:n]

def carregar_ply_rapido(ply_path):
    """Carrega PLY extraindo as coordenadas e a classe."""
    plydata = PlyData.read(ply_path)
    v = plydata['vertex'].data
    props = [p.name for p in plydata['vertex'].properties]

    coords = np.column_stack([v['x'], v['y'], v['z']]).astype(np.float64)

    class_field = None
    # Adicionado 'scalar_Scalar_field' como fallback, caso venha do CloudCompare
    for name in ['scalar_Classification', 'scalar_classification',
                 'Classification', 'classification', 'label', 'scalar_Label', 'scalar_Scalar_field']:
        if name in props:
            class_field = name
            break

    labels = np.array(v[class_field]) if class_field else np.zeros(len(coords))
    return coords, labels

def visualizar_equipamento(ply_path, df_flanges, equip_name):
    coords, labels = carregar_ply_rapido(ply_path)
    n_total = len(coords)

    mask_corpo = labels != 1
    mask_flange = labels == 1
    n_flanges = len(df_flanges)
    cores = gerar_cores(n_flanges)

    print(f"  Pontos total: {n_total:,} | Corpo: {mask_corpo.sum():,} | Flange: {mask_flange.sum():,}")

    geometrias = []

    # === CORPO (cinza) ===
    pcd_corpo = o3d.geometry.PointCloud()
    pcd_corpo.points = o3d.utility.Vector3dVector(coords[mask_corpo])
    pcd_corpo.paint_uniform_color([0.4, 0.4, 0.4])

    # Downsample RÁPIDO (Random em vez de Voxel)
    if mask_corpo.sum() > MAX_PONTOS_DISPLAY:
        taxa = MAX_PONTOS_DISPLAY / mask_corpo.sum()
        pcd_corpo = pcd_corpo.random_down_sample(taxa)
        print(f"  Corpo downsampled (Random): {len(pcd_corpo.points):,} pontos")

    geometrias.append(pcd_corpo)

    # === FLANGES (coloridos) ===
    flange_coords = coords[mask_flange]

    if n_flanges > 0 and len(flange_coords) > 0:
        # KDTree nos centróides — atribuição O(n log k)
        centroides = df_flanges[['centroid_x', 'centroid_y', 'centroid_z']].values
        tree = cKDTree(centroides)
        _, atribuicoes = tree.query(flange_coords, k=1, workers=-1)

        cores_array = np.array(cores)
        cores_por_ponto = cores_array[atribuicoes]

        pcd_flanges = o3d.geometry.PointCloud()
        pcd_flanges.points = o3d.utility.Vector3dVector(flange_coords)
        pcd_flanges.colors = o3d.utility.Vector3dVector(cores_por_ponto)

        # Downsample flanges se necessário
        limite_flanges = MAX_PONTOS_DISPLAY // 2
        if len(flange_coords) > limite_flanges:
            taxa = limite_flanges / len(flange_coords)
            pcd_flanges = pcd_flanges.random_down_sample(taxa)
            print(f"  Flanges downsampled (Random): {len(pcd_flanges.points):,} pontos")

        geometrias.append(pcd_flanges)

        # === Bounding boxes (Esferas removidas para aliviar o render) ===
        for i, row in df_flanges.iterrows():
            cx, cy, cz = row['centroid_x'], row['centroid_y'], row['centroid_z']
            ex, ey, ez = row['extensao_x'], row['extensao_y'], row['extensao_z']
            half = np.array([ex, ey, ez]) / 2.0
            
            mn = np.array([cx, cy, cz]) - half
            mx = np.array([cx, cy, cz]) + half

            bbox = o3d.geometry.AxisAlignedBoundingBox(min_bound=mn, max_bound=mx)
            bbox.color = cores[i % n_flanges]
            geometrias.append(bbox)

    # === Render ===
    titulo = f"{equip_name} — {n_flanges} flanges"
    print(f"\n  Abrindo: {titulo}")
    
    try:
        o3d.visualization.draw(
            geometrias,
            title=titulo,
            width=1600,
            height=1000,
            show_ui=True,
            bg_color=(0.05, 0.05, 0.1, 1.0),
            point_size=2,
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
    if not os.path.isfile(CSV_PATH):
        print(f"ERRO: '{CSV_PATH}' não encontrado.")
        return

    df = pd.read_csv(CSV_PATH)
    equipamentos = sorted(df['equipamento'].unique())
    print(f"Equipamentos encontrados no CSV: {len(equipamentos)}\n")

    print("Escolha:")
    print("  0 — Ver TODOS (um por um)")
    for i, eq in enumerate(equipamentos, 1):
        n = len(df[df['equipamento'] == eq])
        print(f"  {i} — {eq} ({n} flanges)")

    escolha = input("\nDigite o número: ").strip()

    if escolha == '0':
        lista = equipamentos
    elif escolha.isdigit() and 1 <= int(escolha) <= len(equipamentos):
        lista = [equipamentos[int(escolha) - 1]]
    else:
        print("Opção inválida.")
        return

    for equip in lista:
        ply_path = os.path.join(PROJECT_DIR, equip + '.ply')
        if not os.path.isfile(ply_path):
            print(f"PLY não encontrado: {ply_path}")
            continue

        df_eq = df[df['equipamento'] == equip].reset_index(drop=True)
        print(f"\n{'=' * 50}")
        print(f" {equip}: {len(df_eq)} flanges")
        print(f"{'=' * 50}")

        visualizar_equipamento(ply_path, df_eq, equip)

if __name__ == '__main__':
    main()
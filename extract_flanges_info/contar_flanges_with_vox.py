"""
Extração robusta de flanges de arquivos PLY para revista científica.

Pipeline:
1. Filtra pontos classificados como flange (label == 1)
2. HDBSCAN clustering (sem eps — descobre clusters automaticamente)
3. Merge automático de clusters próximos (resolve o problema de
   flanges divididos em face superior/inferior)
4. Validação geométrica: descarta clusters com geometria incompatível
5. Gera CSVs de resumo e detalhamento

Dependências: pip install hdbscan plyfile pandas numpy scikit-learn
"""

import os
import numpy as np
import pandas as pd
from plyfile import PlyData
import hdbscan
from scipy.spatial.distance import cdist


def detectar_flanges(coords, min_cluster_size=50):
    """
    Etapa 1: HDBSCAN para clustering inicial.
    min_cluster_size é o único parâmetro, e é robusto:
    significa "um flange precisa ter pelo menos N pontos".
    """
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=10,
        cluster_selection_method='eom',  # Excess of Mass — melhor para clusters de tamanhos variados
        allow_single_cluster=False,
    )
    labels = clusterer.fit_predict(coords)
    return labels


def merge_clusters_proximos(coords, labels, fator_merge=0.8):
    """
    Etapa 2: Merge de clusters cujos centróides estão muito próximos.

    Resolve o problema clássico: flange dividido em face superior e inferior.
    Dois clusters são mergeados se a distância entre centróides é menor que
    (fator_merge × diâmetro médio dos dois clusters).

    fator_merge=0.8: espessura de flange ≈ 5-15% do diâmetro,
    então faces opostas têm centróides bem dentro desse threshold.
    """
    unique_labels = sorted(set(labels) - {-1})
    if len(unique_labels) <= 1:
        return labels

    # Calcular centróide e "diâmetro" (extensão máxima) de cada cluster
    info = {}
    for cid in unique_labels:
        mask = labels == cid
        pts = coords[mask]
        centroid = pts.mean(axis=0)
        extent = pts.max(axis=0) - pts.min(axis=0)
        diameter = extent.max()
        info[cid] = {'centroid': centroid, 'diameter': diameter, 'n_pts': mask.sum()}

    # === DIAGNÓSTICO: Mostrar distâncias entre todos os pares ===
    print(f"\n  --- Diagnóstico de merge (fator={fator_merge}) ---")
    print(f"  {'Par':>12} | {'Dist':>10} | {'Diam_med':>10} | {'Razão':>8} | {'Merge?':>6}")
    print(f"  {'-'*60}")

    merge_map = {cid: cid for cid in unique_labels}

    def find_root(x):
        while merge_map[x] != x:
            merge_map[x] = merge_map[merge_map[x]]
            x = merge_map[x]
        return x

    for i, cid_a in enumerate(unique_labels):
        for cid_b in unique_labels[i + 1:]:
            ca = info[cid_a]['centroid']
            cb = info[cid_b]['centroid']
            dist = np.linalg.norm(ca - cb)
            diam_medio = (info[cid_a]['diameter'] + info[cid_b]['diameter']) / 2.0
            razao = dist / diam_medio if diam_medio > 0 else float('inf')

            deve_merge = dist < fator_merge * diam_medio

            # Só imprime pares com razão < 2.0 (candidatos plausíveis)
            if razao < 2.0:
                tag = "SIM" if deve_merge else "não"
                print(f"  {cid_a:>5}-{cid_b:<5} | {dist:>10.3f} | {diam_medio:>10.3f} | {razao:>8.3f} | {tag:>6}")

            if deve_merge:
                root_a = find_root(cid_a)
                root_b = find_root(cid_b)
                if root_a != root_b:
                    merge_map[root_b] = root_a

    print(f"  {'-'*60}")

    # Aplicar merge nos labels
    new_labels = labels.copy()
    n_merges = 0
    for cid in unique_labels:
        root = find_root(cid)
        if root != cid:
            new_labels[labels == cid] = root
            n_merges += 1
    print(f"  Total de merges realizados: {n_merges}\n")

    return new_labels


def validar_geometria(coords, labels, min_pontos=30,
                      min_diametro=0.01, max_aspect_ratio=8.0):
    """
    Etapa 3: Validação geométrica.
    Descarta clusters que claramente não são flanges:
    - Muito poucos pontos
    - Diâmetro implausível (poeira/ruído)
    - Aspect ratio muito alto (tubo/pipe, não flange)
    """
    unique_labels = sorted(set(labels) - {-1})
    valid_labels = labels.copy()

    descartados = 0
    for cid in unique_labels:
        mask = labels == cid
        pts = coords[mask]
        n = pts.shape[0]

        if n < min_pontos:
            valid_labels[mask] = -1
            descartados += 1
            continue

        extent = pts.max(axis=0) - pts.min(axis=0)
        extent_sorted = np.sort(extent)[::-1]  # maior → menor

        # Diâmetro (maior extensão)
        if extent_sorted[0] < min_diametro:
            valid_labels[mask] = -1
            descartados += 1
            continue

        # Aspect ratio: flange é achatado, mas não absurdamente alongado
        # Se a maior dimensão é >> que as outras, provavelmente é um pedaço de tubo
        if extent_sorted[2] > 1e-6:  # evitar div por zero
            ar = extent_sorted[0] / extent_sorted[2]
        else:
            ar = extent_sorted[0] / max(extent_sorted[1], 1e-6)

        if ar > max_aspect_ratio:
            valid_labels[mask] = -1
            descartados += 1
            continue

    if descartados > 0:
        print(f"  Validação: {descartados} clusters descartados por geometria inválida")

    return valid_labels


def renumerar_labels(labels):
    """Re-numera labels sequencialmente: 0, 1, 2, ..."""
    unique = sorted(set(labels) - {-1})
    mapping = {old: new for new, old in enumerate(unique)}
    new_labels = labels.copy()
    for old, new in mapping.items():
        new_labels[labels == old] = new
    new_labels[labels == -1] = -1
    return new_labels


def extrair_flanges(ply_path):
    """Pipeline completo para um arquivo PLY."""
    plydata = PlyData.read(ply_path)
    vertex = plydata['vertex'].data
    prop_names = [p.name for p in plydata['vertex'].properties]

    x = np.array(vertex['x'], dtype=np.float64)
    y = np.array(vertex['y'], dtype=np.float64)
    z = np.array(vertex['z'], dtype=np.float64)

    # Encontrar campo de classificação
    class_field = None
    for name in ['scalar_Classification', 'scalar_classification',
                 'Classification', 'classification', 'label',
                 'scalar_Label', 'class']:
        if name in prop_names:
            class_field = name
            break

    if class_field is None:
        print(f"  ERRO: Campo de classificação não encontrado.")
        print(f"  Campos disponíveis: {prop_names}")
        return []

    class_values = np.array(vertex[class_field])
    unique_vals, counts = np.unique(class_values, return_counts=True)
    print(f"  Campo: '{class_field}'")
    for v, c in zip(unique_vals, counts):
        print(f"    classe {int(v)}: {c} pontos")

    # Filtrar classe 1 (flanges)
    mask = class_values == 1
    n_pts = mask.sum()
    print(f"  Total pontos flange: {n_pts}")

    if n_pts < 30:
        print(f"  Insuficiente para clustering.")
        return []

    coords = np.column_stack([x[mask], y[mask], z[mask]])

    # --- Pipeline ---

    # 1) HDBSCAN
    # min_cluster_size: adapta ao total de pontos
    # Para ~30k pontos de flange com ~9 flanges → ~3k pts/flange em média
    # Usar ~1% do total como min, com piso de 30 e teto de 500
    mcs = max(30, min(500, int(n_pts * 0.01)))
    print(f"  HDBSCAN min_cluster_size: {mcs}")

    labels = detectar_flanges(coords, min_cluster_size=mcs)
    n_inicial = len(set(labels) - {-1})
    n_ruido = (labels == -1).sum()
    print(f"  Clusters iniciais: {n_inicial} | Ruído: {n_ruido} pontos")

    # 2) Merge de faces
    labels = merge_clusters_proximos(coords, labels)
    n_apos_merge = len(set(labels) - {-1})
    if n_apos_merge != n_inicial:
        print(f"  Após merge de faces: {n_apos_merge} flanges")

    # 3) Validação geométrica
    labels = validar_geometria(coords, labels)
    labels = renumerar_labels(labels)
    n_final = len(set(labels) - {-1})
    print(f"  Flanges finais (validados): {n_final}")

    # Montar resultado
    flanges = []
    for cid in sorted(set(labels) - {-1}):
        cmask = labels == cid
        pts = coords[cmask]
        centroid = pts.mean(axis=0)
        bbox = pts.max(axis=0) - pts.min(axis=0)

        flanges.append({
            'flange_id': int(cid),
            'n_pontos': int(cmask.sum()),
            'centroid_x': round(centroid[0], 4),
            'centroid_y': round(centroid[1], 4),
            'centroid_z': round(centroid[2], 4),
            'extensao_x': round(bbox[0], 4),
            'extensao_y': round(bbox[1], 4),
            'extensao_z': round(bbox[2], 4),
        })

    return flanges


def main():
    project_dir = "./DATA/"

    if not os.path.isdir(project_dir):
        print(f"ERRO: Diretório '{project_dir}' não encontrado.")
        return

    ply_files = sorted([f for f in os.listdir(project_dir) if f.lower().endswith('.ply')])
    print(f"Encontrados {len(ply_files)} arquivos PLY\n")

    if not ply_files:
        return

    todos_flanges = []
    resumo = []

    for ply_file in ply_files:
        equip = os.path.splitext(ply_file)[0]
        print(f"{'=' * 60}")
        print(f" {equip}")
        print(f"{'=' * 60}")

        try:
            flanges = extrair_flanges(os.path.join(project_dir, ply_file))
        except Exception as e:
            print(f"  ERRO: {e}")
            flanges = []

        resumo.append({'equipamento': equip, 'n_flanges': len(flanges)})

        for f in flanges:
            f['equipamento'] = equip
            todos_flanges.append(f)

        print()

    # === Salvar ===
    df_resumo = pd.DataFrame(resumo)
    df_resumo.to_csv('flanges_resumo.csv', index=False)

    print(f"\n{'=' * 60}")
    print(" RESUMO")
    print(f"{'=' * 60}")
    print(df_resumo.to_string(index=False))
    print(f"\n→ flanges_resumo.csv")

    if todos_flanges:
        df_det = pd.DataFrame(todos_flanges)
        cols = ['equipamento', 'flange_id', 'n_pontos',
                'centroid_x', 'centroid_y', 'centroid_z',
                'extensao_x', 'extensao_y', 'extensao_z']
        df_det = df_det[cols]
        df_det.to_csv('flanges_detalhe_vox.csv', index=False)
        print(f"→ flanges_detalhe.csv")

    total = df_resumo['n_flanges'].sum()
    print(f"\nTotal geral: {total} flanges em {len(ply_files)} equipamentos")


if __name__ == '__main__':
    main()
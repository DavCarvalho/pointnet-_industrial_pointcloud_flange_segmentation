"""
Script de Inferência e Visualização
Atualizado para usar a arquitetura PointNet++ atual
"""

import os
import numpy as np
import torch
import open3d as o3d
from plyfile import PlyData
import torch.nn as nn

# CORREÇÃO: Importar do arquivo correto
# Opção 1: Se usar PointNet vanilla
# from pointnet import PointNet, cloud_loader, preprocess_point_cloud

# Opção 2: Se usar PointNet++ advanced (recomendado)
from a_a_a import PointNet2SemSeg, read_ply_with_labels_plyfile

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using Device:", device)

# ==========================================
# FUNÇÕES AUXILIARES
# ==========================================

def load_ply_for_inference(filename):
    """
    Carrega arquivo PLY para inferência
    Adaptado do read_ply_with_labels_plyfile
    """
    plydata = PlyData.read(filename)
    vertex_data = plydata['vertex'].data

    # Extrair coordenadas
    x = vertex_data['x']
    y = vertex_data['y']
    z = vertex_data['z']
    
    # Extrair features opcionais
    r = vertex_data['red'] if 'red' in vertex_data.dtype.names else np.zeros_like(x, dtype=np.float32)
    g = vertex_data['green'] if 'green' in vertex_data.dtype.names else np.zeros_like(x, dtype=np.float32)
    b = vertex_data['blue'] if 'blue' in vertex_data.dtype.names else np.zeros_like(x, dtype=np.float32)
    intensity = vertex_data['intensity'] if 'intensity' in vertex_data.dtype.names else np.zeros_like(x, dtype=np.float32)

    # Montar array de coordenadas e features
    coords = np.vstack((x, y, z)).T  # (N, 3)
    
    # Features adicionais (RGB + Intensity)
    features = np.vstack((r, g, b)).T / 255.0  # Normalizar RGB
    intensity = intensity[:, np.newaxis]  # (N, 1)
    
    return coords, features, intensity

def normalize_point_cloud(coords):
    """
    Normaliza nuvem de pontos para zero-mean e escala unitária
    """
    # Centralizar
    centroid = np.mean(coords, axis=0, keepdims=True)
    coords = coords - centroid
    
    # Escalar para esfera unitária
    max_dist = np.max(np.sqrt(np.sum(coords**2, axis=1)))
    if max_dist > 0:
        coords = coords / max_dist
    
    return coords

def compute_geometric_features(coords):
    """
    Calcula features geométricas (normais, curvatura)
    Necessário para PointNet++ Advanced
    """
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(coords)
    
    # Estimar normais
    pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30)
    )
    normals = np.asarray(pcd.normals)
    
    # Calcular curvatura (simplificado)
    # Aqui você pode usar a mesma lógica do a_a_a.py
    from sklearn.neighbors import NearestNeighbors
    nbrs = NearestNeighbors(n_neighbors=10, algorithm='kd_tree').fit(coords)
    distances, _ = nbrs.kneighbors(coords)
    curvature = np.var(distances, axis=1, keepdims=True)
    
    return normals, curvature

# ==========================================
# MODELO
# ==========================================

def load_model(model_path, num_classes=2, model_type='pointnet2'):
    """
    Carrega modelo treinado
    """
    if model_type == 'pointnet2':
        # PointNet++ Advanced (11 features)
        model = PointNet2SemSeg(
            num_classes=num_classes,
            input_channels=11  # xyz(3) + rgb(3) + intensity(1) + normals(3) + curvature(1)
        ).to(device)
    else:
        # PointNet Vanilla (7 features)
        from pointnet import PointNet
        model = PointNet(
            MLP_1=[128, 256, 512],
            MLP_2=[512, 1024, 2048],
            MLP_3=[1024, 512, 256],
            n_classes=num_classes,
            input_feat=7,
            subsample_size=8192,
            device=device
        ).to(device)
    
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    print(f"✓ Modelo carregado: {model_path}")
    return model

def prepare_data_for_inference(file_path, num_points=8192, use_geometric_features=True):
    """
    Prepara dados para inferência
    """
    # Carregar PLY
    coords, rgb_features, intensity = load_ply_for_inference(file_path)
    
    # Normalizar coordenadas
    coords = normalize_point_cloud(coords)
    
    # Ajustar número de pontos
    n_points = coords.shape[0]
    if n_points > num_points:
        choice = np.random.choice(n_points, num_points, replace=False)
        coords = coords[choice, :]
        rgb_features = rgb_features[choice, :]
        intensity = intensity[choice, :]
    elif n_points < num_points:
        choice = np.random.choice(n_points, num_points - n_points, replace=True)
        coords = np.concatenate([coords, coords[choice, :]], axis=0)
        rgb_features = np.concatenate([rgb_features, rgb_features[choice, :]], axis=0)
        intensity = np.concatenate([intensity, intensity[choice, :]], axis=0)
    
    # Montar features
    features = [coords, rgb_features, intensity]
    
    if use_geometric_features:
        # Adicionar normais e curvatura (para PointNet++ Advanced)
        normals, curvature = compute_geometric_features(coords)
        if normals.shape[0] == coords.shape[0]:
            features.extend([normals, curvature])
    
    # Concatenar todas as features
    all_features = np.concatenate(features, axis=1)  # (N, total_features)
    
    # Converter para tensor
    all_features = torch.from_numpy(all_features).float()
    all_features = all_features.unsqueeze(0)  # (1, N, features)
    
    # Transpor para (1, features, N) se necessário (depende da arquitetura)
    all_features = all_features.permute(0, 2, 1)  # (1, features, N)
    
    return all_features, coords

# ==========================================
# INFERÊNCIA
# ==========================================

def run_segmentation_inference(model, features):
    """
    Executa inferência para segmentação semântica
    Retorna label por ponto
    """
    features = features.to(device)
    
    with torch.no_grad():
        out = model(features)  # (1, num_classes, N)
        pred_labels = torch.argmax(out, dim=1)  # (1, N)
        pred_labels = pred_labels.squeeze(0).cpu().numpy()  # (N,)
    
    return pred_labels

def run_classification_inference(model, features):
    """
    Executa inferência para classificação
    Retorna label único para toda a nuvem
    """
    features = features.to(device)
    
    with torch.no_grad():
        out = model(features)  # (1, num_classes)
        _, pred_label = torch.max(out, dim=1)
        pred_label = pred_label.cpu().numpy()[0]
    
    return pred_label

# ==========================================
# VISUALIZAÇÃO
# ==========================================

def visualize_segmentation(coords, pred_labels, class_names):
    """
    Visualiza resultado de segmentação semântica
    """
    # Cores por classe
    class_colors = {
        0: [0.2, 0.6, 1.0],   # Azul claro para 'Equipamento'
        1: [1.0, 0.2, 0.2],   # Vermelho para 'Flange'
    }
    
    colors = np.array([class_colors.get(label, [0.5, 0.5, 0.5]) for label in pred_labels])
    
    # Criar visualização
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(coords)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    
    # Estatísticas
    unique, counts = np.unique(pred_labels, return_counts=True)
    print("\nDistribuição de classes:")
    for label, count in zip(unique, counts):
        percentage = (count / len(pred_labels)) * 100
        print(f"  {class_names[label]}: {count} pontos ({percentage:.2f}%)")
    
    o3d.visualization.draw_geometries([pcd], window_name="Segmentation Result")

def visualize_classification(coords, prediction, class_names):
    """
    Visualiza resultado de classificação
    """
    class_colors = {
        0: [0, 0, 1],   # Azul para 'Equipamento'
        1: [1, 0, 0],   # Vermelho para 'Flange'
    }
    
    color = class_colors.get(prediction, [0.5, 0.5, 0.5])
    colors = np.tile(color, (coords.shape[0], 1))
    
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(coords)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    
    print(f"\nPredicted Class: {class_names[prediction]}")
    o3d.visualization.draw_geometries([pcd], window_name="Classification Result")

# ==========================================
# MAIN
# ==========================================

def main():
    # Configurações
    model_path = 'aaa_model.pth'  # Modelo PointNet++ Advanced
    class_names = ['Equipamento', 'Flange']
    task = 'segmentation'  # 'segmentation' ou 'classification'
    
    # Carregar modelo
    model = load_model(model_path, num_classes=len(class_names), model_type='pointnet2')
    
    # Diretório de inferência
    inference_dir = './DATA/inference'
    
    if not os.path.exists(inference_dir):
        print(f"⚠ Diretório não encontrado: {inference_dir}")
        print("Criando diretório de exemplo...")
        os.makedirs(inference_dir, exist_ok=True)
        return
    
    # Processar arquivos
    ply_files = [f for f in os.listdir(inference_dir) if f.endswith('.ply')]
    
    if len(ply_files) == 0:
        print(f"⚠ Nenhum arquivo .ply encontrado em {inference_dir}")
        return
    
    print(f"\n{'='*60}")
    print(f"Processando {len(ply_files)} arquivos...")
    print(f"{'='*60}\n")
    
    for file_name in ply_files:
        file_path = os.path.join(inference_dir, file_name)
        print(f"\n📄 Arquivo: {file_name}")
        
        try:
            # Preparar dados
            features, coords = prepare_data_for_inference(
                file_path, 
                num_points=8192,
                use_geometric_features=True
            )
            
            # Inferência
            if task == 'segmentation':
                pred_labels = run_segmentation_inference(model, features)
                visualize_segmentation(coords, pred_labels, class_names)
            else:
                prediction = run_classification_inference(model, features)
                visualize_classification(coords, prediction, class_names)
                
        except Exception as e:
            print(f"❌ Erro ao processar {file_name}: {str(e)}")
            continue

if __name__ == "__main__":
    main()
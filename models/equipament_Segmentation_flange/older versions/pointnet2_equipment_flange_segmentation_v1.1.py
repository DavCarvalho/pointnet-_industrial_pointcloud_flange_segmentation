"""
Inference and Visualization Script
Updated to use current PointNet++ architecture
"""

import os
import numpy as np
import torch
import open3d as o3d
from plyfile import PlyData
import torch.nn as nn

# CORRECTION: Import from correct file
# Option 1: If using vanilla PointNet
# from pointnet import PointNet, cloud_loader, preprocess_point_cloud

# Option 2: If using PointNet++ advanced (recommended)
from models.equipament_Segmentation_flange.pointnet2_equipment_flange_segmentation import PointNet2SemSeg, read_ply_with_labels_plyfile

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using Device:", device)

# ==========================================
# AUXILIARY FUNCTIONS
# ==========================================

def load_ply_for_inference(filename):
    """
    Load PLY file for inference
    Adapted from read_ply_with_labels_plyfile
    """
    plydata = PlyData.read(filename)
    vertex_data = plydata['vertex'].data

    # Extract coordinates
    x = vertex_data['x']
    y = vertex_data['y']
    z = vertex_data['z']
    
    # Extract optional features
    r = vertex_data['red'] if 'red' in vertex_data.dtype.names else np.zeros_like(x, dtype=np.float32)
    g = vertex_data['green'] if 'green' in vertex_data.dtype.names else np.zeros_like(x, dtype=np.float32)
    b = vertex_data['blue'] if 'blue' in vertex_data.dtype.names else np.zeros_like(x, dtype=np.float32)
    intensity = vertex_data['intensity'] if 'intensity' in vertex_data.dtype.names else np.zeros_like(x, dtype=np.float32)

    # Assemble coordinate and feature arrays
    coords = np.vstack((x, y, z)).T  # (N, 3)
    
    # Additional features (RGB + Intensity)
    features = np.vstack((r, g, b)).T / 255.0  # Normalize RGB
    intensity = intensity[:, np.newaxis]  # (N, 1)
    
    return coords, features, intensity

def normalize_point_cloud(coords):
    """
    Normalize point cloud to zero-mean and unit scale
    """
    # Center
    centroid = np.mean(coords, axis=0, keepdims=True)
    coords = coords - centroid
    
    # Scale to unit sphere
    max_dist = np.max(np.sqrt(np.sum(coords**2, axis=1)))
    if max_dist > 0:
        coords = coords / max_dist
    
    return coords

def compute_geometric_features(coords):
    """
    Compute geometric features (normals, curvature)
    Required for PointNet++ Advanced
    """
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(coords)
    
    # Estimate normals
    pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30)
    )
    normals = np.asarray(pcd.normals)
    
    # Compute curvature (simplified)
    # Here you can use the same logic from a_a_a.py
    from sklearn.neighbors import NearestNeighbors
    nbrs = NearestNeighbors(n_neighbors=10, algorithm='kd_tree').fit(coords)
    distances, _ = nbrs.kneighbors(coords)
    curvature = np.var(distances, axis=1, keepdims=True)
    
    return normals, curvature

# ==========================================
# MODEL
# ==========================================

def load_model(model_path, num_classes=2, model_type='pointnet2'):
    """
    Load trained model
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
    print(f"Model loaded: {model_path}")
    return model

def prepare_data_for_inference(file_path, num_points=8192, use_geometric_features=True):
    """
    Prepare data for inference
    """
    # Load PLY
    coords, rgb_features, intensity = load_ply_for_inference(file_path)
    
    # Normalize coordinates
    coords = normalize_point_cloud(coords)
    
    # Adjust number of points
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
    
    # Assemble features
    features = [coords, rgb_features, intensity]
    
    if use_geometric_features:
        # Add normals and curvature (for PointNet++ Advanced)
        normals, curvature = compute_geometric_features(coords)
        if normals.shape[0] == coords.shape[0]:
            features.extend([normals, curvature])
    
    # Concatenate all features
    all_features = np.concatenate(features, axis=1)  # (N, total_features)
    
    # Convert to tensor
    all_features = torch.from_numpy(all_features).float()
    all_features = all_features.unsqueeze(0)  # (1, N, features)
    
    # Transpose to (1, features, N) if necessary (depends on architecture)
    all_features = all_features.permute(0, 2, 1)  # (1, features, N)
    
    return all_features, coords

# ==========================================
# INFERENCE
# ==========================================

def run_segmentation_inference(model, features):
    """
    Execute inference for semantic segmentation
    Returns label per point
    """
    features = features.to(device)
    
    with torch.no_grad():
        out = model(features)  # (1, num_classes, N)
        pred_labels = torch.argmax(out, dim=1)  # (1, N)
        pred_labels = pred_labels.squeeze(0).cpu().numpy()  # (N,)
    
    return pred_labels

def run_classification_inference(model, features):
    """
    Execute inference for classification
    Returns single label for entire cloud
    """
    features = features.to(device)
    
    with torch.no_grad():
        out = model(features)  # (1, num_classes)
        _, pred_label = torch.max(out, dim=1)
        pred_label = pred_label.cpu().numpy()[0]
    
    return pred_label

# ==========================================
# VISUALIZATION
# ==========================================

def visualize_segmentation(coords, pred_labels, class_names):
    """
    Visualize semantic segmentation result
    """
    # Colors per class
    class_colors = {
        0: [0.2, 0.6, 1.0],   # Light blue for 'Equipment'
        1: [1.0, 0.2, 0.2],   # Red for 'Flange'
    }
    
    colors = np.array([class_colors.get(label, [0.5, 0.5, 0.5]) for label in pred_labels])
    
    # Create visualization
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(coords)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    
    # Statistics
    unique, counts = np.unique(pred_labels, return_counts=True)
    print("\nClass distribution:")
    for label, count in zip(unique, counts):
        percentage = (count / len(pred_labels)) * 100
        print(f"  {class_names[label]}: {count} points ({percentage:.2f}%)")
    
    o3d.visualization.draw_geometries([pcd], window_name="Segmentation Result")

def visualize_classification(coords, prediction, class_names):
    """
    Visualize classification result
    """
    class_colors = {
        0: [0, 0, 1],   # Blue for 'Equipment'
        1: [1, 0, 0],   # Red for 'Flange'
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
    # Configuration
    model_path = 'aaa_model.pth'  # PointNet++ Advanced model
    class_names = ['Equipment', 'Flange']
    task = 'segmentation'  # 'segmentation' or 'classification'
    
    # Load model
    model = load_model(model_path, num_classes=len(class_names), model_type='pointnet2')
    
    # Inference directory
    inference_dir = './DATA/inference'
    
    if not os.path.exists(inference_dir):
        print(f"Directory not found: {inference_dir}")
        print("Creating example directory...")
        os.makedirs(inference_dir, exist_ok=True)
        return
    
    # Process files
    ply_files = [f for f in os.listdir(inference_dir) if f.endswith('.ply')]
    
    if len(ply_files) == 0:
        print(f"No .ply files found in {inference_dir}")
        return
    
    print(f"\n{'='*60}")
    print(f"Processing {len(ply_files)} files...")
    print(f"{'='*60}\n")
    
    for file_name in ply_files:
        file_path = os.path.join(inference_dir, file_name)
        print(f"\nFile: {file_name}")
        
        try:
            # Prepare data
            features, coords = prepare_data_for_inference(
                file_path, 
                num_points=8192,
                use_geometric_features=True
            )
            
            # Inference
            if task == 'segmentation':
                pred_labels = run_segmentation_inference(model, features)
                visualize_segmentation(coords, pred_labels, class_names)
            else:
                prediction = run_classification_inference(model, features)
                visualize_classification(coords, prediction, class_names)
                
        except Exception as e:
            print(f"Error processing {file_name}: {str(e)}")
            continue

if __name__ == "__main__":
    main()
import os
import numpy as np
import pandas as pd
from plyfile import PlyData, PlyElement
import hdbscan
import open3d as o3d
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.model_selection import train_test_split
from tqdm.auto import tqdm
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using Device: ", device)

"""
FLANGE CLASSIFICATION MODEL USING POINTNET++

This model trains on segmented point cloud patches:
- FLANGES: Individual flange instances extracted from industrial equipment
- NON-FLANGES: Random patches from other equipment parts (pipes, valves, structures)

The model learns to classify whether a given point cloud patch contains a flange or not.
This is a binary classification task where the output is:
- Class 0: Non-Flange (other industrial equipment parts)
- Class 1: Flange (flange instances)

The training data consists of small point cloud patches (2048 points) extracted from
larger industrial equipment scans, allowing the model to learn local geometric features
that distinguish flanges from other components.
"""

# Function to load data from a PLY file
def load_ply(file_path):
    plydata = PlyData.read(file_path)
    vertex = plydata['vertex']
    x = vertex['x']
    y = vertex['y']
    z = vertex['z']
    if 'scalar_Classification' in vertex.data.dtype.names:
        labels = vertex['scalar_Classification']
    else:
        labels = None
    coords = np.vstack((x, y, z)).astype(np.float32).T  # Shape (N, 3)
    return coords, labels

# Function to extract flanges and non-flanges
def extract_flanges(coords, labels):
    if labels is not None:
        flange_points = coords[labels == 1]
        non_flange_points = coords[labels != 1]
    else:
        # If no labels, consider all points as non-flanges
        flange_points = np.array([])
        non_flange_points = coords
    return flange_points, non_flange_points

# Function to cluster flanges using HDBSCAN
def cluster_flanges_hdbscan(flange_points, min_cluster_size=5):
    clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size)
    clusters = clusterer.fit_predict(flange_points)
    return clusters

# Function to save points to PLY files
def save_ply(equipment_name, category, points, output_dir, flange_info_list=None, min_points_flange=500, min_points_non_flange=15000, max_instances_non_flange=15):
    category_dir = os.path.join(output_dir, category)
    os.makedirs(category_dir, exist_ok=True)

    if category == 'flanges' and isinstance(points, dict) and 'clusters' in points:
        unique_clusters = np.unique(points['clusters'])
        for cluster in unique_clusters:
            if cluster == -1:
                continue
            cluster_points = points['coords'][points['clusters'] == cluster]
            num_points = len(cluster_points)

            # Apply minimum size filter for flanges
            if num_points < min_points_flange:
                print(f"Cluster {cluster} ignored for having only {num_points} points")
                continue

            centroid = cluster_points.mean(axis=0)
            flange_id = f"{equipment_name}_flange_{cluster+1}"
            ply_filename = f"{flange_id}.ply"
            ply_output_path = os.path.join(category_dir, ply_filename)

            # Save flange points to PLY file
            vertex_element = np.array(
                [(p[0], p[1], p[2]) for p in cluster_points],
                dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4')]
            )
            el = PlyElement.describe(vertex_element, 'vertex')
            PlyData([el], text=True).write(ply_output_path)

            # Add information to dataframe
            flange_info = {
                'Equipamento': equipment_name,
                'Flange_ID': flange_id,
                'Num_Pontos': num_points,
                'Centroid_X': centroid[0],
                'Centroid_Y': centroid[1],
                'Centroid_Z': centroid[2],
                'Arquivo_PLY': os.path.join('flanges', ply_filename),
                'Label': 1  # Flange
            }
            flange_info_list.append(flange_info)
            print(f"Saved {flange_id} with {num_points} points")
    else:
        # For non-flanges, save a limited number of instances with random sampling
        if isinstance(points, np.ndarray):
            num_points_total = len(points)
            if num_points_total < min_points_non_flange:
                print(f"Equipment {equipment_name} ignored for having only {num_points_total} points as non-flange")
                return

            # Limit the number of non-flange instances
            num_instances = min(max_instances_non_flange, num_points_total // min_points_non_flange)
            for i in range(num_instances):
                # Randomly sample non-flange points
                sample_indices = np.random.choice(num_points_total, min_points_non_flange, replace=False)
                instance_points = points[sample_indices]

                instance_id = f"{equipment_name}_non_flange_{i+1}"
                ply_filename = f"{instance_id}.ply"
                ply_output_path = os.path.join(category_dir, ply_filename)

                # Save non-flange points to PLY file
                vertex_element = np.array(
                    [(p[0], p[1], p[2]) for p in instance_points],
                    dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4')]
                )
                el = PlyElement.describe(vertex_element, 'vertex')
                PlyData([el], text=True).write(ply_output_path)

                # Add information to dataframe
                centroid = instance_points.mean(axis=0)
                flange_info = {
                    'Equipamento': equipment_name,
                    'Flange_ID': instance_id,
                    'Num_Pontos': len(instance_points),
                    'Centroid_X': centroid[0],
                    'Centroid_Y': centroid[1],
                    'Centroid_Z': centroid[2],
                    'Arquivo_PLY': os.path.join('non_flanges', ply_filename),
                    'Label': 0  # Non-flange
                }
                flange_info_list.append(flange_info)
                print(f"Saved {instance_id} with {len(instance_points)} points")
        else:
            print("Unrecognized 'points' format for saving PLY.")

# Function to process each equipment (PLY file)
def process_equipment(ply_file, project_dir, output_dir, flange_info_list, min_points_flange=500, min_points_non_flange=15000, max_non_flange_instances=15):
    ply_path = os.path.join(project_dir, ply_file)
    equipment_name = os.path.splitext(ply_file)[0]

    coords, labels = load_ply(ply_path)
    flange_points, non_flange_points = extract_flanges(coords, labels)

    # Process flanges
    if len(flange_points) > 0:
        clusters = cluster_flanges_hdbscan(flange_points, min_cluster_size=5)
        num_clusters = len(np.unique(clusters)) - (1 if -1 in clusters else 0)
        print(f"HDBSCAN found {num_clusters} flanges in equipment {equipment_name}")

        if num_clusters > 0:
            # Prepare data to save flanges
            flange_data = {
                'coords': flange_points,
                'clusters': clusters
            }

            # Save flanges (applying min_points_flange filter)
            save_ply(
                equipment_name,
                'flanges',
                flange_data,
                output_dir,
                flange_info_list,
                min_points_flange=min_points_flange
            )
    else:
        print(f"Warning: No flanges found in equipment {equipment_name}")

    # Process non-flanges
    if len(non_flange_points) >= min_points_non_flange:
        # Save a limited number of non-flange instances
        save_ply(
            equipment_name,
            'non_flanges',
            non_flange_points,
            output_dir,
            flange_info_list,
            min_points_flange=min_points_flange,
            min_points_non_flange=min_points_non_flange,
            max_instances_non_flange=max_non_flange_instances
        )
    else:
        print(f"Warning: Not enough points to split non-flanges in equipment {equipment_name}")

# Main function to process the data
def main_data_processing():
    project_dir = "./DATA/"
    output_dir = "./FLANGE_DATASET/"
    os.makedirs(output_dir, exist_ok=True)
    ply_files = [f for f in os.listdir(project_dir) if f.endswith('.ply')]

    flange_info_list = []

    for ply_file in ply_files:
        process_equipment(
            ply_file,
            project_dir,
            output_dir,
            flange_info_list,
            min_points_flange=500,
            min_points_non_flange=15000,
            max_non_flange_instances=15
        )

    # Create the dataframe
    flange_df = pd.DataFrame(flange_info_list)

    # Check class distribution
    print("Class Distribution:")
    print(flange_df['Label'].value_counts())

    # Save dataframe to CSV file
    flange_df.to_csv(os.path.join(output_dir, 'flange_info.csv'), index=False)
    print("Flange and non-flange information saved to flange_info.csv")

# Dataset class to load point clouds
class PointCloudDataset(Dataset):
    def __init__(self, df, num_points=2048, transform=None):
        self.df = df
        self.num_points = num_points
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        label = row['Label']

        ply_path = os.path.join("./FLANGE_DATASET", row['Arquivo_PLY'])

        if not os.path.exists(ply_path):
            raise FileNotFoundError(f"PLY file not found: {ply_path}")

        plydata = PlyData.read(ply_path)
        vertex = plydata['vertex']
        coords = np.vstack((vertex['x'], vertex['y'], vertex['z'])).astype(np.float32).T  # Shape (N, 3)

        # Normalization
        coords = normalize_point_cloud(coords)

        # Apply data augmentation only for "Flange" class during training
        if self.transform and label == 1:
            coords = self.transform(coords)

        # Adjust number of points
        if coords.shape[0] > self.num_points:
            choice = np.random.choice(coords.shape[0], self.num_points, replace=False)
            coords = coords[choice, :]
        elif coords.shape[0] < self.num_points:
            choice = np.random.choice(coords.shape[0], self.num_points - coords.shape[0], replace=True)
            coords = np.concatenate([coords, coords[choice, :]], axis=0)

        # Verify shape
        assert coords.shape == (self.num_points, 3), f"Incorrect shape: {coords.shape}"

        return torch.from_numpy(coords), label

# Function to normalize point clouds
def normalize_point_cloud(coords):
    centroid = np.mean(coords, axis=0, keepdims=True)  # Mean along points
    coords = coords - centroid
    furthest_distance = np.max(np.sqrt(np.sum(coords ** 2, axis=1)))  # Maximum distance from origin
    coords = coords / furthest_distance
    return coords

# Data augmentation function
def data_augmentation(coords):
    # Jittering
    coords += np.random.normal(0, 0.02, size=coords.shape)
    # Rotation
    theta = np.random.uniform(0, 2 * np.pi)
    rotation_matrix = np.array([
        [np.cos(theta), -np.sin(theta), 0],
        [np.sin(theta),  np.cos(theta), 0],
        [0, 0, 1]
    ])
    coords = coords @ rotation_matrix
    # Scaling
    scale = np.random.uniform(0.8, 1.2)
    coords *= scale
    return coords

# Function to prepare dataset and split into train, validation, and test
def prepare_dataset():
    data_dir = "./FLANGE_DATASET/"

    # Read the CSV generated by main_data_processing
    df = pd.read_csv(os.path.join(data_dir, 'flange_info.csv'))

    # Check if both classes have enough samples
    class_counts = df['Label'].value_counts()
    print("Class Counts Before Split:")
    print(class_counts)

    if class_counts.min() < 2:
        raise ValueError("One of the classes has less than 2 samples. Cannot perform stratification.")

    # Split the data, stratifying by 'Label' column
    train_df, temp_df = train_test_split(
        df,
        test_size=0.3,
        stratify=df['Label'],
        random_state=42
    )
    valid_df, test_df = train_test_split(
        temp_df,
        test_size=0.5,
        stratify=temp_df['Label'],
        random_state=42
    )

    print("Class Distribution in Training Set:")
    print(train_df['Label'].value_counts())
    print("Class Distribution in Validation Set:")
    print(valid_df['Label'].value_counts())
    print("Class Distribution in Test Set:")
    print(test_df['Label'].value_counts())

    return train_df.reset_index(drop=True), valid_df.reset_index(drop=True), test_df.reset_index(drop=True)

# Collate function for DataLoader
def cloud_collate(batch):
    clouds, labels = list(zip(*batch))
    clouds = torch.stack(clouds, dim=0)
    labels = torch.tensor(labels, dtype=torch.long)
    return clouds, labels

# Helper functions for PointNet++
def square_distance(src, dst):
    B, N, C = src.shape
    _, M, _ = dst.shape
    dist = -2 * torch.matmul(src, dst.transpose(1, 2))
    dist += torch.sum(src ** 2, -1).unsqueeze(-1)
    dist += torch.sum(dst ** 2, -1).unsqueeze(1)
    return dist

def index_points(points, idx):
    B = points.shape[0]
    batch_indices = torch.arange(B, dtype=torch.long).to(points.device).view(B, *((1,)*(idx.dim()-1)))
    new_points = points[batch_indices, idx, :]
    return new_points

def farthest_point_sample(xyz, npoint):
    device = xyz.device
    B, N, C = xyz.shape
    npoint = min(npoint, N)
    centroids = torch.zeros(B, npoint, dtype=torch.long).to(device)
    distance = torch.ones(B, N).to(device) * 1e10
    farthest = torch.randint(0, N, (B,), dtype=torch.long).to(device)
    batch_indices = torch.arange(B, dtype=torch.long).to(device)
    for i in range(npoint):
        centroids[:, i] = farthest
        centroid = xyz[batch_indices, farthest, :].view(B, 1, C)
        dist = torch.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = torch.max(distance, -1)[1]
    return centroids

def sample_and_group_all(xyz, points):
    device = xyz.device
    B, N, C = xyz.shape
    new_xyz = torch.zeros(B, 1, C).to(device)
    grouped_xyz = xyz.view(B, 1, N, C)
    if points is not None:
        grouped_points = points.view(B, 1, N, -1)
        new_points = torch.cat([grouped_xyz, grouped_points], dim=-1)
    else:
        new_points = grouped_xyz
    return new_xyz, new_points

def sample_and_group(npoint, radius, nsample, xyz, points):
    B, N, C = xyz.shape
    if npoint is None:
        npoint = N
    else:
        npoint = min(npoint, N)
    S = npoint
    fps_idx = farthest_point_sample(xyz, npoint)
    new_xyz = index_points(xyz, fps_idx)
    sqrdists = square_distance(new_xyz, xyz)
    nsample = min(nsample, N)
    group_idx = sqrdists.argsort()[:, :, :nsample]
    grouped_xyz = index_points(xyz, group_idx)
    grouped_xyz_norm = grouped_xyz - new_xyz.unsqueeze(2)
    if points is not None:
        grouped_points = index_points(points, group_idx)
        new_points = torch.cat([grouped_xyz_norm, grouped_points], dim=-1)
    else:
        new_points = grouped_xyz_norm
    return new_xyz, new_points

# PointNet++ Set Abstraction layers
class PointNetSetAbstraction(nn.Module):
    def __init__(self, npoint, radius, nsample, in_channel, mlp):
        super(PointNetSetAbstraction, self).__init__()
        self.npoint = npoint
        self.radius = radius
        self.nsample = nsample
        last_channel = in_channel
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        for out_channel in mlp:
            self.mlp_convs.append(nn.Conv2d(last_channel, out_channel, 1))
            self.mlp_bns.append(nn.BatchNorm2d(out_channel))
            last_channel = out_channel

    def forward(self, xyz, points):
        if self.npoint is not None:
            new_xyz, new_points = sample_and_group(self.npoint, self.radius, self.nsample, xyz, points)
        else:
            new_xyz, new_points = sample_and_group_all(xyz, points)
        new_points = new_points.permute(0, 3, 1, 2)
        for i, conv in enumerate(self.mlp_convs):
            bn = self.mlp_bns[i]
            new_points = F.relu(bn(conv(new_points)))
        new_points = torch.max(new_points, -1)[0]
        new_points = new_points.permute(0, 2, 1)
        return new_xyz, new_points

# PointNet++ Classification model
class PointNet2Classifier(nn.Module):
    def __init__(self, num_classes=2):
        super(PointNet2Classifier, self).__init__()
        self.sa1 = PointNetSetAbstraction(npoint=512, radius=0.2, nsample=32, in_channel=3, mlp=[64, 64, 128])
        self.sa2 = PointNetSetAbstraction(npoint=128, radius=0.4, nsample=64, in_channel=128+3, mlp=[128, 128, 256])
        self.sa3 = PointNetSetAbstraction(npoint=None, radius=None, nsample=None, in_channel=256+3, mlp=[256, 512, 1024])
        self.fc1 = nn.Linear(1024, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.drop1 = nn.Dropout(0.4)
        self.fc2 = nn.Linear(512, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.drop2 = nn.Dropout(0.4)
        self.fc3 = nn.Linear(256, num_classes)

    def forward(self, xyz):
        B, N, C = xyz.shape
        points = None
        l1_xyz, l1_points = self.sa1(xyz, points)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)
        x = l3_points.view(B, -1)
        x = self.drop1(F.relu(self.bn1(self.fc1(x))))
        x = self.drop2(F.relu(self.bn2(self.fc2(x))))
        x = self.fc3(x)
        return x

# Function to train the model
def train_model():
    train_df, valid_df, test_df = prepare_dataset()
    num_points = 2048

    # Apply data augmentation only on training set
    train_dataset = PointCloudDataset(train_df, num_points=num_points, transform=data_augmentation)
    valid_dataset = PointCloudDataset(valid_df, num_points=num_points)
    test_dataset = PointCloudDataset(test_df, num_points=num_points)

    # Calculate class weights
    class_counts = train_df['Label'].value_counts().sort_index()
    class_weights = 1. / class_counts
    samples_weight = train_df['Label'].apply(lambda x: class_weights[x]).values
    samples_weight = torch.from_numpy(samples_weight).float()
    sampler = WeightedRandomSampler(samples_weight, len(samples_weight))

    batch_size = 16
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=4,
        collate_fn=cloud_collate
    )
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        collate_fn=cloud_collate
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        collate_fn=cloud_collate
    )

    num_epochs = 200
    learning_rate = 0.0005
    n_classes = 2

    model = PointNet2Classifier(num_classes=n_classes).to(device)

    # Define loss function with class weights
    weight_tensor = torch.tensor([1.0, class_counts[0]/class_counts[1]], dtype=torch.float32).to(device)
    criterion = nn.CrossEntropyLoss(weight=weight_tensor)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)

    best_acc = 0.0
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, device)
        valid_loss, valid_acc = eval_epoch(model, valid_loader, criterion, device)
        scheduler.step()

        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"Valid Loss: {valid_loss:.4f}, Valid Acc: {valid_acc:.2f}%")

        if valid_acc > best_acc:
            best_acc = valid_acc
            torch.save(model.state_dict(), 'best_pointnet2_model.pth')
            print("Model saved!")

    # Load best model and evaluate on test set
    model.load_state_dict(torch.load('best_pointnet2_model.pth'))
    test_loss, test_acc = eval_epoch(model, test_loader, criterion, device)
    print(f"\nTest Loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}%")

    # Plot confusion matrix
    class_names = ['Non-Flange', 'Flange']
    plot_confusion_matrix(model, test_loader, device, class_names)

    return model

# Functions to train and evaluate each epoch
def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for clouds, labels in tqdm(loader, desc='Train'):
        clouds, labels = clouds.to(device).float(), labels.to(device)
        optimizer.zero_grad()
        outputs = model(clouds)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * labels.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    return running_loss / total, 100. * correct / total

def eval_epoch(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for clouds, labels in tqdm(loader, desc='Eval'):
            clouds, labels = clouds.to(device).float(), labels.to(device)
            outputs = model(clouds)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * labels.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    return running_loss / total, 100. * correct / total

# Function to plot confusion matrix
def plot_confusion_matrix(model, loader, device, class_names):
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for clouds, labels in loader:
            clouds, labels = clouds.to(device).float(), labels.to(device)
            outputs = model(clouds)
            _, predicted = outputs.max(1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    cm = confusion_matrix(all_labels, all_preds)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot()
    plt.show()
    print(classification_report(all_labels, all_preds, target_names=class_names))

# Function to visualize predictions
def visualize_prediction(model, dataset, idx):
    model.eval()
    cloud, label = dataset[idx]
    cloud = cloud.unsqueeze(0).to(device).float()

    with torch.no_grad():
        output = model(cloud)
        _, predicted = output.max(1)

    predicted = predicted.item()
    cloud = cloud.cpu().numpy().squeeze()  # (N, 3)

    # Use Open3D for visualization
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(cloud.astype(np.float64))

    if predicted == 1:
        pcd.paint_uniform_color([1, 0, 0])  # Red for flange
        title = 'Predicted: Flange'
    else:
        pcd.paint_uniform_color([0, 0, 1])  # Blue for non-flange
        title = 'Predicted: Non-Flange'

    print(f"{title} (True Label: {'Flange' if label == 1 else 'Non-Flange'})")
    o3d.visualization.draw_geometries([pcd])

if __name__ == "__main__":
    # Data processing
    #main_data_processing()

    # Model training
    best_model = train_model()

    # Visualize some predictions
    train_df, valid_df, test_df = prepare_dataset()
    num_points = 2048
    test_dataset = PointCloudDataset(test_df, num_points=num_points)

    # Visualize some predictions
    num_visualizations = min(5, len(test_dataset))
    for i in range(num_visualizations):
        visualize_prediction(best_model, test_dataset, i)
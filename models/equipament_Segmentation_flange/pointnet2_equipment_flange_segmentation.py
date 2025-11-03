# Necessary Imports

import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
import torchnet as tnt
import torch.optim as optim
from sklearn.metrics import confusion_matrix, classification_report
import copy
from glob import glob
import os
import functools
from tqdm.auto import tqdm
import time
import csv
from plyfile import PlyData
import random
import logging
import open3d as o3d
from sklearn.neighbors import KDTree  # Added to estimate curvature

# Debug Settings
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

# Logging Configuration
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Check Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using Device:", device)

# Define Training Parameters

class Args:
    pass

args = Args()
class_names = ['equipment', 'flange']
args.n_class = len(class_names)
args.n_epoch = 5
args.subsample_size = 8192  # Increased to allow more points per sample
args.batch_size = 2
args.input_feats = 'xyzrgbi'  # We will update the number of input channels later
args.lr = 1e-3
args.wd = 1e-5
args.cuda = device.type == 'cuda'

# Data Paths Configuration

project_dir = "./DATA/"
pointcloud_train_files = glob(os.path.join(project_dir, "train/*.ply"))
pointcloud_test_files = glob(os.path.join(project_dir, "test/*.ply"))

print(f"{len(pointcloud_train_files)} training files, {len(pointcloud_test_files)} test files")
print(torch.cuda.is_available())
print(torch.cuda.device_count())
print(torch.cuda.get_device_name(0))

# PLY File Reader

def read_ply_with_labels_plyfile(filename):
    plydata = PlyData.read(filename)
    vertex_data = plydata['vertex'].data

    x = vertex_data['x']
    y = vertex_data['y']
    z = vertex_data['z']
    r = vertex_data['red'] if 'red' in vertex_data.dtype.names else np.zeros_like(x, dtype=np.float32)
    g = vertex_data['green'] if 'green' in vertex_data.dtype.names else np.zeros_like(x, dtype=np.float32)
    b = vertex_data['blue'] if 'blue' in vertex_data.dtype.names else np.zeros_like(x, dtype=np.float32)
    intensity = vertex_data['intensity'] if 'intensity' in vertex_data.dtype.names else np.zeros_like(x, dtype=np.float32)
    labels = vertex_data['scalar_Classification'] if 'scalar_Classification' in vertex_data.dtype.names else np.zeros_like(x, dtype=np.int32)

    labels = labels.astype(np.int32)

    # Consistency check
    if len(x) != len(labels):
        raise ValueError(f"Error in file {filename}: number of points ({len(x)}) does not match number of labels ({len(labels)}).")

    data = {
        'x': x,
        'y': y,
        'z': z,
        'r': r,
        'g': g,
        'b': b,
        'intensity': intensity,
        'scalar_Classification': labels
    }
    return data

# Data Augmentation Functions

def random_rotation(cloud, angle=None):
    if angle is None:
        angle = np.random.uniform(-np.pi, np.pi)
    rotation_matrix = np.array([
        [np.cos(angle), -np.sin(angle), 0],
        [np.sin(angle),  np.cos(angle), 0],
        [0, 0, 1]
    ])
    rotation_matrix = torch.from_numpy(rotation_matrix).float().to(cloud.device)
    cloud[:3, :] = rotation_matrix @ cloud[:3, :]
    return cloud

def random_scaling(cloud, scale_range=(0.8, 1.2)):
    scale = np.random.uniform(*scale_range)
    scaled_cloud = cloud.clone()
    scaled_cloud[:3, :] *= scale
    return scaled_cloud

def jitter(cloud, sigma=0.01, clip=0.05):
    jittered = cloud.clone()
    noise = torch.clamp(torch.randn_like(jittered[:3, :]) * sigma, -clip, clip)
    jittered[:3, :] += noise
    return jittered

def shift(cloud, shift_range=0.1):
    shifts = np.random.uniform(-shift_range, shift_range, 3)
    shifted_cloud = cloud.clone()
    shifted_cloud[:3, :] += torch.tensor(shifts, device=cloud.device).unsqueeze(1)
    return shifted_cloud

def augment_cloud(cloud, gt):
    cloud = random_rotation(cloud)
    cloud = random_scaling(cloud)
    cloud = jitter(cloud)
    cloud = shift(cloud)
    return cloud, gt

def augment_flange(cloud, gt):
    flange_indices = (gt == 1)
    if flange_indices.sum() == 0:
        return cloud, gt

    # Add more specific transformations for flanges
    rotation_angles = np.random.uniform(-np.pi/4, np.pi/4, 3)  # Rotation in 3 axes
    scale_factors = np.random.uniform(0.9, 1.1, 3)  # Non-uniform scale

    # Apply more robust transformations
    cloud_flange = cloud[:, flange_indices]
    for i in range(3):
        cloud_flange[i:i+1, :] *= scale_factors[i]

    # Apply rotations in 3 axes
    Rx = torch.tensor([[1, 0, 0],
                       [0, np.cos(rotation_angles[0]), -np.sin(rotation_angles[0])],
                       [0, np.sin(rotation_angles[0]),  np.cos(rotation_angles[0])]], dtype=torch.float32, device=cloud.device)
    Ry = torch.tensor([[ np.cos(rotation_angles[1]), 0, np.sin(rotation_angles[1])],
                       [0, 1, 0],
                       [-np.sin(rotation_angles[1]), 0, np.cos(rotation_angles[1])]], dtype=torch.float32, device=cloud.device)
    Rz = torch.tensor([[np.cos(rotation_angles[2]), -np.sin(rotation_angles[2]), 0],
                       [np.sin(rotation_angles[2]),  np.cos(rotation_angles[2]), 0],
                       [0, 0, 1]], dtype=torch.float32, device=cloud.device)
    rotation_matrix = Rz @ Ry @ Rx
    cloud_flange[:3, :] = rotation_matrix @ cloud_flange[:3, :]

    # Add Gaussian noise specific to edges
    edge_noise = torch.randn_like(cloud_flange[:3, :]) * 0.002
    cloud_flange[:3, :] += edge_noise

    cloud[:, flange_indices] = cloud_flange
    return cloud, gt

def cloud_collate(batch):
    clouds, labels = list(zip(*batch))
    return clouds, labels

# Function to estimate local curvature

def estimate_curvature(xyz):
    # xyz: numpy array of shape (3, N)
    xyz = xyz.T  # Shape (N, 3)
    N = xyz.shape[0]
    curvature = np.zeros(N)

    # Use KDTree for neighborhood
    tree = KDTree(xyz)
    k = 16  # Number of neighbors
    for i in range(N):
        idx = tree.query(xyz[i:i+1], k=k, return_distance=False)
        neighbors = xyz[idx[0]]  # Shape (k, 3)
        # Center the points
        neighbors_centered = neighbors - np.mean(neighbors, axis=0)
        cov = np.dot(neighbors_centered.T, neighbors_centered)
        eigvals, _ = np.linalg.eigh(cov)
        eigvals = np.sort(eigvals)
        curvature[i] = eigvals[0] / (np.sum(eigvals) + 1e-8)
    return curvature  # Shape (N,)

# Data Loading Function with Addition of Geometric Features

def cloud_loader(tile_name, features_used, file_type='ply', max_points=8192, augment=False):
    data = read_ply_with_labels_plyfile(tile_name)

    if 'scalar_Classification' not in data:
        raise ValueError(f"'scalar_Classification' property not found in {tile_name}")

    features = []

    # xyz coordinates
    xyz = np.vstack((data['x'], data['y'], data['z']))
    mean_f = np.mean(xyz, axis=1, keepdims=True)
    min_f = np.min(xyz, axis=1, keepdims=True)
    xyz[0] -= mean_f[0]
    xyz[1] -= mean_f[1]
    xyz[2] -= min_f[2]
    features.append(xyz)

    # Calculate normals
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz.T)
    pcd.estimate_normals()
    normals = np.asarray(pcd.normals).T  # Shape (3, N)
    features.append(normals)

    # Add local curvature
    curvature = estimate_curvature(xyz)  # Shape (N,)
    features.append(curvature[np.newaxis, :])  # Shape (1, N)

    # RGB
    if 'rgb' in features_used:
        if 'r' in data and 'g' in data and 'b' in data:
            rgb = np.vstack((data['r'], data['g'], data['b']))
            features.append(rgb)
        else:
            print(f"RGB columns not found in {tile_name}. Using zeros.")
            colors = np.zeros((3, len(data['x'])), dtype=np.float32)
            features.append(colors)

    # Intensity
    if 'i' in features_used:
        if 'intensity' in data:
            intensity = data['intensity']
            IQR = np.quantile(intensity, 0.75) - np.quantile(intensity, 0.25)
            if IQR != 0:
                n_intensity = ((intensity - np.median(intensity)) / IQR)
                n_intensity -= np.min(n_intensity)
            else:
                n_intensity = intensity - np.min(intensity)
            intensity = n_intensity[np.newaxis, :]
            features.append(intensity.astype(np.float32))
        else:
            print(f"'intensity' column not found in {tile_name}. Using zeros.")
            intensity = np.zeros((1, len(data['x'])), dtype=np.float32)
            features.append(intensity)

    gt = data['scalar_Classification'].astype(np.int32)

    # Map labels
    label_mapping = {0: 0, 1: 1}
    gt_mapped = np.array([label_mapping.get(label, -1) for label in gt])

    # Check for invalid labels
    if -1 in gt_mapped:
        invalid_labels = set(gt[gt_mapped == -1])
        raise ValueError(f"Found labels {invalid_labels} not in mapping in {tile_name}")

    # Normalize features
    for i in range(len(features)):
        features[i] = (features[i] - np.mean(features[i], axis=1, keepdims=True)) / (
                np.std(features[i], axis=1, keepdims=True) + 1e-8)

    cloud_data = np.vstack(features)

    cloud_data = torch.from_numpy(cloud_data).type(torch.float32)
    gt_mapped = torch.from_numpy(gt_mapped).long()

    # Implement stratified sampling
    flange_indices = (gt_mapped == 1).nonzero(as_tuple=False).squeeze()
    equipment_indices = (gt_mapped == 0).nonzero(as_tuple=False).squeeze()

    num_flange = flange_indices.shape[0]
    num_equipment = equipment_indices.shape[0]

    desired_flange_points = max(int(max_points * 0.4), 1)  # Ensure at least 20% flange points
    desired_equipment_points = max_points - desired_flange_points

    if num_flange == 0:
        # No flange points, sample only equipment
        sampled_equipment_indices = np.random.choice(equipment_indices.cpu(), max_points, replace=True)
        sampled_indices = sampled_equipment_indices
    else:
        sampled_flange_indices = np.random.choice(flange_indices.cpu(), desired_flange_points, replace=True)
        sampled_equipment_indices = np.random.choice(equipment_indices.cpu(), desired_equipment_points, replace=True)
        sampled_indices = np.concatenate([sampled_flange_indices, sampled_equipment_indices])
        np.random.shuffle(sampled_indices)

    cloud_data = cloud_data[:, sampled_indices]
    gt_mapped = gt_mapped[sampled_indices]

    if augment:
        cloud_data, gt_mapped = augment_cloud(cloud_data, gt_mapped)
        cloud_data, gt_mapped = augment_flange(cloud_data, gt_mapped)

    return cloud_data, gt_mapped

# Define Feature Usage and File Type

cloud_features = "xyzrgbi"  # Updated to include new features
file_type = 'ply'

# Maximum Number of Points to Load per File

max_points = args.subsample_size

# Define the number of input channels (features)
# Let's calculate the number of features that will be used
num_input_features = 0

# xyz coordinates (always used)
num_input_features += 3  # xyz

# Normals (always added)
num_input_features += 3  # normals

# Curvature (always added)
num_input_features += 1  # curvature

# RGB
if 'rgb' in cloud_features:
    num_input_features += 3

# Intensity
if 'i' in cloud_features:
    num_input_features += 1

args.n_input_feats = num_input_features

print(f"Number of input features: {args.n_input_feats}")

# Split Data into Train, Validation and Test

valid_ratio = 0.1  # 10% for validation
num_valid = int(len(pointcloud_train_files) * valid_ratio) + 2  # Add 2 equipment for validation
valid_index = np.random.choice(len(pointcloud_train_files), num_valid, replace=False)
valid_list = [pointcloud_train_files[i] for i in valid_index]
train_list = [pointcloud_train_files[i] for i in np.setdiff1d(list(range(len(pointcloud_train_files))), valid_index)]
test_list = pointcloud_test_files

print(f"{len(train_list)} training files, {len(valid_list)} validation files, {len(test_list)} test files")

# Create Datasets

train_set = tnt.dataset.ListDataset(
    train_list,
    functools.partial(
        cloud_loader,
        features_used=cloud_features,
        file_type=file_type,
        max_points=max_points,
        augment=True
    )
)

valid_set = tnt.dataset.ListDataset(
    valid_list,
    functools.partial(
        cloud_loader,
        features_used=cloud_features,
        file_type=file_type,
        max_points=max_points,
        augment=False
    )
)

test_set = tnt.dataset.ListDataset(
    test_list,
    functools.partial(
        cloud_loader,
        features_used=cloud_features,
        file_type=file_type,
        max_points=max_points,
        augment=False
    )
)

# Define Improved PointNet++ Architecture

from pointnet2_ops.pointnet2_modules import PointnetFPModule, PointnetSAModuleMSG, PointnetSAModule

class PointNet2SemSeg(nn.Module):
    def __init__(self, num_classes=2, input_channels=8):
        super().__init__()
        
        # Set Abstraction Layers
        self.sa1 = PointnetSAModuleMSG(
            npoint=2048,
            radii=[0.1, 0.2],
            nsamples=[32, 64],
            mlps=[
                [input_channels, 64, 64, 128], 
                [input_channels, 64, 96, 128]
            ],
            use_xyz=True
        )

        self.sa2 = PointnetSAModuleMSG(
            npoint=512,
            radii=[0.2, 0.4],
            nsamples=[64, 128],
            mlps=[
                [256, 128, 128, 256],
                [256, 128, 196, 256]
            ],
            use_xyz=True
        )

        self.sa3 = PointnetSAModule(
            npoint=128,
            radius=0.8,
            nsample=256,
            mlp=[512, 256, 512, 1024],
            use_xyz=True
        )

        # Attention Mechanism
        self.attention = nn.Sequential(
            nn.Conv1d(1024, 512, 1),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Conv1d(512, 1, 1),
            nn.Sigmoid()
        )

        # Feature Propagation Layers - Adjustments in input channels
        self.fp3 = PointnetFPModule(mlp=[1536, 512, 512])  # 512 (sa2) + 1024 (sa3) = 1536
        self.fp2 = PointnetFPModule(mlp=[768, 256, 256])   # 256 (sa1) + 512 (fp3) = 768
        self.fp1 = PointnetFPModule(mlp=[267, 128, 128])   # (input_channels + 3) + 256 = 267

        # Classifier
        self.classifier = nn.Sequential(
            nn.Conv1d(128, 128, 1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Conv1d(128, num_classes, 1)
        )

    def forward(self, xyz, features):
        xyz = xyz.float()
        features = features.float()

        # Set Abstraction
        l0_xyz = xyz
        l0_points = features

        l1_xyz, l1_points = self.sa1(l0_xyz, l0_points)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)

        # Attention Mechanism
        attention_weights = self.attention(l3_points)
        l3_points = l3_points * attention_weights

        # Feature Propagation
        l2_points = self.fp3(l2_xyz, l3_xyz, l2_points, l3_points)
        l1_points = self.fp2(l1_xyz, l2_xyz, l1_points, l2_points)
        l0_points = self.fp1(l0_xyz, l1_xyz, torch.cat([l0_xyz.transpose(1,2), features], dim=1), l1_points)

        # Classifier
        x = self.classifier(l0_points)
        x = x.transpose(2, 1).contiguous()

        return x

# Define PointCloudClassifier

class PointCloudClassifier:
    def __init__(self, args, device=torch.device("cpu")):
        self.subsample_size = args.subsample_size
        self.n_input_feats = args.n_input_feats
        self.n_class = args.n_class
        self.is_cuda = args.cuda
        self.device = device

    def run(self, model, clouds, train=False):
        if train:
            model.train()
        else:
            model.eval()

        clouds = clouds.to(self.device, non_blocking=True)
        pointcloud = clouds.permute(0, 2, 1).float()  # [B, N, C]

        # Separate xyz and features
        xyz = pointcloud[:, :, :3].contiguous()  # [B, N, 3]
        features = pointcloud[:, :, 3:].transpose(1, 2).contiguous()  # [B, C', N]

        # Make predictions
        pred = model(xyz, features)  # [B, N, num_classes]

        return pred

# Define ConfusionMatrix

class ConfusionMatrix:
    def __init__(self, n_class, class_names):
        self.CM = np.zeros((n_class, n_class))
        self.n_class = n_class
        self.class_names = class_names

    def clear(self):
        self.CM = np.zeros((self.n_class, self.n_class))

    def add_batch(self, gt, pred):
        if len(gt) != len(pred):
            raise ValueError(f"Inconsistent number of samples: {len(gt)} in gt, {len(pred)} in pred")
        self.CM += confusion_matrix(gt, pred, labels=list(range(self.n_class)))

    def overall_accuracy(self):
        return 100 * self.CM.trace() / self.CM.sum()

    def class_IoU(self, show=1):
        ious = np.diag(self.CM) / (np.sum(self.CM, 1) + np.sum(self.CM, 0) - np.diag(self.CM))
        if show:
            print(' / '.join('{} : {:3.2f}%'.format(name, 100 * iou) for name, iou in zip(self.class_names, ious)))
        return 100 * np.nanmean(ious)

    def print_confusion_matrix(self):
        print("Confusion Matrix:")
        print(self.CM)
        if self.n_class <= 10:
            y_true = []
            y_pred = []
            for i in range(self.n_class):
                y_true.extend([i] * int(self.CM[i].sum()))
                y_pred.extend([i] * int(self.CM[i][i]))
                for j in range(self.n_class):
                    if j != i:
                        y_pred.extend([j] * int(self.CM[i][j]))
            print(classification_report(y_true, y_pred, target_names=self.class_names))
        else:
            print("Too many classes to display detailed report.")

# Loss Functions

def dice_loss(pred, target, smooth=1.):
    num_classes = pred.shape[2]
    pred = torch.softmax(pred, dim=2)
    with torch.no_grad():
        target_one_hot = F.one_hot(target, num_classes)  # [B, N, num_classes]
    target_one_hot = target_one_hot.permute(0, 2, 1).float()  # [B, num_classes, N]
    intersection = (pred.transpose(1, 2) * target_one_hot).sum(dim=2)
    union = pred.transpose(1, 2).sum(dim=2) + target_one_hot.sum(dim=2)
    dice_score = (2. * intersection + smooth) / (union + smooth)
    dice_loss_value = 1 - dice_score
    return dice_loss_value.mean()

def focal_loss(inputs, targets, alpha, gamma=2.0, reduction='mean'):
    alpha = alpha.to(inputs.device)
    inputs_flat = inputs.view(-1, inputs.shape[-1])
    targets_flat = targets.view(-1)
    CE_loss = nn.functional.cross_entropy(inputs_flat, targets_flat, reduction='none')
    pt = torch.exp(-CE_loss)
    F_loss = alpha[targets_flat] * (1 - pt) ** gamma * CE_loss
    return F_loss.mean() if reduction == 'mean' else F_loss.sum()

def custom_flange_loss(pred, target, class_weights, boundary_weight=2.0):
    focal = focal_loss(pred, target, class_weights)
    dice = dice_loss(pred, target)

    # Detect flange edges
    target_unsqueezed = target.unsqueeze(1).float()
    
    # Calculate boundary loss using 1D convolution
    kernel_size = 3
    padding = kernel_size // 2
    
    # Create 1D kernel
    kernel = torch.ones((1, 1, kernel_size), device=pred.device)
    
    # Apply 1D convolution
    target_conv = F.conv1d(target_unsqueezed, kernel, padding=padding)
    boundary_region = (target_conv > 0) & (target_conv < kernel_size)
    
    # Calculate boundary loss
    ce_loss = nn.functional.cross_entropy(pred.view(-1, pred.shape[-1]), 
                                        target.view(-1), 
                                        reduction='none')
    ce_loss = ce_loss.view(target.shape[0], -1)
    boundary_loss = ce_loss * boundary_region.squeeze(1).float() * boundary_weight

    return focal + dice + boundary_loss.mean()

# Training and Evaluation Functions

def train(model, PCC, optimizer, args, device, class_weights):
    model.train()
    loader = torch.utils.data.DataLoader(
        train_set,
        collate_fn=cloud_collate,
        batch_size=args.batch_size,
        shuffle=True,
        pin_memory=True
    )
    loader = tqdm(loader, ncols=500, leave=False, desc="Training")
    
    cm = ConfusionMatrix(args.n_class, class_names=class_names)
    loss_meter = tnt.meter.AverageValueMeter()

    for clouds, gts in loader:
        optimizer.zero_grad()
        clouds = [c.to(device, non_blocking=True) for c in clouds]
        gts = [g.to(device=device, dtype=torch.long, non_blocking=True) for g in gts]

        # Stack clouds and labels
        clouds = torch.stack(clouds, dim=0)  # [B, C, N]
        gts = torch.stack(gts, dim=0)        # [B, N]

        # Make predictions
        pred = PCC.run(model, clouds, train=True)  # [B, N, num_classes]

        # Calculate loss
        total_loss = custom_flange_loss(pred, gts, class_weights)

        total_loss.backward()
        optimizer.step()

        loss_meter.add(total_loss.item())
        cm.add_batch(gts.cpu().flatten(), pred.argmax(2).cpu().flatten())

    cm.print_confusion_matrix()
    return cm, loss_meter.value()[0]

def eval_model(model, PCC, test, args, device, class_weights):
    model.eval()
    if test:
        loader = torch.utils.data.DataLoader(
            test_set,
            collate_fn=cloud_collate,
            batch_size=args.batch_size,
            shuffle=False,
            pin_memory=True
        )
        loader = tqdm(loader, ncols=500, leave=False, desc="Test")
    else:
        loader = torch.utils.data.DataLoader(
            valid_set,
            collate_fn=cloud_collate,
            batch_size=args.batch_size,
            shuffle=False,
            drop_last=False,
            pin_memory=True
        )
        loader = tqdm(loader, ncols=500, leave=False, desc="Validation")

    loss_meter = tnt.meter.AverageValueMeter()
    cm = ConfusionMatrix(args.n_class, class_names=class_names)

    with torch.no_grad():
        for clouds, gts in loader:
            clouds = [c.to(device, non_blocking=True) for c in clouds]
            gts = [g.to(device=device, dtype=torch.long, non_blocking=True) for g in gts]

            # Stack clouds and labels
            clouds = torch.stack(clouds, dim=0)  # [B, C, N]
            gts = torch.stack(gts, dim=0)  # [B, N]

            # Make predictions
            pred = PCC.run(model, clouds, train=False)  # [B, N, num_classes]

            # Calculate loss
            total_loss = custom_flange_loss(pred, gts, class_weights)

            loss_meter.add(total_loss.item())
            cm.add_batch(gts.cpu().flatten(), pred.argmax(2).cpu().flatten())

    cm.print_confusion_matrix()
    return cm, loss_meter.value()[0]

# Training Loop

def train_full(args, device):
    # Update the number of input channels
    input_channels = args.n_input_feats - 3  # We subtract 3 because xyz coordinates are used separately

    model = PointNet2SemSeg(
        num_classes=args.n_class,
        input_channels=input_channels
    ).to(device)

    print('Total number of parameters:', sum(p.numel() for p in model.parameters()))

    best_model = None
    best_mIoU = 0
    no_improve_epochs = 0
    early_stop_patience = 20

    PCC = PointCloudClassifier(args, device=device)

    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)

    class_weights = torch.tensor([1.0, 50.0], dtype=torch.float32, device=device)
    class_weights.requires_grad = False
    print(f"Class Weights: {class_weights}")

    total_flange_points = 0
    for _, gt in train_set:
        total_flange_points += (gt == 1).sum().item()
    print(f"Total flange points in training set: {total_flange_points}")
    if total_flange_points == 0:
        raise ValueError("No flange points found in training set.")

    metrics_pn = {}
    metrics_pn['definition'] = [
        ['train_oa', 'train_mIoU', 'train_loss'],
        ['valid_oa', 'valid_mIoU', 'valid_loss'],
        ['test_oa', 'test_mIoU', 'test_loss']
    ]

    for i_epoch in tqdm(range(args.n_epoch), desc='Training'):
        cm_train, loss_train = train(model, PCC, optimizer, args, device, class_weights)
        mIoU = cm_train.class_IoU()
        tqdm.write(f'Epoch {i_epoch:3d} -> Training Accuracy: {cm_train.overall_accuracy():.2f}%, Training mIoU: {mIoU:.2f}%, Training Loss: {loss_train:.4f}')

        metrics_pn[i_epoch] = [[cm_train.overall_accuracy(), mIoU, loss_train]]

        cm_valid, loss_valid = eval_model(model, PCC, False, args=args, device=device, class_weights=class_weights)
        mIoU_valid = cm_valid.class_IoU()

        metrics_pn[i_epoch].append([cm_valid.overall_accuracy(), mIoU_valid, loss_valid])

        scheduler.step()

        best_valid = False
        if mIoU_valid > best_mIoU:
            best_valid = True
            best_mIoU = mIoU_valid
            best_model = copy.deepcopy(model)
            no_improve_epochs = 0
            tqdm.write(f'Best performance at epoch {i_epoch:3d} -> Validation Accuracy: {cm_valid.overall_accuracy():.2f}%, Validation mIoU: {mIoU_valid:.2f}%, Validation Loss: {loss_valid:.4f}')
        else:
            no_improve_epochs += 1
            tqdm.write(f'Epoch {i_epoch:3d} -> Validation Accuracy: {cm_valid.overall_accuracy():.2f}%, Validation mIoU: {mIoU_valid:.2f}%, Validation Loss: {loss_valid:.4f}')

        if no_improve_epochs >= early_stop_patience:
            tqdm.write(f'Early stopping at epoch {i_epoch} due to {early_stop_patience} epochs without improvement.')
            break

        if i_epoch == args.n_epoch - 1 or best_valid:
            cm_test, loss_test = eval_model(best_model, PCC, True, args=args, device=device, class_weights=class_weights)
            mIoU_test = cm_test.class_IoU()
            tqdm.write(f'Epoch {i_epoch:3d} -> Test Accuracy: {cm_test.overall_accuracy():.2f}%, Test mIoU: {mIoU_test:.2f}%, Test Loss: {loss_test:.4f}')

            metrics_pn[i_epoch].append([cm_test.overall_accuracy(), mIoU_test, loss_test])

    return best_model, metrics_pn, optimizer, scheduler

# Prediction Visualization

def tile_prediction(tile_name, model=None, PCC=None, Visualization=True, features_used='xyzrgbi', file_type='ply'):
    cloud, gt = cloud_loader(tile_name, features_used, file_type=file_type)

    # Ensure consistent number of points
    if cloud.shape[1] > args.subsample_size:
        cloud = cloud[:, :args.subsample_size]
        gt = gt[:args.subsample_size]

    clouds = cloud.unsqueeze(0).to(device)  # [1, C, N]
    labels = PCC.run(model, clouds, train=False)
    labels = labels.argmax(2).squeeze(0).cpu().numpy()

    xyz = cloud[:3, :].numpy().T

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)

    if Visualization:
        colors = np.zeros((labels.shape[0], 3))
        colormap = {
            0: [0, 0, 1],  # Blue for 'equipment'
            1: [1, 0, 0],  # Red for 'flange'
        }
        for label in np.unique(labels):
            colors[labels == label] = colormap.get(label, [0.5, 0.5, 0.5])
        pcd.colors = o3d.utility.Vector3dVector(colors)
        o3d.visualization.draw_geometries([pcd])

    return pcd, labels

# Export Model and Metrics

def export_results(trained_model, metrics_pn, project_dir, optimizer, scheduler):
    model_path = f'./pointnet222improved2_model_{os.path.basename(project_dir)}.pth'
    torch.save({
        'model_state_dict': trained_model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict()
    }, model_path)
    print(f"Model saved at {model_path}")

    metrics_path = f"./metrics_pointnet222improved2_{os.path.basename(project_dir)}.csv"
    with open(metrics_path, "w", newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Epoch', 'Train_OA', 'Train_mIoU', 'Train_Loss',
                         'Valid_OA', 'Valid_mIoU', 'Valid_Loss',
                         'Test_OA', 'Test_mIoU', 'Test_Loss'])
        for epoch, metrics in metrics_pn.items():
            if epoch == 'definition':
                continue
            if len(metrics) == 3:
                train_metrics, valid_metrics, test_metrics = metrics
            elif len(metrics) == 2:
                train_metrics, valid_metrics = metrics
                test_metrics = [0, 0, 0]
            else:
                train_metrics = metrics[0]
                valid_metrics = [0, 0, 0]
                test_metrics = [0, 0, 0]
            writer.writerow([epoch] + train_metrics + valid_metrics + test_metrics)
    print(f"Metrics saved at {metrics_path}")

# Main Execution

if __name__ == "__main__":
    t0 = time.time()
    trained_model, metrics_pn, optimizer, scheduler = train_full(args, device)
    t1 = time.time()

    print('-' * 50)
    print(f"Total training time: {t1 - t0:.2f} seconds")
    print('=' * 50)

    PCC = PointCloudClassifier(args, device=device)

    if len(test_list) > 0:
        # Visualize 2 more equipment
        for idx in range(min(2, len(test_list))):
            selection = test_list[idx]
            print(f"Visualizing test file: {selection}")
            pcd, labels = tile_prediction(selection, model=trained_model, PCC=PCC, file_type=file_type)
    else:
        print("No test files found. Skipping prediction visualization.")

    export_results(trained_model, metrics_pn, project_dir, optimizer, scheduler)

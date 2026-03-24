"""
POINTNET++ SEMANTIC SEGMENTATION — LABELS DATASET
===================================================
Segmentação semântica ponto a ponto de flanges em nuvens de pontos industriais.
Usa os PLY originais com scalar_Classification (0=equipamento, 1=flange).
Split por equipamento para evitar vazamento de dados.

Adaptado para usar o dataset baseado em labels diretos (sem HDBSCAN).
"""

import os
import sys
import random
import copy
import csv
import time
import numpy as np
import pandas as pd
from glob import glob
from plyfile import PlyData

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from sklearn.metrics import confusion_matrix, classification_report
from sklearn.neighbors import KDTree
from tqdm.auto import tqdm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ╔══════════════════════════════════════════════════════════════╗
# ║                      CONFIGURAÇÃO                           ║
# ╚══════════════════════════════════════════════════════════════╝

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Diretório com os PLY originais (cada arquivo = 1 equipamento com labels)
DATA_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, "DATA"))

# Saídas
MODEL_PATH = os.path.join(SCRIPT_DIR, "best_segmentation_labels.pth")
HISTORY_PATH = os.path.join(SCRIPT_DIR, "segmentation_history_labels.csv")
CM_PATH = os.path.join(SCRIPT_DIR, "segmentation_cm_labels.png")
REPORT_PATH = os.path.join(SCRIPT_DIR, "segmentation_report_labels.txt")
CURVES_PATH = os.path.join(SCRIPT_DIR, "segmentation_curves_labels.png")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hiperparâmetros
NUM_POINTS = 8192          # pontos por amostra (maior que classificação)
BATCH_SIZE = 2             # menor porque cada sample é grande
NUM_EPOCHS = 200
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-5
EARLY_STOP_PATIENCE = 30
FLANGE_RATIO = 0.4         # fração mínima de pontos flange por amostra
CLASS_WEIGHT_FLANGE = 50.0  # peso da classe flange na loss
SEED = 42

# Features: xyz + normals + curvature + rgb + intensity = 11 canais
# (xyz é separado, então input_channels = 8)
USE_RGB = True
USE_INTENSITY = True

CLASS_NAMES = ["equipment", "flange"]

print(f"Device: {DEVICE}")
print(f"Data dir: {DATA_DIR}")


# ╔══════════════════════════════════════════════════════════════╗
# ║                   REPRODUTIBILIDADE                          ║
# ╚══════════════════════════════════════════════════════════════╝

def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


seed_everything(SEED)


# ╔══════════════════════════════════════════════════════════════╗
# ║                   LEITURA DE DADOS                           ║
# ╚══════════════════════════════════════════════════════════════╝

CLASS_FIELD_CANDIDATES = [
    "scalar_Classification", "scalar_classification",
    "Classification", "classification",
    "label", "scalar_Label", "class"
]


def find_class_field(vertex_properties):
    prop_names = [p.name for p in vertex_properties]
    for name in CLASS_FIELD_CANDIDATES:
        if name in prop_names:
            return name
    raise ValueError(f"Campo de classificação não encontrado. Campos: {prop_names}")


def read_ply_with_features(filepath):
    """Lê PLY e retorna coords, features extras e labels."""
    plydata = PlyData.read(filepath)
    vertex = plydata["vertex"].data
    prop_names = [p.name for p in plydata["vertex"].properties]

    # Coordenadas XYZ
    x = np.array(vertex["x"], dtype=np.float32)
    y = np.array(vertex["y"], dtype=np.float32)
    z = np.array(vertex["z"], dtype=np.float32)
    xyz = np.column_stack([x, y, z])

    # Labels
    class_field = find_class_field(plydata["vertex"].properties)
    labels = np.array(vertex[class_field], dtype=np.int32)

    # RGB (se disponível)
    rgb = None
    if USE_RGB:
        if "red" in prop_names and "green" in prop_names and "blue" in prop_names:
            r = np.array(vertex["red"], dtype=np.float32)
            g = np.array(vertex["green"], dtype=np.float32)
            b = np.array(vertex["blue"], dtype=np.float32)
            rgb = np.column_stack([r, g, b])

    # Intensidade (se disponível)
    intensity = None
    if USE_INTENSITY and "intensity" in prop_names:
        intensity = np.array(vertex["intensity"], dtype=np.float32).reshape(-1, 1)

    return xyz, labels, rgb, intensity


def estimate_normals_and_curvature(xyz, k=16):
    """Estima normais e curvatura local via PCA em vizinhanças KNN."""
    N = xyz.shape[0]
    normals = np.zeros((N, 3), dtype=np.float32)
    curvature = np.zeros((N, 1), dtype=np.float32)

    tree = KDTree(xyz)
    # Processar em blocos para não estourar memória
    block_size = 50000
    for start in range(0, N, block_size):
        end = min(start + block_size, N)
        block = xyz[start:end]
        idx = tree.query(block, k=k, return_distance=False)

        for i_local, i_global in enumerate(range(start, end)):
            neighbors = xyz[idx[i_local]]
            centered = neighbors - np.mean(neighbors, axis=0)
            cov = centered.T @ centered
            eigvals, eigvecs = np.linalg.eigh(cov)
            # Normal = eigenvector com menor eigenvalue
            normals[i_global] = eigvecs[:, 0]
            # Curvatura = menor eigenvalue / soma
            curvature[i_global] = eigvals[0] / (np.sum(eigvals) + 1e-8)

    return normals, curvature


def normalize_xyz_unit_sphere(xyz):
    """
    Centraliza em torno da origem e escala pela maior distância ao centro.
    Mantém a geometria consistente para o PointNet++.
    """
    xyz = xyz.astype(np.float32)
    center = np.mean(xyz, axis=0, keepdims=True)
    xyz = xyz - center

    radius = np.max(np.linalg.norm(xyz, axis=1))
    if radius < 1e-8:
        radius = 1.0

    xyz = xyz / radius
    return xyz.astype(np.float32)


def normalize_scalar_channel(x):
    """
    Normaliza 1 canal escalar por z-score.
    """
    x = x.astype(np.float32)
    mean = np.mean(x, axis=0, keepdims=True)
    std = np.std(x, axis=0, keepdims=True) + 1e-8
    return ((x - mean) / std).astype(np.float32)


def build_features(xyz_centered, normals, curvature, rgb, intensity):
    """
    Monta as features do jeito certo:
    - xyz: normalização geométrica (unit sphere)
    - normals: mantém como vetor direcional e renormaliza módulo
    - curvature/rgb/intensity: normalização por canal
    """
    # XYZ geométrico
    xyz_geom = normalize_xyz_unit_sphere(xyz_centered)

    # Normais: garantir módulo ~1
    normals = normals.astype(np.float32)
    normals_norm = np.linalg.norm(normals, axis=1, keepdims=True) + 1e-8
    normals = normals / normals_norm

    # Curvatura
    curvature = normalize_scalar_channel(curvature)

    # RGB
    if rgb is None:
        rgb = np.zeros((len(xyz_centered), 3), dtype=np.float32)
    else:
        rgb = rgb.astype(np.float32)
        # Se vier 0..255, joga para 0..1
        if np.max(rgb) > 1.5:
            rgb = rgb / 255.0

    # Intensidade
    if intensity is None:
        intensity = np.zeros((len(xyz_centered), 1), dtype=np.float32)
    else:
        intensity = normalize_scalar_channel(intensity)

    features = np.concatenate(
        [xyz_geom, normals, curvature, rgb, intensity],
        axis=1
    ).astype(np.float32)

    return features


# ╔══════════════════════════════════════════════════════════════╗
# ║                   SPLIT POR EQUIPAMENTO                      ║
# ╚══════════════════════════════════════════════════════════════╝

def split_equipment_files(data_dir, seed=42):
    """Divide os arquivos PLY em treino/validação/teste por equipamento."""
    ply_files = sorted(glob(os.path.join(data_dir, "*.ply")))
    if not ply_files:
        raise FileNotFoundError(f"Nenhum arquivo PLY encontrado em {data_dir}")

    print(f"Total de equipamentos (PLY): {len(ply_files)}")

    rng = np.random.default_rng(seed)
    indices = np.arange(len(ply_files))
    rng.shuffle(indices)

    n = len(indices)
    n_test = max(1, int(n * 0.20))
    n_valid = max(1, int(n * 0.10))
    n_train = n - n_test - n_valid

    train_files = [ply_files[i] for i in indices[:n_train]]
    valid_files = [ply_files[i] for i in indices[n_train:n_train + n_valid]]
    test_files = [ply_files[i] for i in indices[n_train + n_valid:]]

    print(f"Train: {len(train_files)}, Valid: {len(valid_files)}, Test: {len(test_files)}")
    return train_files, valid_files, test_files


# ╔══════════════════════════════════════════════════════════════╗
# ║                      DATASET                                ║
# ╚══════════════════════════════════════════════════════════════╝

class SegmentationDataset(Dataset):
    """
    Dataset para segmentação semântica.
    Cada __getitem__ retorna um patch de NUM_POINTS pontos com features e labels.
    Usa amostragem estratificada para garantir presença de flanges.
    """

    def __init__(self, ply_files, num_points=8192, epoch_size=100,
                 train=True, flange_ratio=0.4):
        self.ply_files = ply_files
        self.num_points = num_points
        self.epoch_size = epoch_size
        self.train = train
        self.flange_ratio = flange_ratio
        self.cache = {}

    def __len__(self):
        return self.epoch_size

    def _load_equipment(self, filepath):
        if filepath not in self.cache:
            xyz, labels_raw, rgb, intensity = read_ply_with_features(filepath)

            # Binarizar labels: 1=flange, tudo resto=0 (equipamento)
            labels = np.where(labels_raw == 1, 1, 0).astype(np.int64)

            # Centralizar coordenadas
            mean_xyz = np.mean(xyz, axis=0, keepdims=True)
            min_z = np.min(xyz[:, 2:3], axis=0, keepdims=True)
            xyz_centered = xyz.copy()
            xyz_centered[:, 0] -= mean_xyz[0, 0]
            xyz_centered[:, 1] -= mean_xyz[0, 1]
            xyz_centered[:, 2] -= min_z[0, 0]

            # Estimar normais e curvatura
            print(f"  Calculando normais/curvatura para {os.path.basename(filepath)}...")
            normals, curvature = estimate_normals_and_curvature(xyz_centered, k=16)

            # Montar features 
            features = build_features(
                xyz_centered=xyz_centered,
                normals=normals,
                curvature=curvature,
                rgb=rgb,
                intensity=intensity
            )

            # Índices por classe
            flange_idx = np.where(labels == 1)[0]
            equip_idx = np.where(labels != 1)[0]

            self.cache[filepath] = {
                "features": features,
                "labels": labels,
                "flange_idx": flange_idx,
                "equip_idx": equip_idx,
            }

            n_flange = len(flange_idx)
            n_equip = len(equip_idx)
            print(f"    -> {n_flange} flange, {n_equip} equipamento "
                  f"({100*n_flange/(n_flange+n_equip):.1f}% flange)")

        return self.cache[filepath]

    def __getitem__(self, idx):
        # Escolher equipamento
        if self.train:
            file_idx = np.random.randint(0, len(self.ply_files))
        else:
            file_idx = idx % len(self.ply_files)

        data = self._load_equipment(self.ply_files[file_idx])
        features = data["features"]
        labels = data["labels"]
        flange_idx = data["flange_idx"]
        equip_idx = data["equip_idx"]

        n_flange = len(flange_idx)
        n_equip = len(equip_idx)
        n_total = n_flange + n_equip

        if self.train:
            # Treino: amostragem estratificada com flange_ratio
            if n_flange > 0 and self.flange_ratio is not None:
                desired_flange = max(1, int(self.num_points * self.flange_ratio))
                desired_equip = self.num_points - desired_flange

                sampled_flange = np.random.choice(
                    flange_idx, desired_flange, replace=(n_flange < desired_flange)
                )
                sampled_equip = np.random.choice(
                    equip_idx, desired_equip, replace=(n_equip < desired_equip)
                )
                sampled = np.concatenate([sampled_flange, sampled_equip])
            else:
                sampled = np.random.choice(n_total, self.num_points, replace=(n_total < self.num_points))
            np.random.shuffle(sampled)
        else:
            # Valid/test: distribuição natural, determinístico por idx
            rng = np.random.default_rng(idx + SEED)
            all_idx = np.arange(n_total)
            sampled = rng.choice(all_idx, self.num_points, replace=(n_total < self.num_points))

        feat_sample = features[sampled]
        label_sample = labels[sampled]

        # Data augmentation (só treino)
        if self.train:
            feat_sample = self._augment(feat_sample)

        feat_tensor = torch.from_numpy(feat_sample.astype(np.float32))
        label_tensor = torch.from_numpy(label_sample.astype(np.int64))

        return feat_tensor, label_tensor

    def _augment(self, features):
        features = features.copy()
        xyz = features[:, :3]
        normals = features[:, 3:6]

        # Rotação em z
        theta = np.random.uniform(0, 2 * np.pi)
        rot = np.array([
            [np.cos(theta), -np.sin(theta), 0],
            [np.sin(theta),  np.cos(theta), 0],
            [0, 0, 1]
        ], dtype=np.float32)

        xyz = xyz @ rot.T
        normals = normals @ rot.T

        # Renormalizar normais
        normals_norm = np.linalg.norm(normals, axis=1, keepdims=True) + 1e-8
        normals = normals / normals_norm

        # Escala só no xyz
        scale = np.random.uniform(0.9, 1.1)
        xyz *= scale

        # Jitter só no xyz
        xyz += np.random.normal(0, 0.01, size=xyz.shape).astype(np.float32)

        features[:, :3] = xyz
        features[:, 3:6] = normals
        return features


def seg_collate(batch):
    feats, labels = zip(*batch)
    feats = torch.stack(feats, dim=0)    # (B, N, C)
    labels = torch.stack(labels, dim=0)  # (B, N)
    return feats, labels


# ╔══════════════════════════════════════════════════════════════╗
# ║                POINTNET++ SEGMENTAÇÃO (Pure PyTorch)         ║
# ╚══════════════════════════════════════════════════════════════╝

def square_distance(src, dst):
    dist = -2 * torch.matmul(src, dst.transpose(1, 2))
    dist += torch.sum(src ** 2, dim=-1).unsqueeze(-1)
    dist += torch.sum(dst ** 2, dim=-1).unsqueeze(1)
    return dist


def index_points(points, idx):
    B = points.shape[0]
    batch_idx = torch.arange(B, dtype=torch.long, device=points.device).view(
        B, *((1,) * (idx.dim() - 1))
    )
    return points[batch_idx, idx, :]


def farthest_point_sample(xyz, npoint):
    device = xyz.device
    B, N, C = xyz.shape
    npoint = min(npoint, N)

    centroids = torch.zeros(B, npoint, dtype=torch.long, device=device)
    distance = torch.ones(B, N, device=device) * 1e10
    farthest = torch.randint(0, N, (B,), dtype=torch.long, device=device)
    batch_idx = torch.arange(B, dtype=torch.long, device=device)

    for i in range(npoint):
        centroids[:, i] = farthest
        centroid = xyz[batch_idx, farthest, :].view(B, 1, C)
        dist = torch.sum((xyz - centroid) ** 2, dim=-1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = torch.max(distance, dim=-1)[1]

    return centroids


def sample_and_group(npoint, nsample, xyz, points):
    B, N, C = xyz.shape
    npoint = min(npoint, N) if npoint is not None else N

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

    return new_xyz, new_points, fps_idx


def sample_and_group_all(xyz, points):
    device = xyz.device
    B, N, C = xyz.shape
    new_xyz = torch.zeros(B, 1, C, device=device)
    grouped_xyz = xyz.view(B, 1, N, C)

    if points is not None:
        grouped_points = points.view(B, 1, N, -1)
        new_points = torch.cat([grouped_xyz, grouped_points], dim=-1)
    else:
        new_points = grouped_xyz

    return new_xyz, new_points


class SetAbstraction(nn.Module):
    def __init__(self, npoint, nsample, in_channel, mlp):
        super().__init__()
        self.npoint = npoint
        self.nsample = nsample

        last_ch = in_channel
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        for out_ch in mlp:
            self.convs.append(nn.Conv2d(last_ch, out_ch, 1))
            self.bns.append(nn.BatchNorm2d(out_ch))
            last_ch = out_ch

    def forward(self, xyz, points):
        if self.npoint is not None:
            new_xyz, new_points, fps_idx = sample_and_group(
                self.npoint, self.nsample, xyz, points
            )
        else:
            new_xyz, new_points = sample_and_group_all(xyz, points)
            fps_idx = None

        # (B, npoint, nsample, C) -> (B, C, npoint, nsample)
        new_points = new_points.permute(0, 3, 1, 2)

        for conv, bn in zip(self.convs, self.bns):
            new_points = F.relu(bn(conv(new_points)))

        new_points = torch.max(new_points, dim=-1)[0]  # (B, C', npoint)
        new_points = new_points.permute(0, 2, 1)       # (B, npoint, C')

        return new_xyz, new_points, fps_idx


class FeaturePropagation(nn.Module):
    def __init__(self, in_channel, mlp):
        super().__init__()
        last_ch = in_channel
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        for out_ch in mlp:
            self.convs.append(nn.Conv1d(last_ch, out_ch, 1))
            self.bns.append(nn.BatchNorm1d(out_ch))
            last_ch = out_ch

    def forward(self, xyz1, xyz2, points1, points2):
        """
        xyz1: (B, N, 3) — posições de maior resolução
        xyz2: (B, S, 3) — posições de menor resolução
        points1: (B, N, D1) — features skip connection
        points2: (B, S, D2) — features a interpolar
        """
        B, N, C = xyz1.shape
        _, S, _ = xyz2.shape

        if S == 1:
            # Caso global: replicar para todos os pontos
            interpolated = points2.repeat(1, N, 1)
        else:
            # Interpolação por distância (3 vizinhos mais próximos)
            dists = square_distance(xyz1, xyz2)  # (B, N, S)
            dists, idx = dists.sort(dim=-1)
            dists = dists[:, :, :3]
            idx = idx[:, :, :3]

            dist_recip = 1.0 / (dists + 1e-8)
            norm = torch.sum(dist_recip, dim=2, keepdim=True)
            weight = dist_recip / norm  # (B, N, 3)

            interpolated_points = index_points(points2, idx)  # (B, N, 3, D2)
            interpolated = torch.sum(
                interpolated_points * weight.unsqueeze(-1), dim=2
            )  # (B, N, D2)

        if points1 is not None:
            new_points = torch.cat([points1, interpolated], dim=-1)
        else:
            new_points = interpolated

        # (B, N, D) -> (B, D, N)
        new_points = new_points.permute(0, 2, 1)

        for conv, bn in zip(self.convs, self.bns):
            new_points = F.relu(bn(conv(new_points)))

        return new_points.permute(0, 2, 1)  # (B, N, D')


class PointNet2SegModel(nn.Module):
    """PointNet++ para segmentação semântica com encoder-decoder."""

    def __init__(self, num_classes=2, input_channels=8):
        super().__init__()

        # Encoder (Set Abstraction)
        # input_channels = features extras (sem xyz, que são 3)
        self.sa1 = SetAbstraction(
            npoint=2048, nsample=32, in_channel=3 + input_channels, mlp=[64, 64, 128]
        )
        self.sa2 = SetAbstraction(
            npoint=512, nsample=64, in_channel=3 + 128, mlp=[128, 128, 256]
        )
        self.sa3 = SetAbstraction(
            npoint=128, nsample=128, in_channel=3 + 256, mlp=[256, 256, 512]
        )
        self.sa4 = SetAbstraction(
            npoint=None, nsample=None, in_channel=3 + 512, mlp=[512, 512, 1024]
        )

        # Attention
        self.attention = nn.Sequential(
            nn.Conv1d(1024, 512, 1),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Conv1d(512, 1024, 1),
            nn.Sigmoid()
        )

        # Decoder (Feature Propagation)
        self.fp4 = FeaturePropagation(in_channel=1024 + 512, mlp=[512, 512])
        self.fp3 = FeaturePropagation(in_channel=512 + 256, mlp=[256, 256])
        self.fp2 = FeaturePropagation(in_channel=256 + 128, mlp=[128, 128])
        self.fp1 = FeaturePropagation(
            in_channel=128 + input_channels, mlp=[128, 128]
        )

        # Segmentation head
        self.seg_head = nn.Sequential(
            nn.Conv1d(128, 128, 1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Conv1d(128, num_classes, 1)
        )

    def forward(self, features):
        """
        features: (B, N, C_total) onde C_total = 3(xyz) + input_channels
        Retorna: (B, N, num_classes)
        """
        xyz = features[:, :, :3].contiguous()              # (B, N, 3)
        extra_feats = features[:, :, 3:].contiguous()      # (B, N, input_channels)

        # Encoder
        l0_xyz = xyz
        l0_points = extra_feats

        l1_xyz, l1_points, _ = self.sa1(l0_xyz, l0_points)
        l2_xyz, l2_points, _ = self.sa2(l1_xyz, l1_points)
        l3_xyz, l3_points, _ = self.sa3(l2_xyz, l2_points)
        l4_xyz, l4_points, _ = self.sa4(l3_xyz, l3_points)

        # Attention no nível global
        attn_input = l4_points.permute(0, 2, 1)  # (B, C, 1)
        attn_weights = self.attention(attn_input)
        l4_points = (l4_points.permute(0, 2, 1) * attn_weights).permute(0, 2, 1)

        # Decoder
        l3_points = self.fp4(l3_xyz, l4_xyz, l3_points, l4_points)
        l2_points = self.fp3(l2_xyz, l3_xyz, l2_points, l3_points)
        l1_points = self.fp2(l1_xyz, l2_xyz, l1_points, l2_points)
        l0_points = self.fp1(l0_xyz, l1_xyz, l0_points, l1_points)

        # Segmentation head
        x = l0_points.permute(0, 2, 1)  # (B, 128, N)
        x = self.seg_head(x)             # (B, num_classes, N)
        x = x.permute(0, 2, 1)           # (B, N, num_classes)

        return x


# ╔══════════════════════════════════════════════════════════════╗
# ║                     LOSS FUNCTIONS                           ║
# ╚══════════════════════════════════════════════════════════════╝

def focal_loss(pred, target, class_weights, gamma=2.0):
    """Focal loss para lidar com desbalanceamento."""
    pred_flat = pred.reshape(-1, pred.shape[-1])
    target_flat = target.reshape(-1)
    ce = F.cross_entropy(pred_flat, target_flat, reduction="none")
    pt = torch.exp(-ce)
    weights = class_weights[target_flat]
    loss = weights * (1 - pt) ** gamma * ce
    return loss.mean()


def dice_loss(pred, target, smooth=1.0):
    """Dice loss para melhorar sobreposição."""
    num_classes = pred.shape[-1]
    pred_soft = torch.softmax(pred, dim=-1)

    target_oh = F.one_hot(target, num_classes).float()  # (B, N, C)

    intersection = (pred_soft * target_oh).sum(dim=1)  # (B, C)
    union = pred_soft.sum(dim=1) + target_oh.sum(dim=1)
    dice = (2.0 * intersection + smooth) / (union + smooth)
    return (1 - dice).mean()


def boundary_loss(pred, target, boundary_weight=2.0):
    """Penaliza erros nas bordas flange/equipamento."""
    target_float = target.unsqueeze(1).float()  # (B, 1, N)
    kernel = torch.ones(1, 1, 3, device=pred.device)
    target_conv = F.conv1d(target_float, kernel, padding=1)
    boundary = ((target_conv > 0) & (target_conv < 3)).squeeze(1).float()

    ce = F.cross_entropy(
        pred.reshape(-1, pred.shape[-1]),
        target.reshape(-1),
        reduction="none"
    ).reshape(target.shape)

    return (ce * boundary * boundary_weight).mean()


def combined_loss(pred, target, class_weights):
    """Combinação de focal + dice (sem boundary — pontos não têm ordem espacial)."""
    fl = focal_loss(pred, target, class_weights)
    dl = dice_loss(pred, target)
    return fl + dl


# ╔══════════════════════════════════════════════════════════════╗
# ║                   MÉTRICAS                                   ║
# ╚══════════════════════════════════════════════════════════════╝

class SegMetrics:
    def __init__(self, n_classes, class_names):
        self.n_classes = n_classes
        self.class_names = class_names
        self.cm = np.zeros((n_classes, n_classes), dtype=np.int64)

    def clear(self):
        self.cm = np.zeros((self.n_classes, self.n_classes), dtype=np.int64)

    def add_batch(self, gt, pred):
        self.cm += confusion_matrix(
            gt.flatten(), pred.flatten(),
            labels=list(range(self.n_classes))
        )

    def overall_accuracy(self):
        return 100.0 * self.cm.trace() / max(self.cm.sum(), 1)

    def class_iou(self):
        ious = np.diag(self.cm) / (
            self.cm.sum(1) + self.cm.sum(0) - np.diag(self.cm) + 1e-8
        )
        return ious

    def mean_iou(self):
        return 100.0 * np.nanmean(self.class_iou())

    def print_report(self):
        ious = self.class_iou()
        print("Per-class IoU:")
        for name, iou in zip(self.class_names, ious):
            print(f"  {name}: {100*iou:.2f}%")
        print(f"mIoU: {self.mean_iou():.2f}%")
        print(f"OA:   {self.overall_accuracy():.2f}%")


# ╔══════════════════════════════════════════════════════════════╗
# ║                 TREINO / AVALIAÇÃO                           ║
# ╚══════════════════════════════════════════════════════════════╝

def train_epoch(model, loader, optimizer, class_weights, device):
    model.train()
    metrics = SegMetrics(2, CLASS_NAMES)
    total_loss = 0.0
    n_samples = 0

    for feats, labels in tqdm(loader, desc="Train", leave=False):
        feats = feats.to(device).float()
        labels = labels.to(device)

        optimizer.zero_grad()
        pred = model(feats)  # (B, N, 2)
        loss = combined_loss(pred, labels, class_weights)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * labels.size(0)
        n_samples += labels.size(0)

        pred_cls = pred.argmax(dim=-1).cpu().numpy()
        gt_cls = labels.cpu().numpy()
        metrics.add_batch(gt_cls, pred_cls)

    avg_loss = total_loss / max(n_samples, 1)
    return avg_loss, metrics.overall_accuracy(), metrics.mean_iou()


def eval_epoch(model, loader, class_weights, device):
    model.eval()
    metrics = SegMetrics(2, CLASS_NAMES)
    total_loss = 0.0
    n_samples = 0

    with torch.no_grad():
        for feats, labels in tqdm(loader, desc="Eval", leave=False):
            feats = feats.to(device).float()
            labels = labels.to(device)

            pred = model(feats)
            loss = combined_loss(pred, labels, class_weights)

            total_loss += loss.item() * labels.size(0)
            n_samples += labels.size(0)

            pred_cls = pred.argmax(dim=-1).cpu().numpy()
            gt_cls = labels.cpu().numpy()
            metrics.add_batch(gt_cls, pred_cls)

    avg_loss = total_loss / max(n_samples, 1)
    return avg_loss, metrics.overall_accuracy(), metrics.mean_iou(), metrics


# ╔══════════════════════════════════════════════════════════════╗
# ║                    PLOTS E EXPORTS                           ║
# ╚══════════════════════════════════════════════════════════════╝

def plot_training_curves(history_path, save_path):
    """Plota curvas de loss, OA e mIoU ao longo do treinamento."""
    df = pd.read_csv(history_path)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Loss
    axes[0].plot(df["epoch"], df["train_loss"], label="Train", linewidth=1.5)
    axes[0].plot(df["epoch"], df["valid_loss"], label="Valid", linewidth=1.5)
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Loss")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # OA
    axes[1].plot(df["epoch"], df["train_oa"], label="Train", linewidth=1.5)
    axes[1].plot(df["epoch"], df["valid_oa"], label="Valid", linewidth=1.5)
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Overall Accuracy (%)")
    axes[1].set_title("Overall Accuracy")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    # mIoU
    axes[2].plot(df["epoch"], df["train_miou"], label="Train", linewidth=1.5)
    axes[2].plot(df["epoch"], df["valid_miou"], label="Valid", linewidth=1.5)
    axes[2].set_xlabel("Epoch")
    axes[2].set_ylabel("mIoU (%)")
    axes[2].set_title("Mean IoU")
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Curvas salvas em: {save_path}")


def save_confusion_matrix(metrics, save_path):
    """Salva a matriz de confusão como imagem."""
    from sklearn.metrics import ConfusionMatrixDisplay

    fig, ax = plt.subplots(figsize=(6, 6))
    disp = ConfusionMatrixDisplay(
        confusion_matrix=metrics.cm,
        display_labels=CLASS_NAMES
    )
    disp.plot(ax=ax, cmap="Blues", colorbar=False)
    ax.set_title("Confusion Matrix — Segmentation")
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Matriz de confusão salva em: {save_path}")


# ╔══════════════════════════════════════════════════════════════╗
# ║                      MAIN                                   ║
# ╚══════════════════════════════════════════════════════════════╝

def main():
    t_start = time.time()

    # Split por equipamento
    train_files, valid_files, test_files = split_equipment_files(DATA_DIR, seed=SEED)

    # Calcular número de features
    # xyz(3) + normals(3) + curvature(1) + rgb(3) + intensity(1) = 11
    input_channels = 3 + 1  # normals + curvature
    if USE_RGB:
        input_channels += 3
    if USE_INTENSITY:
        input_channels += 1
    # total = 8 (sem xyz)

    print(f"\nInput channels (sem xyz): {input_channels}")

    # Datasets
    train_dataset = SegmentationDataset(
        train_files, num_points=NUM_POINTS,
        epoch_size=100, train=True, flange_ratio=FLANGE_RATIO
    )
    valid_dataset = SegmentationDataset(
        valid_files, num_points=NUM_POINTS,
        epoch_size=50, train=False, flange_ratio=None  # distribuição natural
    )
    test_dataset = SegmentationDataset(
        test_files, num_points=NUM_POINTS,
        epoch_size=50, train=False, flange_ratio=None  # distribuição natural
    )

    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE,
        shuffle=True, num_workers=0, collate_fn=seg_collate
    )
    valid_loader = DataLoader(
        valid_dataset, batch_size=BATCH_SIZE,
        shuffle=False, num_workers=0, collate_fn=seg_collate
    )
    test_loader = DataLoader(
        test_dataset, batch_size=BATCH_SIZE,
        shuffle=False, num_workers=0, collate_fn=seg_collate
    )

    # Modelo
    model = PointNet2SegModel(
        num_classes=2,
        input_channels=input_channels
    ).to(DEVICE)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Parâmetros do modelo: {n_params:,}")

    # Loss weights
    class_weights = torch.tensor(
        [1.0, CLASS_WEIGHT_FLANGE], dtype=torch.float32, device=DEVICE
    )

    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)

    # Training loop
    best_miou = -1.0
    best_model = None
    no_improve = 0

    history = {
        "epoch": [], "train_loss": [], "train_oa": [], "train_miou": [],
        "valid_loss": [], "valid_oa": [], "valid_miou": []
    }

    print(f"\n{'='*60}")
    print(f"Iniciando treinamento — {NUM_EPOCHS} épocas")
    print(f"{'='*60}\n")

    for epoch in range(1, NUM_EPOCHS + 1):
        # Train
        train_loss, train_oa, train_miou = train_epoch(
            model, train_loader, optimizer, class_weights, DEVICE
        )

        # Validate
        valid_loss, valid_oa, valid_miou, _ = eval_epoch(
            model, valid_loader, class_weights, DEVICE
        )

        scheduler.step()

        # Log
        history["epoch"].append(epoch)
        history["train_loss"].append(train_loss)
        history["train_oa"].append(train_oa)
        history["train_miou"].append(train_miou)
        history["valid_loss"].append(valid_loss)
        history["valid_oa"].append(valid_oa)
        history["valid_miou"].append(valid_miou)

        improved = ""
        if valid_miou > best_miou:
            best_miou = valid_miou
            best_model = copy.deepcopy(model)
            torch.save(best_model.state_dict(), MODEL_PATH)
            no_improve = 0
            improved = " ★ BEST"
        else:
            no_improve += 1

        print(
            f"Epoch {epoch:3d}/{NUM_EPOCHS} | "
            f"Train: loss={train_loss:.4f} OA={train_oa:.1f}% mIoU={train_miou:.1f}% | "
            f"Valid: loss={valid_loss:.4f} OA={valid_oa:.1f}% mIoU={valid_miou:.1f}%"
            f"{improved}"
        )

        if no_improve >= EARLY_STOP_PATIENCE:
            print(f"\nEarly stopping após {EARLY_STOP_PATIENCE} épocas sem melhora.")
            break

    # Salvar histórico
    pd.DataFrame(history).to_csv(HISTORY_PATH, index=False)
    print(f"\nHistórico salvo em: {HISTORY_PATH}")

    # Plotar curvas
    plot_training_curves(HISTORY_PATH, CURVES_PATH)

    # Avaliação final no teste
    print(f"\n{'='*60}")
    print("AVALIAÇÃO NO CONJUNTO DE TESTE")
    print(f"{'='*60}")

    best_model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    best_model.to(DEVICE)

    test_loss, test_oa, test_miou, test_metrics = eval_epoch(
        best_model, test_loader, class_weights, DEVICE
    )

    print(f"\nTest Loss: {test_loss:.4f}")
    print(f"Test OA:   {test_oa:.2f}%")
    print(f"Test mIoU: {test_miou:.2f}%")
    test_metrics.print_report()

    # Salvar confusion matrix
    save_confusion_matrix(test_metrics, CM_PATH)

    # Salvar relatório
    ious = test_metrics.class_iou()
    with open(REPORT_PATH, "w") as f:
        f.write(f"Test Overall Accuracy: {test_oa:.2f}%\n")
        f.write(f"Test mIoU: {test_miou:.2f}%\n\n")
        f.write("Per-class IoU:\n")
        for name, iou in zip(CLASS_NAMES, ious):
            f.write(f"  {name}: {100*iou:.2f}%\n")
        f.write(f"\nConfusion Matrix:\n{test_metrics.cm}\n")
        f.write(f"\nHiperparâmetros:\n")
        f.write(f"  NUM_POINTS: {NUM_POINTS}\n")
        f.write(f"  BATCH_SIZE: {BATCH_SIZE}\n")
        f.write(f"  NUM_EPOCHS: {NUM_EPOCHS}\n")
        f.write(f"  LEARNING_RATE: {LEARNING_RATE}\n")
        f.write(f"  WEIGHT_DECAY: {WEIGHT_DECAY}\n")
        f.write(f"  CLASS_WEIGHT_FLANGE: {CLASS_WEIGHT_FLANGE}\n")
        f.write(f"  FLANGE_RATIO: {FLANGE_RATIO}\n")
        f.write(f"  SEED: {SEED}\n")
        f.write(f"  EARLY_STOP_PATIENCE: {EARLY_STOP_PATIENCE}\n")
    print(f"Relatório salvo em: {REPORT_PATH}")

    t_total = time.time() - t_start
    print(f"\nTempo total: {t_total/60:.1f} minutos")


if __name__ == "__main__":
    main()
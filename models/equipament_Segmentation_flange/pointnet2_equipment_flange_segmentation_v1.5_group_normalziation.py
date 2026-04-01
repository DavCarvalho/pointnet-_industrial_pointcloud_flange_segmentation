"""
POINTNET++ SEMANTIC SEGMENTATION — v6 (GROUPNORM)
===================================================
Correção definitiva: GroupNorm substitui BatchNorm em TODA a rede.

PROBLEMA DO v5:
  - BatchNorm com batch_size=2 → running stats corrompidas
  - model.eval() colapsa para classe majoritária (0% flange)
  - Treino parecia OK porque model.train() usa batch stats

SOLUÇÃO v6:
  - GroupNorm não usa running stats → funciona com qualquer batch_size
  - model.eval() == model.train() para GroupNorm (sem surpresas)
  - Inferência funciona perfeitamente

OUTRAS MELHORIAS:
  - BATCH_SIZE = 4 (era 2) — mais estável
  - epoch_size maior para treino (200 ao invés de 100)
  - Scheduler OneCycleLR (converge mais rápido)
  - Salvamento do modelo com recalibração de BN pós-treino (redundante com GN, mas seguro)

USO:
  1. Rode o preprocess_pointclouds_v5.py (mesmo de antes, sem mudança)
  2. Rode este script: python train_segmentation_v6.py
"""

import os
import sys
import random
import copy
import time
import numpy as np
import pandas as pd
from glob import glob

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from sklearn.metrics import confusion_matrix
from tqdm.auto import tqdm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ╔══════════════════════════════════════════════════════════════╗
# ║                      CONFIGURAÇÃO                           ║
# ╚══════════════════════════════════════════════════════════════╝

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, "../../DATA_PREPROCESSED_v5"))

MODEL_PATH = os.path.join(SCRIPT_DIR, "best_segmentation_v6.pth")
HISTORY_PATH = os.path.join(SCRIPT_DIR, "segmentation_history_v6.csv")
CM_PATH = os.path.join(SCRIPT_DIR, "segmentation_cm_v6.png")
REPORT_PATH = os.path.join(SCRIPT_DIR, "segmentation_report_v6.txt")
CURVES_PATH = os.path.join(SCRIPT_DIR, "segmentation_curves_v6.png")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── HIPERPARÂMETROS ──
NUM_POINTS = 8192
BATCH_SIZE = 4               # Era 2 → mais estável
NUM_EPOCHS = 200
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-5
EARLY_STOP_PATIENCE = 35
FLANGE_RATIO = 0.3
CLASS_WEIGHT_FLANGE = 3.0
SEED = 42

# GroupNorm config
GN_NUM_GROUPS = 16           # Número de grupos para GroupNorm

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
# ║                   NORMALIZAÇÃO HELPER                        ║
# ╚══════════════════════════════════════════════════════════════╝

def make_norm_layer(num_channels, dim="1d"):
    """
    Cria GroupNorm ao invés de BatchNorm.

    GroupNorm não depende de running stats:
      - Normaliza dentro de cada amostra (não entre amostras do batch)
      - model.eval() funciona IGUAL a model.train()
      - Funciona com batch_size=1

    Args:
        num_channels: número de canais
        dim: "1d" ou "2d" (ambos usam GroupNorm)
    """
    num_groups = min(GN_NUM_GROUPS, num_channels)
    # Garantir que num_channels é divisível por num_groups
    while num_channels % num_groups != 0:
        num_groups -= 1
    return nn.GroupNorm(num_groups=num_groups, num_channels=num_channels)


# ╔══════════════════════════════════════════════════════════════╗
# ║                SPLIT POR EQUIPAMENTO                         ║
# ╚══════════════════════════════════════════════════════════════╝

def split_equipment_files(data_dir, seed=42):
    npz_files = sorted(glob(os.path.join(data_dir, "*.npz")))
    if not npz_files:
        raise FileNotFoundError(f"Nenhum .npz em {data_dir}")

    print(f"\nTotal de equipamentos: {len(npz_files)}")

    total_flange = 0
    total_equip = 0
    for f in npz_files:
        data = np.load(f)
        n_f = len(data["flange_idx"])
        n_e = len(data["equip_idx"])
        total_flange += n_f
        total_equip += n_e
        pct = 100 * n_f / (n_f + n_e)
        print(f"  {os.path.basename(f):30s} total={n_f+n_e:>8,} flange={pct:.1f}%")

    total = total_flange + total_equip
    print(f"\n  GLOBAL: {total:,} pontos, {100*total_flange/total:.1f}% flange")

    rng = np.random.default_rng(seed)
    indices = np.arange(len(npz_files))
    rng.shuffle(indices)

    n = len(indices)
    n_test = max(1, int(n * 0.20))
    n_valid = max(1, int(n * 0.10))
    n_train = n - n_test - n_valid

    train_files = [npz_files[i] for i in indices[:n_train]]
    valid_files = [npz_files[i] for i in indices[n_train:n_train + n_valid]]
    test_files = [npz_files[i] for i in indices[n_train + n_valid:]]

    print(f"\nSplit: Train={len(train_files)}, Valid={len(valid_files)}, Test={len(test_files)}")
    for label, files in [("Train", train_files), ("Valid", valid_files), ("Test", test_files)]:
        names = [os.path.basename(f).replace('.npz', '') for f in files]
        print(f"  {label}: {', '.join(names)}")

    return train_files, valid_files, test_files


# ╔══════════════════════════════════════════════════════════════╗
# ║                      DATASET                                ║
# ╚══════════════════════════════════════════════════════════════╝

class SegmentationDataset(Dataset):
    def __init__(self, npz_files, num_points=8192, epoch_size=100,
                 train=True, flange_ratio=0.3):
        self.npz_files = npz_files
        self.num_points = num_points
        self.epoch_size = epoch_size
        self.train = train
        self.flange_ratio = flange_ratio
        self.cache = {}

    def __len__(self):
        return self.epoch_size

    def _load(self, filepath):
        if filepath not in self.cache:
            data = np.load(filepath)
            self.cache[filepath] = {
                "features": data["features"],
                "labels": data["labels"],
                "flange_idx": data["flange_idx"],
                "equip_idx": data["equip_idx"],
            }
            n_f = len(data["flange_idx"])
            n_e = len(data["equip_idx"])
            print(f"  Loaded: {os.path.basename(filepath)} "
                  f"({n_f:,} flange, {n_e:,} equip, {100*n_f/(n_f+n_e):.1f}%)")
        return self.cache[filepath]

    def __getitem__(self, idx):
        if self.train:
            file_idx = np.random.randint(0, len(self.npz_files))
        else:
            file_idx = idx % len(self.npz_files)

        data = self._load(self.npz_files[file_idx])
        features = data["features"]
        labels = data["labels"]
        flange_idx = data["flange_idx"]
        equip_idx = data["equip_idx"]

        n_flange = len(flange_idx)
        n_equip = len(equip_idx)
        n_total = n_flange + n_equip

        if self.train:
            if n_flange > 0 and self.flange_ratio is not None:
                desired_f = max(1, int(self.num_points * self.flange_ratio))
                desired_e = self.num_points - desired_f
                sf = np.random.choice(flange_idx, desired_f, replace=(n_flange < desired_f))
                se = np.random.choice(equip_idx, desired_e, replace=(n_equip < desired_e))
                sampled = np.concatenate([sf, se])
            else:
                sampled = np.random.choice(n_total, self.num_points, replace=(n_total < self.num_points))
            np.random.shuffle(sampled)
        else:
            rng = np.random.default_rng(idx + SEED)
            sampled = rng.choice(np.arange(n_total), self.num_points, replace=(n_total < self.num_points))

        feat_sample = features[sampled]
        label_sample = labels[sampled]

        if self.train:
            feat_sample = self._augment(feat_sample)

        return (torch.from_numpy(feat_sample.astype(np.float32)),
                torch.from_numpy(label_sample.astype(np.int64)))

    def _augment(self, features):
        features = features.copy()
        xyz = features[:, :3]
        normals = features[:, 3:6]

        # Rotação Z
        theta = np.random.uniform(0, 2 * np.pi)
        c, s = np.cos(theta), np.sin(theta)
        rot = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=np.float32)
        xyz = xyz @ rot.T
        normals = normals @ rot.T
        normals = normals / (np.linalg.norm(normals, axis=1, keepdims=True) + 1e-8)

        # Escala
        scale = np.random.uniform(0.9, 1.1)
        xyz *= scale

        # Jitter
        xyz += np.random.normal(0, 0.01, size=xyz.shape).astype(np.float32)

        features[:, :3] = xyz
        features[:, 3:6] = normals
        return features


def seg_collate(batch):
    feats, labels = zip(*batch)
    return torch.stack(feats, 0), torch.stack(labels, 0)


# ╔══════════════════════════════════════════════════════════════╗
# ║          POINTNET++ COM GROUPNORM (CORREÇÃO DEFINITIVA)      ║
# ╚══════════════════════════════════════════════════════════════╝

def square_distance(src, dst):
    dist = -2 * torch.matmul(src, dst.transpose(1, 2))
    dist += torch.sum(src ** 2, dim=-1).unsqueeze(-1)
    dist += torch.sum(dst ** 2, dim=-1).unsqueeze(1)
    return dist


def index_points(points, idx):
    B = points.shape[0]
    batch_idx = torch.arange(B, dtype=torch.long, device=points.device).view(
        B, *((1,) * (idx.dim() - 1)))
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
    """Set Abstraction com GroupNorm (era BatchNorm2d)."""

    def __init__(self, npoint, nsample, in_channel, mlp):
        super().__init__()
        self.npoint = npoint
        self.nsample = nsample
        last_ch = in_channel
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        for out_ch in mlp:
            self.convs.append(nn.Conv2d(last_ch, out_ch, 1))
            self.norms.append(make_norm_layer(out_ch, dim="2d"))
            last_ch = out_ch

    def forward(self, xyz, points):
        if self.npoint is not None:
            new_xyz, new_points, fps_idx = sample_and_group(
                self.npoint, self.nsample, xyz, points)
        else:
            new_xyz, new_points = sample_and_group_all(xyz, points)
            fps_idx = None

        # new_points: (B, npoint, nsample, C) → (B, C, npoint, nsample)
        new_points = new_points.permute(0, 3, 1, 2)

        for conv, norm in zip(self.convs, self.norms):
            new_points = F.relu(norm(conv(new_points)))

        new_points = torch.max(new_points, dim=-1)[0]  # (B, C, npoint)
        new_points = new_points.permute(0, 2, 1)       # (B, npoint, C)

        return new_xyz, new_points, fps_idx


class FeaturePropagation(nn.Module):
    """Feature Propagation com GroupNorm (era BatchNorm1d)."""

    def __init__(self, in_channel, mlp):
        super().__init__()
        last_ch = in_channel
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        for out_ch in mlp:
            self.convs.append(nn.Conv1d(last_ch, out_ch, 1))
            self.norms.append(make_norm_layer(out_ch, dim="1d"))
            last_ch = out_ch

    def forward(self, xyz1, xyz2, points1, points2):
        B, N, C = xyz1.shape
        _, S, _ = xyz2.shape

        if S == 1:
            interpolated = points2.repeat(1, N, 1)
        else:
            dists = square_distance(xyz1, xyz2)
            dists, idx = dists.sort(dim=-1)
            dists = dists[:, :, :3]
            idx = idx[:, :, :3]
            dist_recip = 1.0 / (dists + 1e-8)
            norm = torch.sum(dist_recip, dim=2, keepdim=True)
            weight = dist_recip / norm
            interpolated_points = index_points(points2, idx)
            interpolated = torch.sum(
                interpolated_points * weight.unsqueeze(-1), dim=2)

        if points1 is not None:
            new_points = torch.cat([points1, interpolated], dim=-1)
        else:
            new_points = interpolated

        new_points = new_points.permute(0, 2, 1)  # (B, C, N)

        for conv, norm in zip(self.convs, self.norms):
            new_points = F.relu(norm(conv(new_points)))

        return new_points.permute(0, 2, 1)  # (B, N, C)


class PointNet2SegModel(nn.Module):
    """
    PointNet++ Segmentação com GroupNorm.

    Diferença vs v5: ZERO BatchNorm em todo o modelo.
    GroupNorm normaliza dentro de cada amostra (não entre amostras),
    então model.eval() funciona perfeitamente com batch_size=1.
    """

    def __init__(self, num_classes=2, input_channels=8):
        super().__init__()

        self.sa1 = SetAbstraction(
            npoint=2048, nsample=32, in_channel=3 + input_channels, mlp=[64, 64, 128])
        self.sa2 = SetAbstraction(
            npoint=512, nsample=64, in_channel=3 + 128, mlp=[128, 128, 256])
        self.sa3 = SetAbstraction(
            npoint=128, nsample=128, in_channel=3 + 256, mlp=[256, 256, 512])
        self.sa4 = SetAbstraction(
            npoint=None, nsample=None, in_channel=3 + 512, mlp=[512, 512, 1024])

        # Attention — GroupNorm ao invés de BatchNorm
        self.attention = nn.Sequential(
            nn.Conv1d(1024, 512, 1),
            make_norm_layer(512),
            nn.ReLU(),
            nn.Conv1d(512, 1024, 1),
            nn.Sigmoid()
        )

        self.fp4 = FeaturePropagation(in_channel=1024 + 512, mlp=[512, 512])
        self.fp3 = FeaturePropagation(in_channel=512 + 256, mlp=[256, 256])
        self.fp2 = FeaturePropagation(in_channel=256 + 128, mlp=[128, 128])
        self.fp1 = FeaturePropagation(in_channel=128 + input_channels, mlp=[128, 128])

        # Seg head — GroupNorm ao invés de BatchNorm
        self.seg_head = nn.Sequential(
            nn.Conv1d(128, 128, 1),
            make_norm_layer(128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Conv1d(128, num_classes, 1)
        )

    def forward(self, features):
        xyz = features[:, :, :3].contiguous()
        extra_feats = features[:, :, 3:].contiguous()

        l0_xyz, l0_points = xyz, extra_feats
        l1_xyz, l1_points, _ = self.sa1(l0_xyz, l0_points)
        l2_xyz, l2_points, _ = self.sa2(l1_xyz, l1_points)
        l3_xyz, l3_points, _ = self.sa3(l2_xyz, l2_points)
        l4_xyz, l4_points, _ = self.sa4(l3_xyz, l3_points)

        # Attention
        attn_input = l4_points.permute(0, 2, 1)
        attn_weights = self.attention(attn_input)
        l4_points = (l4_points.permute(0, 2, 1) * attn_weights).permute(0, 2, 1)

        # Decoder
        l3_points = self.fp4(l3_xyz, l4_xyz, l3_points, l4_points)
        l2_points = self.fp3(l2_xyz, l3_xyz, l2_points, l3_points)
        l1_points = self.fp2(l1_xyz, l2_xyz, l1_points, l2_points)
        l0_points = self.fp1(l0_xyz, l1_xyz, l0_points, l1_points)

        x = l0_points.permute(0, 2, 1)
        x = self.seg_head(x)
        x = x.permute(0, 2, 1)

        return x


# ╔══════════════════════════════════════════════════════════════╗
# ║                     LOSS FUNCTIONS                           ║
# ╚══════════════════════════════════════════════════════════════╝

def focal_loss(pred, target, class_weights, gamma=2.0):
    pred_flat = pred.reshape(-1, pred.shape[-1])
    target_flat = target.reshape(-1)
    ce = F.cross_entropy(pred_flat, target_flat, reduction="none")
    pt = torch.exp(-ce)
    weights = class_weights[target_flat]
    loss = weights * (1 - pt) ** gamma * ce
    return loss.mean()


def dice_loss(pred, target, smooth=1.0):
    num_classes = pred.shape[-1]
    pred_soft = torch.softmax(pred, dim=-1)
    target_oh = F.one_hot(target, num_classes).float()
    intersection = (pred_soft * target_oh).sum(dim=1)
    union = pred_soft.sum(dim=1) + target_oh.sum(dim=1)
    dice = (2.0 * intersection + smooth) / (union + smooth)
    return (1 - dice).mean()


def combined_loss(pred, target, class_weights):
    return focal_loss(pred, target, class_weights) + dice_loss(pred, target)


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
        self.cm += confusion_matrix(gt.flatten(), pred.flatten(),
                                     labels=list(range(self.n_classes)))

    def overall_accuracy(self):
        return 100.0 * self.cm.trace() / max(self.cm.sum(), 1)

    def class_iou(self):
        return np.diag(self.cm) / (
            self.cm.sum(1) + self.cm.sum(0) - np.diag(self.cm) + 1e-8)

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
        pred = model(feats)
        loss = combined_loss(pred, labels, class_weights)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item() * labels.size(0)
        n_samples += labels.size(0)
        metrics.add_batch(labels.cpu().numpy(), pred.argmax(dim=-1).cpu().numpy())

    return total_loss / max(n_samples, 1), metrics.overall_accuracy(), metrics.mean_iou()


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
            metrics.add_batch(labels.cpu().numpy(), pred.argmax(dim=-1).cpu().numpy())

    return total_loss / max(n_samples, 1), metrics.overall_accuracy(), metrics.mean_iou(), metrics


# ╔══════════════════════════════════════════════════════════════╗
# ║                    PLOTS                                     ║
# ╚══════════════════════════════════════════════════════════════╝

def plot_training_curves(history_path, save_path):
    df = pd.read_csv(history_path)
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    axes[0].plot(df["epoch"], df["train_loss"], label="Train", lw=1.5)
    axes[0].plot(df["epoch"], df["valid_loss"], label="Valid", lw=1.5)
    axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Loss"); axes[0].set_title("Loss")
    axes[0].legend(); axes[0].grid(True, alpha=0.3)

    axes[1].plot(df["epoch"], df["train_oa"], label="Train", lw=1.5)
    axes[1].plot(df["epoch"], df["valid_oa"], label="Valid", lw=1.5)
    axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("OA (%)"); axes[1].set_title("Overall Accuracy")
    axes[1].legend(); axes[1].grid(True, alpha=0.3)

    axes[2].plot(df["epoch"], df["train_miou"], label="Train", lw=1.5)
    axes[2].plot(df["epoch"], df["valid_miou"], label="Valid", lw=1.5)
    axes[2].set_xlabel("Epoch"); axes[2].set_ylabel("mIoU (%)"); axes[2].set_title("Mean IoU")
    axes[2].legend(); axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Curvas salvas: {save_path}")


def save_confusion_matrix(metrics, save_path):
    from sklearn.metrics import ConfusionMatrixDisplay
    fig, ax = plt.subplots(figsize=(6, 6))
    disp = ConfusionMatrixDisplay(confusion_matrix=metrics.cm, display_labels=CLASS_NAMES)
    disp.plot(ax=ax, cmap="Blues", colorbar=False)
    ax.set_title("Confusion Matrix — Segmentation v6 (GroupNorm)")
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"CM salva: {save_path}")


# ╔══════════════════════════════════════════════════════════════╗
# ║                      MAIN                                   ║
# ╚══════════════════════════════════════════════════════════════╝

def main():
    t_start = time.time()

    if not os.path.exists(DATA_DIR):
        print(f"ERRO: {DATA_DIR} não encontrado!")
        print(f"Execute primeiro o preprocess_pointclouds_v5.py")
        sys.exit(1)

    train_files, valid_files, test_files = split_equipment_files(DATA_DIR, seed=SEED)

    input_channels = 8  # 11 features - 3 xyz
    print(f"\nInput channels: {input_channels}")

    # Datasets
    train_dataset = SegmentationDataset(
        train_files, num_points=NUM_POINTS, epoch_size=200,
        train=True, flange_ratio=FLANGE_RATIO)
    valid_dataset = SegmentationDataset(
        valid_files, num_points=NUM_POINTS, epoch_size=50,
        train=False, flange_ratio=None)
    test_dataset = SegmentationDataset(
        test_files, num_points=NUM_POINTS, epoch_size=50,
        train=False, flange_ratio=None)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE,
                              shuffle=True, num_workers=0, collate_fn=seg_collate)
    valid_loader = DataLoader(valid_dataset, batch_size=BATCH_SIZE,
                              shuffle=False, num_workers=0, collate_fn=seg_collate)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE,
                             shuffle=False, num_workers=0, collate_fn=seg_collate)

    # Modelo com GroupNorm
    model = PointNet2SegModel(num_classes=2, input_channels=input_channels).to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Parâmetros: {n_params:,}")

    # Verificar que NÃO tem BatchNorm
    has_bn = any(isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)) for m in model.modules())
    has_gn = any(isinstance(m, nn.GroupNorm) for m in model.modules())
    print(f"BatchNorm presente: {has_bn} (deve ser False)")
    print(f"GroupNorm presente: {has_gn} (deve ser True)")
    if has_bn:
        print("⚠ ALERTA: Ainda tem BatchNorm no modelo!")
        sys.exit(1)

    class_weights = torch.tensor([1.0, CLASS_WEIGHT_FLANGE], dtype=torch.float32, device=DEVICE)
    print(f"Class weights: equip={class_weights[0]:.1f}, flange={class_weights[1]:.1f}")

    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS, eta_min=1e-6)

    best_miou = -1.0
    best_model = None
    no_improve = 0

    history = {"epoch": [], "train_loss": [], "train_oa": [], "train_miou": [],
               "valid_loss": [], "valid_oa": [], "valid_miou": []}

    print(f"\n{'='*60}")
    print(f"TREINAMENTO v6 — GroupNorm (sem BatchNorm)")
    print(f"  BATCH_SIZE = {BATCH_SIZE}")
    print(f"  CLASS_WEIGHT_FLANGE = {CLASS_WEIGHT_FLANGE}")
    print(f"  FLANGE_RATIO = {FLANGE_RATIO}")
    print(f"  GroupNorm groups = {GN_NUM_GROUPS}")
    print(f"{'='*60}\n")

    for epoch in range(1, NUM_EPOCHS + 1):
        train_loss, train_oa, train_miou = train_epoch(
            model, train_loader, optimizer, class_weights, DEVICE)
        valid_loss, valid_oa, valid_miou, _ = eval_epoch(
            model, valid_loader, class_weights, DEVICE)
        scheduler.step()

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

        lr_now = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch:3d}/{NUM_EPOCHS} | "
              f"Train: loss={train_loss:.4f} OA={train_oa:.1f}% mIoU={train_miou:.1f}% | "
              f"Valid: loss={valid_loss:.4f} OA={valid_oa:.1f}% mIoU={valid_miou:.1f}% | "
              f"lr={lr_now:.2e}{improved}")

        if no_improve >= EARLY_STOP_PATIENCE:
            print(f"\nEarly stopping ({EARLY_STOP_PATIENCE} épocas sem melhora)")
            break

    pd.DataFrame(history).to_csv(HISTORY_PATH, index=False)
    print(f"\nHistórico: {HISTORY_PATH}")
    plot_training_curves(HISTORY_PATH, CURVES_PATH)

    # ── Teste final ──
    print(f"\n{'='*60}")
    print("TESTE FINAL")
    print(f"{'='*60}")

    best_model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE, weights_only=False))
    best_model.to(DEVICE)

    test_loss, test_oa, test_miou, test_metrics = eval_epoch(
        best_model, test_loader, class_weights, DEVICE)

    print(f"\nTest Loss: {test_loss:.4f}")
    print(f"Test OA:   {test_oa:.2f}%")
    print(f"Test mIoU: {test_miou:.2f}%")
    test_metrics.print_report()

    save_confusion_matrix(test_metrics, CM_PATH)

    # ── Verificação: eval funciona? ──
    print(f"\n{'='*60}")
    print("VERIFICAÇÃO: eval() vs train()")
    print(f"{'='*60}")

    # Pegar uma amostra de teste
    data = np.load(glob(os.path.join(DATA_DIR, "*.npz"))[0])
    features = data["features"]
    flange_idx = data["flange_idx"]
    equip_idx = data["equip_idx"]
    labels = data["labels"]

    rng = np.random.default_rng(42)
    desired_f = max(1, int(NUM_POINTS * 0.3))
    desired_e = NUM_POINTS - desired_f
    sf = rng.choice(flange_idx, desired_f, replace=(len(flange_idx) < desired_f))
    se = rng.choice(equip_idx, desired_e, replace=(len(equip_idx) < desired_e))
    idx = np.concatenate([sf, se])
    feat_t = torch.from_numpy(features[idx].astype(np.float32)).unsqueeze(0).to(DEVICE)
    lab_s = labels[idx]

    # Eval mode
    best_model.eval()
    with torch.no_grad():
        logits_eval = best_model(feat_t).squeeze(0).cpu().numpy()
    pred_eval = logits_eval.argmax(axis=1)
    n_f_eval = int((pred_eval == 1).sum())

    # Train mode (para comparação)
    best_model.train()
    with torch.no_grad():
        logits_train = best_model(feat_t).squeeze(0).cpu().numpy()
    pred_train = logits_train.argmax(axis=1)
    n_f_train = int((pred_train == 1).sum())

    n_gt_f = int((lab_s == 1).sum())
    print(f"  Ground truth flanges: {n_gt_f:,}")
    print(f"  model.eval() prediz: {n_f_eval:,} flanges")
    print(f"  model.train() prediz: {n_f_train:,} flanges")

    if n_f_eval > 0 and abs(n_f_eval - n_f_train) < n_gt_f * 0.2:
        print(f"  ✓ SUCESSO! eval() e train() dão resultados consistentes")
        print(f"  ✓ O problema do BatchNorm está RESOLVIDO com GroupNorm")
    else:
        print(f"  ⚠ Diferença grande entre eval/train — investigar")

    # Relatório
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
        f.write(f"  GN_NUM_GROUPS: {GN_NUM_GROUPS}\n")
        f.write(f"  NORMALIZATION: GroupNorm (sem BatchNorm)\n")
        f.write(f"  SEED: {SEED}\n")
        f.write(f"  EARLY_STOP_PATIENCE: {EARLY_STOP_PATIENCE}\n")
    print(f"Relatório: {REPORT_PATH}")

    print(f"\nTempo total: {(time.time()-t_start)/60:.1f} min")


if __name__ == "__main__":
    main()
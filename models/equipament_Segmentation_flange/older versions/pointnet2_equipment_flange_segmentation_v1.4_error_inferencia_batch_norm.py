"""
POINTNET++ SEMANTIC SEGMENTATION — v4 (CARREGA .NPZ PRÉ-PROCESSADOS)
=====================================================================
Segmentação semântica ponto a ponto de flanges em nuvens de pontos industriais.

MUDANÇAS vs versão anterior:
  1. Carrega dos .npz pré-processados (features, labels, índices já prontos)
     → Sem recalcular normais/curvatura a cada load (era o gargalo de memória)
  2. CLASS_WEIGHT_FLANGE reduzido (3-5x ao invés de 50x)
     → Os dados agora já têm ~20% flange, não precisa compensar tanto
  3. FLANGE_RATIO no sampler reduzido (0.3 ao invés de 0.4)
     → Combinado com o rebalanceamento dos dados, evita over-fitting em flange
  4. Scheduler CosineAnnealing ao invés de StepLR
     → Convergência mais suave
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

# Diretório com os .npz pré-processados (saída do preprocess v5)
DATA_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, "../../DATA_PREPROCESSED_v5"))

# Saídas
MODEL_PATH = os.path.join(SCRIPT_DIR, "best_segmentation_v5.pth")
HISTORY_PATH = os.path.join(SCRIPT_DIR, "segmentation_history_v5.csv")
CM_PATH = os.path.join(SCRIPT_DIR, "segmentation_cm_v5.png")
REPORT_PATH = os.path.join(SCRIPT_DIR, "segmentation_report_v5.txt")
CURVES_PATH = os.path.join(SCRIPT_DIR, "segmentation_curves_v5.png")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ┌─────────────────────────────────────────────────────────────┐
# │  HIPERPARÂMETROS AJUSTADOS                                  │
# │                                                             │
# │  A ideia: os dados já vêm com ~20% flange (do preprocess   │
# │  v4 com ratio=4). Então NÃO precisamos de compensação       │
# │  agressiva na loss nem no sampler.                          │
# │                                                             │
# │  CLASS_WEIGHT_FLANGE = 3.0 (era 50!)                        │
# │    → Dados já rebalanceados, peso leve basta                │
# │  FLANGE_RATIO = 0.3 (era 0.4)                              │
# │    → Garante ver flange, mas sem dominar                    │
# └─────────────────────────────────────────────────────────────┘
NUM_POINTS = 8192
BATCH_SIZE = 2
NUM_EPOCHS = 200
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-5
EARLY_STOP_PATIENCE = 30
FLANGE_RATIO = 0.3           # fração de flanges por amostra (era 0.4)
CLASS_WEIGHT_FLANGE = 3.0    # peso da classe flange na loss (era 50!)
SEED = 42

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
# ║                SPLIT POR EQUIPAMENTO                         ║
# ╚══════════════════════════════════════════════════════════════╝

def split_equipment_files(data_dir, seed=42):
    """Divide os arquivos .npz em treino/validação/teste por equipamento."""
    npz_files = sorted(glob(os.path.join(data_dir, "*.npz")))
    if not npz_files:
        raise FileNotFoundError(f"Nenhum arquivo .npz encontrado em {data_dir}")

    print(f"\nTotal de equipamentos (.npz): {len(npz_files)}")

    # Mostrar estatísticas de cada arquivo
    total_flange = 0
    total_equip = 0
    for f in npz_files:
        data = np.load(f)
        n_f = len(data["flange_idx"])
        n_e = len(data["equip_idx"])
        total_flange += n_f
        total_equip += n_e
        pct = 100 * n_f / (n_f + n_e)
        print(f"  {os.path.basename(f):30s} "
              f"total={n_f+n_e:>8,} flange={pct:.1f}%")

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
    """
    Dataset que carrega dos .npz pré-processados.
    Sem recálculo de normais/curvatura — tudo já está pronto.
    """

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

    def _load_equipment(self, filepath):
        """Carrega do .npz — instantâneo, sem processamento pesado."""
        if filepath not in self.cache:
            data = np.load(filepath)
            self.cache[filepath] = {
                "features": data["features"],       # (N, 11)
                "labels": data["labels"],            # (N,)
                "flange_idx": data["flange_idx"],    # índices onde label==1
                "equip_idx": data["equip_idx"],      # índices onde label==0
            }
            n_f = len(data["flange_idx"])
            n_e = len(data["equip_idx"])
            print(f"  Carregado: {os.path.basename(filepath)} "
                  f"({n_f:,} flange, {n_e:,} equip, "
                  f"{100*n_f/(n_f+n_e):.1f}% flange)")
        return self.cache[filepath]

    def __getitem__(self, idx):
        # Escolher equipamento
        if self.train:
            file_idx = np.random.randint(0, len(self.npz_files))
        else:
            file_idx = idx % len(self.npz_files)

        data = self._load_equipment(self.npz_files[file_idx])
        features = data["features"]
        labels = data["labels"]
        flange_idx = data["flange_idx"]
        equip_idx = data["equip_idx"]

        n_flange = len(flange_idx)
        n_equip = len(equip_idx)
        n_total = n_flange + n_equip

        if self.train:
            # Treino: amostragem estratificada
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
                sampled = np.random.choice(n_total, self.num_points,
                                           replace=(n_total < self.num_points))
            np.random.shuffle(sampled)
        else:
            # Valid/test: distribuição natural, determinístico
            rng = np.random.default_rng(idx + SEED)
            all_idx = np.arange(n_total)
            sampled = rng.choice(all_idx, self.num_points,
                                 replace=(n_total < self.num_points))

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

        normals_norm = np.linalg.norm(normals, axis=1, keepdims=True) + 1e-8
        normals = normals / normals_norm

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
    feats = torch.stack(feats, dim=0)
    labels = torch.stack(labels, dim=0)
    return feats, labels


# ╔══════════════════════════════════════════════════════════════╗
# ║                POINTNET++ SEGMENTAÇÃO                        ║
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

        new_points = new_points.permute(0, 3, 1, 2)

        for conv, bn in zip(self.convs, self.bns):
            new_points = F.relu(bn(conv(new_points)))

        new_points = torch.max(new_points, dim=-1)[0]
        new_points = new_points.permute(0, 2, 1)

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
                interpolated_points * weight.unsqueeze(-1), dim=2
            )

        if points1 is not None:
            new_points = torch.cat([points1, interpolated], dim=-1)
        else:
            new_points = interpolated

        new_points = new_points.permute(0, 2, 1)

        for conv, bn in zip(self.convs, self.bns):
            new_points = F.relu(bn(conv(new_points)))

        return new_points.permute(0, 2, 1)


class PointNet2SegModel(nn.Module):
    """PointNet++ para segmentação semântica com encoder-decoder."""

    def __init__(self, num_classes=2, input_channels=8):
        super().__init__()

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

        self.attention = nn.Sequential(
            nn.Conv1d(1024, 512, 1),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Conv1d(512, 1024, 1),
            nn.Sigmoid()
        )

        self.fp4 = FeaturePropagation(in_channel=1024 + 512, mlp=[512, 512])
        self.fp3 = FeaturePropagation(in_channel=512 + 256, mlp=[256, 256])
        self.fp2 = FeaturePropagation(in_channel=256 + 128, mlp=[128, 128])
        self.fp1 = FeaturePropagation(
            in_channel=128 + input_channels, mlp=[128, 128]
        )

        self.seg_head = nn.Sequential(
            nn.Conv1d(128, 128, 1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Conv1d(128, num_classes, 1)
        )

    def forward(self, features):
        xyz = features[:, :, :3].contiguous()
        extra_feats = features[:, :, 3:].contiguous()

        l0_xyz = xyz
        l0_points = extra_feats

        l1_xyz, l1_points, _ = self.sa1(l0_xyz, l0_points)
        l2_xyz, l2_points, _ = self.sa2(l1_xyz, l1_points)
        l3_xyz, l3_points, _ = self.sa3(l2_xyz, l2_points)
        l4_xyz, l4_points, _ = self.sa4(l3_xyz, l3_points)

        attn_input = l4_points.permute(0, 2, 1)
        attn_weights = self.attention(attn_input)
        l4_points = (l4_points.permute(0, 2, 1) * attn_weights).permute(0, 2, 1)

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
    """Focal loss para lidar com desbalanceamento residual."""
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
    target_oh = F.one_hot(target, num_classes).float()

    intersection = (pred_soft * target_oh).sum(dim=1)
    union = pred_soft.sum(dim=1) + target_oh.sum(dim=1)
    dice = (2.0 * intersection + smooth) / (union + smooth)
    return (1 - dice).mean()


def combined_loss(pred, target, class_weights):
    """Combinação de focal + dice."""
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
        pred = model(feats)
        loss = combined_loss(pred, labels, class_weights)
        loss.backward()

        # Gradient clipping para estabilidade
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

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
    df = pd.read_csv(history_path)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    axes[0].plot(df["epoch"], df["train_loss"], label="Train", linewidth=1.5)
    axes[0].plot(df["epoch"], df["valid_loss"], label="Valid", linewidth=1.5)
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Loss")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(df["epoch"], df["train_oa"], label="Train", linewidth=1.5)
    axes[1].plot(df["epoch"], df["valid_oa"], label="Valid", linewidth=1.5)
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Overall Accuracy (%)")
    axes[1].set_title("Overall Accuracy")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

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
    from sklearn.metrics import ConfusionMatrixDisplay

    fig, ax = plt.subplots(figsize=(6, 6))
    disp = ConfusionMatrixDisplay(
        confusion_matrix=metrics.cm,
        display_labels=CLASS_NAMES
    )
    disp.plot(ax=ax, cmap="Blues", colorbar=False)
    ax.set_title("Confusion Matrix — Segmentation v5")
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Matriz de confusão salva em: {save_path}")


# ╔══════════════════════════════════════════════════════════════╗
# ║                      MAIN                                   ║
# ╚══════════════════════════════════════════════════════════════╝

def main():
    t_start = time.time()

    # Verificar se DATA_DIR existe e tem .npz
    if not os.path.exists(DATA_DIR):
        print(f"ERRO: Diretório {DATA_DIR} não encontrado!")
        print(f"Execute primeiro o preprocess_pointclouds_v4.py")
        sys.exit(1)

    # Split por equipamento
    train_files, valid_files, test_files = split_equipment_files(DATA_DIR, seed=SEED)

    # Features: xyz(3) + normals(3) + curvature(1) + rgb(3) + intensity(1) = 11
    # input_channels = 11 - 3 (xyz) = 8
    input_channels = 8
    print(f"\nInput channels (sem xyz): {input_channels}")

    # Datasets — carrega dos .npz (instantâneo!)
    train_dataset = SegmentationDataset(
        train_files, num_points=NUM_POINTS,
        epoch_size=100, train=True, flange_ratio=FLANGE_RATIO
    )
    valid_dataset = SegmentationDataset(
        valid_files, num_points=NUM_POINTS,
        epoch_size=50, train=False, flange_ratio=None
    )
    test_dataset = SegmentationDataset(
        test_files, num_points=NUM_POINTS,
        epoch_size=50, train=False, flange_ratio=None
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

    # Loss weights — REDUZIDO de 50 para 3
    class_weights = torch.tensor(
        [1.0, CLASS_WEIGHT_FLANGE], dtype=torch.float32, device=DEVICE
    )
    print(f"Class weights: equip={class_weights[0]:.1f}, flange={class_weights[1]:.1f}")

    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    # CosineAnnealing ao invés de StepLR — convergência mais suave
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=NUM_EPOCHS, eta_min=1e-6
    )

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
    print(f"  CLASS_WEIGHT_FLANGE = {CLASS_WEIGHT_FLANGE}")
    print(f"  FLANGE_RATIO (sampler) = {FLANGE_RATIO}")
    print(f"  Scheduler = CosineAnnealingLR")
    print(f"{'='*60}\n")

    for epoch in range(1, NUM_EPOCHS + 1):
        train_loss, train_oa, train_miou = train_epoch(
            model, train_loader, optimizer, class_weights, DEVICE
        )

        valid_loss, valid_oa, valid_miou, _ = eval_epoch(
            model, valid_loader, class_weights, DEVICE
        )

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
        print(
            f"Epoch {epoch:3d}/{NUM_EPOCHS} | "
            f"Train: loss={train_loss:.4f} OA={train_oa:.1f}% mIoU={train_miou:.1f}% | "
            f"Valid: loss={valid_loss:.4f} OA={valid_oa:.1f}% mIoU={valid_miou:.1f}% | "
            f"lr={lr_now:.2e}{improved}"
        )

        if no_improve >= EARLY_STOP_PATIENCE:
            print(f"\nEarly stopping após {EARLY_STOP_PATIENCE} épocas sem melhora.")
            break

    # Salvar histórico
    pd.DataFrame(history).to_csv(HISTORY_PATH, index=False)
    print(f"\nHistórico salvo em: {HISTORY_PATH}")

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

    save_confusion_matrix(test_metrics, CM_PATH)

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
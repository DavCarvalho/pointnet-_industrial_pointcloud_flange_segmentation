"""
KAIST CLASSIFIER PRETRAIN — v4 (PRÉ-PROCESSADO + RÁPIDO)
=========================================================

PROBLEMA RESOLVIDO vs v3:
  v3 lia .xyz lazy durante treino. Arquivos do KAIST vão de 360 pts até
  1.4M pts. Quando o DataLoader pegava um arquivo grande, o regex parser
  travava por minutos. Pior: cache acumulava na RAM e entrava em swap.

  Sintoma observado: 1 batch demorou 1+ hora sem progresso = travou.

SOLUÇÃO:
  1. Pré-processa TODO o dataset UMA VEZ (etapa offline):
     - Lê cada .xyz com numpy.loadtxt (rápido)
     - Sub-amostra agressivamente pra MAX_POINTS_PER_SAMPLE
     - Normaliza pra esfera unitária
     - Salva tudo em UM arquivo .npz (~50-200 MB total)

  2. Treino lê APENAS esse .npz, tudo em RAM, sem disco depois disso.

  3. NUM_POINTS reduzido pra 1024 (PointNet++ original usa isso pra ModelNet).
     FPS é quadrático: 2048→1024 acelera forward em ~4x.

  4. Limite de pontos por arquivo: 4096. Mesmo arquivos enormes
     são reduzidos pra isso UMA vez no preprocess. Memória total
     fica ~3700 × 4096 × 3 × 4 = ~180 MB (cabe sem suor).

USO:
  Primeira vez (faz preprocess + treina):
    python train_kaist_classifier_v4.py

  Segunda vez (já tem o .npz, só treina):
    python train_kaist_classifier_v4.py
  (detecta que o cache existe e pula o preprocess)

  Forçar reprocessar:
    python train_kaist_classifier_v4.py --rebuild
"""

import os
import sys
import re
import time
import copy
import json
import random
import argparse
import numpy as np
import pandas as pd
from glob import glob
from collections import Counter

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

from sklearn.metrics import confusion_matrix, classification_report
from tqdm.auto import tqdm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ╔══════════════════════════════════════════════════════════════╗
# ║                      CONFIGURAÇÃO                            ║
# ╚══════════════════════════════════════════════════════════════╝

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

KAIST_DIR = r"D:\nuvem\Repository_for_segmented_point_clouds_of_fitting_parts_in_the_process_plants_for_deep_learning_application"

PREPROCESSED_PATH = os.path.join(SCRIPT_DIR, "kaist_preprocessed.npz")
MODEL_PATH        = os.path.join(SCRIPT_DIR, "kaist_classifier.pth")
BACKBONE_PATH     = os.path.join(SCRIPT_DIR, "kaist_backbone.pth")
LABEL_MAP_PATH    = os.path.join(SCRIPT_DIR, "kaist_label_map.json")
HISTORY_PATH      = os.path.join(SCRIPT_DIR, "kaist_history.csv")
CURVES_PATH       = os.path.join(SCRIPT_DIR, "kaist_curves.png")
CM_PATH           = os.path.join(SCRIPT_DIR, "kaist_confusion_matrix.png")
REPORT_PATH       = os.path.join(SCRIPT_DIR, "kaist_report.txt")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── PRÉ-PROCESSAMENTO ──
# Quantos pontos guardar por amostra (depois do preprocess).
# Mais que isso é desperdício pra classificação de componente.
MAX_POINTS_PER_SAMPLE = 4096
MIN_POINTS_PER_FILE   = 50    # arquivos com menos pontos são descartados

# ── TREINO ──
NUM_POINTS          = 1024    # FPS é O(N²): 2048→1024 acelera forward 4x
BATCH_SIZE          = 32
NUM_EPOCHS          = 80
LEARNING_RATE       = 1e-3
WEIGHT_DECAY        = 1e-4
EARLY_STOP_PATIENCE = 20
SEED                = 42

LABEL_SMOOTHING     = 0.1

AUG_ROT_Z_RANGE_DEG = 15.0
AUG_JITTER_STD      = 0.005
AUG_SCALE_RANGE     = (0.95, 1.05)

USE_WEIGHTED_SAMPLER = True
USE_CLASS_WEIGHTS    = True
CLASS_WEIGHT_CAP     = 10.0

VALID_RATIO = 0.10
TEST_RATIO  = 0.10


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
# ║              LEITURA RÁPIDA DE .xyz                          ║
# ╚══════════════════════════════════════════════════════════════╝

_FLOAT_RE = re.compile(r"[-+]?(?:\d*\.\d+|\d+)(?:[eE][-+]?\d+)?")


def read_xyz_file_fast(path):
    """
    Lê .xyz usando numpy.loadtxt (rápido, em C).
    Fallback pra regex se o formato for esquisito.
    Retorna Nx3 float32 ou None.
    """
    # Tentativa 1: numpy.loadtxt
    try:
        pts = np.loadtxt(path, dtype=np.float32, usecols=(0, 1, 2))
        if pts.ndim == 1:
            return None
        if len(pts) < MIN_POINTS_PER_FILE:
            return None
        return pts
    except Exception:
        pass

    # Tentativa 2: parsing linha por linha (lento mas robusto)
    pts = []
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            nums = _FLOAT_RE.findall(line.strip())
            if len(nums) >= 3:
                try:
                    pts.append((float(nums[0]), float(nums[1]), float(nums[2])))
                except ValueError:
                    pass
    if len(pts) < MIN_POINTS_PER_FILE:
        return None
    return np.array(pts, dtype=np.float32)


def normalize_unit_sphere(pts):
    center = pts.mean(axis=0, keepdims=True)
    pts = pts - center
    radius = float(np.max(np.linalg.norm(pts, axis=1)))
    if radius < 1e-8:
        radius = 1.0
    return (pts / radius).astype(np.float32)


def downsample_to(pts, target):
    """Sub-amostra ou super-amostra pra exatamente `target` pontos."""
    n = len(pts)
    if n == target:
        return pts
    if n > target:
        idx = np.random.default_rng(SEED).choice(n, target, replace=False)
    else:
        idx = np.random.default_rng(SEED).choice(n, target, replace=True)
    return pts[idx]


# ╔══════════════════════════════════════════════════════════════╗
# ║              PREPROCESS — RODA UMA VEZ                       ║
# ╚══════════════════════════════════════════════════════════════╝

def preprocess_kaist(kaist_dir, output_path):
    """
    Lê todo o KAIST, normaliza, sub-amostra pra MAX_POINTS_PER_SAMPLE,
    salva num único .npz.

    Estrutura do .npz:
      points:       (N, MAX_POINTS_PER_SAMPLE, 3) float32
      labels:       (N,) int64
      label_to_idx: dict (salvo separado em json)
      idx_to_label: dict
    """
    if not os.path.isdir(kaist_dir):
        raise FileNotFoundError(f"Diretório KAIST não encontrado: {kaist_dir}")

    class_folders = sorted([
        d for d in os.listdir(kaist_dir)
        if os.path.isdir(os.path.join(kaist_dir, d))
    ])
    label_to_idx = {name: i for i, name in enumerate(class_folders)}
    idx_to_label = {i: name for name, i in label_to_idx.items()}

    print(f"\nPRÉ-PROCESSANDO KAIST")
    print(f"  De: {kaist_dir}")
    print(f"  Para: {output_path}")
    print(f"  Pontos por amostra: {MAX_POINTS_PER_SAMPLE}")
    print(f"  Classes: {len(class_folders)}")
    print()

    all_points = []
    all_labels = []
    n_skipped = 0
    t_start = time.time()

    for cls_name in class_folders:
        cls_dir = os.path.join(kaist_dir, cls_name)
        files = sorted(glob(os.path.join(cls_dir, "*.xyz")))
        cls_idx = label_to_idx[cls_name]
        n_loaded = 0
        t_cls = time.time()

        for f in files:
            pts = read_xyz_file_fast(f)
            if pts is None:
                n_skipped += 1
                continue
            pts = normalize_unit_sphere(pts)
            pts = downsample_to(pts, MAX_POINTS_PER_SAMPLE)
            all_points.append(pts)
            all_labels.append(cls_idx)
            n_loaded += 1

        elapsed = time.time() - t_cls
        print(f"  {cls_name:<20s} {n_loaded:>5d}/{len(files):>5d} arquivos  ({elapsed:.1f}s)")

    points = np.stack(all_points, axis=0).astype(np.float32)
    labels = np.array(all_labels, dtype=np.int64)

    total_mb = points.nbytes / 1024 / 1024
    print(f"\n  Total carregado: {len(points):,} amostras")
    print(f"  Pulados:         {n_skipped}")
    print(f"  Tamanho na RAM:  {total_mb:.1f} MB")
    print(f"  Shape:           {points.shape}")
    print(f"  Tempo total:     {(time.time()-t_start)/60:.1f} min")

    print(f"\n  Salvando em {output_path}...")
    np.savez(output_path,
             points=points,
             labels=labels)
    file_mb = os.path.getsize(output_path) / 1024 / 1024
    print(f"  Arquivo: {file_mb:.1f} MB")

    # Salva label map
    with open(LABEL_MAP_PATH, "w") as f:
        json.dump({"label_to_idx": label_to_idx,
                   "idx_to_label": {str(k): v for k, v in idx_to_label.items()}},
                  f, indent=2)
    print(f"  Label map: {LABEL_MAP_PATH}")

    return points, labels, label_to_idx, idx_to_label


def load_preprocessed(path):
    print(f"Carregando dataset pré-processado: {path}")
    data = np.load(path)
    points = data["points"]
    labels = data["labels"]

    with open(LABEL_MAP_PATH, "r") as f:
        lm = json.load(f)
    label_to_idx = lm["label_to_idx"]
    idx_to_label = {int(k): v for k, v in lm["idx_to_label"].items()}

    print(f"  Shape:   {points.shape}")
    print(f"  RAM:     {points.nbytes / 1024 / 1024:.1f} MB")
    print(f"  Classes: {len(label_to_idx)}")
    return points, labels, label_to_idx, idx_to_label


# ╔══════════════════════════════════════════════════════════════╗
# ║              SPLIT ESTRATIFICADO                             ║
# ╚══════════════════════════════════════════════════════════════╝

def stratified_split(labels, valid_ratio, test_ratio, seed=42):
    """Retorna índices (train, valid, test) estratificados por classe."""
    rng = np.random.default_rng(seed)
    train_idx, valid_idx, test_idx = [], [], []
    for cls in np.unique(labels):
        cls_idx = np.where(labels == cls)[0]
        rng.shuffle(cls_idx)
        n = len(cls_idx)
        n_test = max(1, int(n * test_ratio))
        n_valid = max(1, int(n * valid_ratio))
        n_train = n - n_test - n_valid
        if n_train <= 0:
            train_idx.extend(cls_idx.tolist())
            valid_idx.append(cls_idx[0])
            test_idx.append(cls_idx[0])
            continue
        train_idx.extend(cls_idx[:n_train].tolist())
        valid_idx.extend(cls_idx[n_train:n_train + n_valid].tolist())
        test_idx.extend(cls_idx[n_train + n_valid:].tolist())
    return np.array(train_idx), np.array(valid_idx), np.array(test_idx)


def compute_class_weights(labels, num_classes, cap=10.0):
    counts = Counter(labels.tolist())
    weights = np.zeros(num_classes, dtype=np.float32)
    total = sum(counts.values())
    for c in range(num_classes):
        n_c = counts.get(c, 1)
        weights[c] = total / (num_classes * n_c)
    weights = np.minimum(weights, cap)
    weights = weights * (num_classes / weights.sum())
    return weights, counts


# ╔══════════════════════════════════════════════════════════════╗
# ║                      DATASET                                 ║
# ╚══════════════════════════════════════════════════════════════╝

class KaistCachedDataset(Dataset):
    """
    Dataset que opera em pontos JÁ PRÉ-CARREGADOS na RAM.
    Sem IO durante o treino.
    """
    def __init__(self, points, labels, indices, num_points=1024, train=True):
        self.points = points        # array compartilhado (todas amostras)
        self.labels = labels        # array compartilhado
        self.indices = indices      # quais amostras esse split usa
        self.num_points = num_points
        self.train = train

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        global_idx = self.indices[i]
        pts = self.points[global_idx]    # (MAX_POINTS_PER_SAMPLE, 3)
        label = int(self.labels[global_idx])

        N = len(pts)
        if N >= self.num_points:
            sel = np.random.choice(N, self.num_points, replace=False)
        else:
            sel = np.random.choice(N, self.num_points, replace=True)
        pts = pts[sel].copy()

        if self.train:
            pts = self._augment(pts)

        return torch.from_numpy(pts), torch.tensor(label, dtype=torch.long)

    def _augment(self, pts):
        max_rad = np.deg2rad(AUG_ROT_Z_RANGE_DEG)
        theta = np.random.uniform(-max_rad, max_rad)
        c, s = np.cos(theta), np.sin(theta)
        rot = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=np.float32)
        pts = pts @ rot.T
        scale = np.random.uniform(*AUG_SCALE_RANGE)
        pts = pts * scale
        pts += np.random.normal(0, AUG_JITTER_STD, pts.shape).astype(np.float32)
        return pts


# ╔══════════════════════════════════════════════════════════════╗
# ║              MODELO COM BATCHNORM                            ║
# ╚══════════════════════════════════════════════════════════════╝

def square_distance(src, dst):
    d = -2 * torch.matmul(src, dst.transpose(1, 2))
    d += torch.sum(src ** 2, -1).unsqueeze(-1)
    d += torch.sum(dst ** 2, -1).unsqueeze(1)
    return d

def index_points(pts, idx):
    B = pts.shape[0]
    bi = torch.arange(B, device=pts.device).view(B, *((1,) * (idx.dim() - 1)))
    return pts[bi, idx, :]

def farthest_point_sample(xyz, npoint):
    B, N, C = xyz.shape
    npoint = min(npoint, N)
    centroids = torch.zeros(B, npoint, dtype=torch.long, device=xyz.device)
    dist = torch.ones(B, N, device=xyz.device) * 1e10
    far = torch.randint(0, N, (B,), dtype=torch.long, device=xyz.device)
    bi = torch.arange(B, dtype=torch.long, device=xyz.device)
    for i in range(npoint):
        centroids[:, i] = far
        cen = xyz[bi, far, :].view(B, 1, C)
        d = torch.sum((xyz - cen) ** 2, -1)
        mask = d < dist
        dist[mask] = d[mask]
        far = torch.max(dist, -1)[1]
    return centroids

def sample_and_group(npoint, nsample, xyz, points):
    B, N, C = xyz.shape
    npoint = min(npoint, N) if npoint else N
    fps = farthest_point_sample(xyz, npoint)
    new_xyz = index_points(xyz, fps)
    nsample = min(nsample, N)
    gidx = square_distance(new_xyz, xyz).argsort()[:, :, :nsample]
    gnorm = index_points(xyz, gidx) - new_xyz.unsqueeze(2)
    if points is not None:
        new_pts = torch.cat([gnorm, index_points(points, gidx)], -1)
    else:
        new_pts = gnorm
    return new_xyz, new_pts, fps

def sample_and_group_all(xyz, points):
    B, N, C = xyz.shape
    new_xyz = torch.zeros(B, 1, C, device=xyz.device)
    gnorm = xyz.view(B, 1, N, C)
    new_pts = torch.cat([gnorm, points.view(B, 1, N, -1)], -1) if points is not None else gnorm
    return new_xyz, new_pts


class SetAbstraction(nn.Module):
    """BatchNorm + bias=False (BN absorve o bias)."""
    def __init__(self, npoint, nsample, in_channel, mlp):
        super().__init__()
        self.npoint, self.nsample = npoint, nsample
        lc = in_channel
        self.convs, self.norms = nn.ModuleList(), nn.ModuleList()
        for oc in mlp:
            self.convs.append(nn.Conv2d(lc, oc, 1, bias=False))
            self.norms.append(nn.BatchNorm2d(oc))
            lc = oc
    def forward(self, xyz, pts):
        if self.npoint:
            new_xyz, new_pts, fps = sample_and_group(self.npoint, self.nsample, xyz, pts)
        else:
            new_xyz, new_pts = sample_and_group_all(xyz, pts); fps = None
        new_pts = new_pts.permute(0, 3, 1, 2)
        for c, n in zip(self.convs, self.norms):
            new_pts = F.relu(n(c(new_pts)))
        return new_xyz, torch.max(new_pts, -1)[0].permute(0, 2, 1), fps


class PointNet2Backbone(nn.Module):
    """
    npoints reduzidos pra bater com NUM_POINTS=1024:
      sa1: 1024 → 512
      sa2:  512 → 128
      sa3:  128 → 32
      sa4: pooling global

    Estrutura de pesos (in_channels/out_channels) IDÊNTICA ao v3,
    então o backbone continua compatível com o segmentador v9.
    Só os npoints mudam, e isso não afeta as shapes dos pesos.
    """
    def __init__(self, input_channels=0):
        super().__init__()
        self.sa1 = SetAbstraction(512, 32, 3 + input_channels, [64, 64, 128])
        self.sa2 = SetAbstraction(128, 64, 3 + 128, [128, 128, 256])
        self.sa3 = SetAbstraction(32, 128, 3 + 256, [256, 256, 512])
        self.sa4 = SetAbstraction(None, None, 3 + 512, [512, 512, 1024])

    def forward(self, x):
        xyz = x[:, :, :3].contiguous()
        ex = x[:, :, 3:].contiguous() if x.shape[-1] > 3 else None
        l1x, l1p, _ = self.sa1(xyz, ex)
        l2x, l2p, _ = self.sa2(l1x, l1p)
        l3x, l3p, _ = self.sa3(l2x, l2p)
        l4x, l4p, _ = self.sa4(l3x, l3p)
        return l4p.squeeze(1)


class KaistClassifier(nn.Module):
    def __init__(self, num_classes, input_channels=0):
        super().__init__()
        self.backbone = PointNet2Backbone(input_channels=input_channels)
        self.head = nn.Sequential(
            nn.Linear(1024, 512, bias=False),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(512, 256, bias=False),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, num_classes),
        )
    def forward(self, x):
        return self.head(self.backbone(x))


# ╔══════════════════════════════════════════════════════════════╗
# ║                    TREINO / EVAL                             ║
# ╚══════════════════════════════════════════════════════════════╝

def compute_loss(logits, labels, class_weights, label_smoothing=0.1):
    return F.cross_entropy(
        logits, labels,
        weight=class_weights,
        label_smoothing=label_smoothing
    )


def train_epoch(model, loader, optimizer, class_weights, device):
    model.train()
    total_loss, correct, n = 0.0, 0, 0
    for pts, labels in tqdm(loader, desc="Train", leave=False):
        pts, labels = pts.to(device).float(), labels.to(device)
        optimizer.zero_grad()
        logits = model(pts)
        loss = compute_loss(logits, labels, class_weights, LABEL_SMOOTHING)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item() * labels.size(0)
        correct += (logits.argmax(-1) == labels).sum().item()
        n += labels.size(0)
    return total_loss / max(n, 1), 100 * correct / max(n, 1)


@torch.no_grad()
def eval_epoch(model, loader, class_weights, device):
    model.eval()
    total_loss, correct, n = 0.0, 0, 0
    all_preds, all_labels = [], []
    for pts, labels in tqdm(loader, desc="Eval", leave=False):
        pts, labels = pts.to(device).float(), labels.to(device)
        logits = model(pts)
        loss = compute_loss(logits, labels, class_weights, LABEL_SMOOTHING)
        preds = logits.argmax(-1)
        total_loss += loss.item() * labels.size(0)
        correct += (preds == labels).sum().item()
        n += labels.size(0)
        all_preds.append(preds.cpu().numpy())
        all_labels.append(labels.cpu().numpy())

    avg_loss = total_loss / max(n, 1)
    acc = 100 * correct / max(n, 1)
    return avg_loss, acc, np.concatenate(all_preds), np.concatenate(all_labels)


def per_class_accuracy(preds, labels, idx_to_label):
    result = {}
    for cls_idx, cls_name in idx_to_label.items():
        mask = labels == cls_idx
        if mask.sum() == 0:
            result[cls_name] = float('nan')
        else:
            result[cls_name] = 100 * (preds[mask] == cls_idx).mean()
    return result


def print_per_class(preds, labels, idx_to_label):
    pca = per_class_accuracy(preds, labels, idx_to_label)
    sorted_items = sorted(pca.items(),
                          key=lambda x: -x[1] if not np.isnan(x[1]) else -1)
    print("    Per-class accuracy:")
    for name, acc in sorted_items:
        bar = "█" * int(acc / 5) if not np.isnan(acc) else ""
        print(f"      {name:<20s} {acc:5.1f}%  {bar}")


def plot_curves(history_path, save_path):
    df = pd.read_csv(history_path)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].plot(df["epoch"], df["train_loss"], label="Train", lw=1.5)
    axes[0].plot(df["epoch"], df["valid_loss"], label="Valid", lw=1.5)
    axes[0].set_title("Loss"); axes[0].legend(); axes[0].grid(True, alpha=0.3)
    axes[1].plot(df["epoch"], df["train_acc"], label="Train", lw=1.5)
    axes[1].plot(df["epoch"], df["valid_acc"], label="Valid", lw=1.5)
    axes[1].set_title("Accuracy (%)"); axes[1].legend(); axes[1].grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight"); plt.close()


def save_confusion_matrix(cm, idx_to_label, save_path):
    n = len(idx_to_label)
    fig, ax = plt.subplots(figsize=(max(8, n*0.6), max(7, n*0.5)))
    cm_norm = cm.astype(float) / np.maximum(cm.sum(axis=1, keepdims=True), 1)
    im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(n)); ax.set_yticks(range(n))
    labels = [idx_to_label[i] for i in range(n)]
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticklabels(labels)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    ax.set_title("KAIST Classifier — Confusion Matrix")
    for i in range(n):
        for j in range(n):
            ax.text(j, i, f"{int(cm[i, j])}", ha="center", va="center",
                    color="white" if cm_norm[i, j] > 0.5 else "black",
                    fontsize=7)
    plt.colorbar(im, ax=ax)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight"); plt.close()


# ╔══════════════════════════════════════════════════════════════╗
# ║                          MAIN                                ║
# ╚══════════════════════════════════════════════════════════════╝

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rebuild", action="store_true",
                        help="Reprocessa o dataset mesmo se .npz já existir")
    args = parser.parse_args()

    t_start = time.time()

    # 1) Pré-processamento (uma vez)
    if args.rebuild or not os.path.isfile(PREPROCESSED_PATH):
        if args.rebuild:
            print("--rebuild: forçando reprocessamento")
        points, labels, label_to_idx, idx_to_label = preprocess_kaist(
            KAIST_DIR, PREPROCESSED_PATH
        )
    else:
        points, labels, label_to_idx, idx_to_label = load_preprocessed(PREPROCESSED_PATH)

    num_classes = len(label_to_idx)

    # 2) Split estratificado
    train_idx, valid_idx, test_idx = stratified_split(
        labels, VALID_RATIO, TEST_RATIO, SEED
    )
    print(f"\nSplit: Train={len(train_idx)}, Valid={len(valid_idx)}, Test={len(test_idx)}")

    # 3) Class weights
    train_labels = labels[train_idx]
    class_weights_np, train_counts = compute_class_weights(
        train_labels, num_classes, cap=CLASS_WEIGHT_CAP
    )
    print(f"\nClass weights (cap={CLASS_WEIGHT_CAP}):")
    for cls_idx in range(num_classes):
        n_train = train_counts.get(cls_idx, 0)
        print(f"  {idx_to_label[cls_idx]:<20s} n={n_train:>5d}  w={class_weights_np[cls_idx]:.3f}")

    class_weights = torch.tensor(class_weights_np, dtype=torch.float32, device=DEVICE) \
        if USE_CLASS_WEIGHTS else None

    # 4) Datasets
    train_ds = KaistCachedDataset(points, labels, train_idx, NUM_POINTS, train=True)
    valid_ds = KaistCachedDataset(points, labels, valid_idx, NUM_POINTS, train=False)
    test_ds  = KaistCachedDataset(points, labels, test_idx,  NUM_POINTS, train=False)

    # 5) Sampler balanceado
    if USE_WEIGHTED_SAMPLER:
        sample_weights = np.array(
            [1.0 / train_counts[lbl] for lbl in train_labels],
            dtype=np.float64
        )
        sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(train_idx),
            replacement=True
        )
        train_loader = DataLoader(train_ds, BATCH_SIZE, sampler=sampler,
                                  num_workers=0, drop_last=True)
        print(f"\n  WeightedRandomSampler ativo")
    else:
        train_loader = DataLoader(train_ds, BATCH_SIZE, shuffle=True,
                                  num_workers=0, drop_last=True)

    valid_loader = DataLoader(valid_ds, BATCH_SIZE, shuffle=False, num_workers=0)
    test_loader  = DataLoader(test_ds,  BATCH_SIZE, shuffle=False, num_workers=0)

    # 6) Modelo
    model = KaistClassifier(num_classes=num_classes, input_channels=0).to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"\nParâmetros: {n_params:,}")
    print(f"Device: {DEVICE}")

    has_gn = any(isinstance(m, nn.GroupNorm) for m in model.modules())
    has_bn = any(isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)) for m in model.modules())
    print(f"GroupNorm? {has_gn}  (deve ser False)")
    print(f"BatchNorm? {has_bn}  (deve ser True)")
    assert not has_gn and has_bn

    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE,
                            weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS,
                                                     eta_min=1e-6)

    # 7) Loop
    best_acc, best_state, no_improve = -1.0, None, 0
    history = {"epoch": [], "train_loss": [], "train_acc": [],
               "valid_loss": [], "valid_acc": []}

    print(f"\n{'='*60}\nTREINANDO KAIST v4 (BatchNorm + cache pré-processado)\n{'='*60}")
    print(f"  NUM_POINTS:         {NUM_POINTS}")
    print(f"  BATCH_SIZE:         {BATCH_SIZE}")
    print(f"  Augmentação:        ROT_Z=±{AUG_ROT_Z_RANGE_DEG}° jitter={AUG_JITTER_STD}")
    print()

    for epoch in range(1, NUM_EPOCHS + 1):
        t_ep = time.time()
        tl, ta = train_epoch(model, train_loader, optimizer, class_weights, DEVICE)
        vl, va, vp, vgt = eval_epoch(model, valid_loader, class_weights, DEVICE)
        scheduler.step()
        ep_time = time.time() - t_ep

        history["epoch"].append(epoch)
        history["train_loss"].append(tl); history["train_acc"].append(ta)
        history["valid_loss"].append(vl); history["valid_acc"].append(va)

        tag = ""
        if va > best_acc:
            best_acc = va
            best_state = copy.deepcopy(model.state_dict())
            no_improve = 0
            tag = " ★ BEST"
            torch.save(best_state, MODEL_PATH)
            backbone_state = {k.replace("backbone.", ""): v
                              for k, v in best_state.items()
                              if k.startswith("backbone.")}
            torch.save(backbone_state, BACKBONE_PATH)
        else:
            no_improve += 1

        lr = optimizer.param_groups[0]["lr"]
        print(f"Epoch {epoch:3d}/{NUM_EPOCHS} ({ep_time:.0f}s) | "
              f"T loss={tl:.4f} acc={ta:.2f}% | "
              f"V loss={vl:.4f} acc={va:.2f}% | lr={lr:.2e}{tag}")

        if epoch == 1 or epoch % 5 == 0:
            print_per_class(vp, vgt, idx_to_label)

        if no_improve >= EARLY_STOP_PATIENCE:
            print(f"\nEarly stopping após {EARLY_STOP_PATIENCE} épocas sem melhora")
            break

    pd.DataFrame(history).to_csv(HISTORY_PATH, index=False)
    plot_curves(HISTORY_PATH, CURVES_PATH)

    # 8) Teste final
    print(f"\n{'='*60}\nTESTE FINAL\n{'='*60}")
    model.load_state_dict(best_state)
    test_loss, test_acc, preds, gts = eval_epoch(model, test_loader, class_weights, DEVICE)
    print(f"\nTest loss = {test_loss:.4f}  |  Test acc = {test_acc:.2f}%")
    print()
    print_per_class(preds, gts, idx_to_label)

    cm = confusion_matrix(gts, preds, labels=list(range(num_classes)))
    save_confusion_matrix(cm, idx_to_label, CM_PATH)

    target_names = [idx_to_label[i] for i in range(num_classes)]
    report = classification_report(gts, preds, target_names=target_names,
                                    zero_division=0)
    print("\n" + report)

    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        f.write("KAIST CLASSIFIER REPORT v4\n" + "="*60 + "\n\n")
        f.write(f"Classes: {num_classes}\n")
        f.write(f"NUM_POINTS: {NUM_POINTS}\n")
        f.write(f"Best valid acc: {best_acc:.2f}%\n")
        f.write(f"Test acc:       {test_acc:.2f}%\n\n")
        f.write("Per-class accuracy (test):\n")
        for name, acc in per_class_accuracy(preds, gts, idx_to_label).items():
            f.write(f"  {name:<20s} {acc:5.1f}%\n")
        f.write("\n" + report)
    print(f"\nRelatório: {REPORT_PATH}")
    print(f"Backbone:  {BACKBONE_PATH}")
    print(f"Tempo:     {(time.time()-t_start)/60:.1f} min")


if __name__ == "__main__":
    main()
"""
CORREÇÃO DO MODELO — Recalibração de BatchNorm + Inferência
=============================================================
O modelo foi treinado com batch_size=2, causando running stats
do BatchNorm corrompidas. Este script:

  1. Recalibra as running stats fazendo forward passes nos dados de treino
  2. Salva o modelo corrigido
  3. Roda inferência com o modelo corrigido
  4. Gera visualizações para artigo

COMO FUNCIONA:
  - Reseta running_mean e running_var de todos os BatchNorms
  - Faz N forward passes com batch_size=2 (mínimo para BN funcionar)
     usando amostras estratificadas (como no treino)
  - As novas running stats convergem para valores estáveis
  - Salva o modelo com stats corrigidas
  - Inferência agora funciona em eval() normalmente

USO:
  python fix_batchnorm_and_infer.py
"""

import os
import sys
import time
import numpy as np
from glob import glob

import torch
import torch.nn as nn
import torch.nn.functional as F

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

from sklearn.metrics import confusion_matrix


# ╔══════════════════════════════════════════════════════════════╗
# ║                      CONFIGURAÇÃO                           ║
# ╚══════════════════════════════════════════════════════════════╝

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

DATA_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, "../../DATA_PREPROCESSED_v5"))
MODEL_PATH = os.path.join(SCRIPT_DIR, "best_segmentation_v5.pth")
FIXED_MODEL_PATH = os.path.join(SCRIPT_DIR, "best_segmentation_v5_fixed.pth")
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "article_figures")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Recalibração
CALIBRATION_BATCHES = 500      # Número de batches para recalibrar BN
CALIBRATION_BATCH_SIZE = 2     # Mínimo para BN funcionar em train mode
NUM_POINTS = 8192
FLANGE_RATIO = 0.3             # Mesmo do treino
SEED = 42

# Inferência
NUM_VOTES = 10
OVERLAP_FACTOR = 3.0

# Visualização
DPI = 300
POINT_SIZE = 0.3
MAX_POINTS_RENDER = 200_000

COLOR_EQUIPMENT = np.array([0.2, 0.4, 0.8])
COLOR_FLANGE = np.array([0.9, 0.3, 0.1])
COLOR_ERROR_FN = np.array([1.0, 0.0, 0.0])
COLOR_ERROR_FP = np.array([1.0, 0.8, 0.0])
COLOR_CORRECT = np.array([0.85, 0.85, 0.85])

CLASS_NAMES = ["equipment", "flange"]

print(f"Device: {DEVICE}")


# ╔══════════════════════════════════════════════════════════════╗
# ║                      MODELO                                 ║
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
            new_xyz, new_points, fps_idx = sample_and_group(self.npoint, self.nsample, xyz, points)
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
            interpolated = torch.sum(interpolated_points * weight.unsqueeze(-1), dim=2)
        if points1 is not None:
            new_points = torch.cat([points1, interpolated], dim=-1)
        else:
            new_points = interpolated
        new_points = new_points.permute(0, 2, 1)
        for conv, bn in zip(self.convs, self.bns):
            new_points = F.relu(bn(conv(new_points)))
        return new_points.permute(0, 2, 1)

class PointNet2SegModel(nn.Module):
    def __init__(self, num_classes=2, input_channels=8):
        super().__init__()
        self.sa1 = SetAbstraction(npoint=2048, nsample=32, in_channel=3+input_channels, mlp=[64,64,128])
        self.sa2 = SetAbstraction(npoint=512, nsample=64, in_channel=3+128, mlp=[128,128,256])
        self.sa3 = SetAbstraction(npoint=128, nsample=128, in_channel=3+256, mlp=[256,256,512])
        self.sa4 = SetAbstraction(npoint=None, nsample=None, in_channel=3+512, mlp=[512,512,1024])
        self.attention = nn.Sequential(nn.Conv1d(1024,512,1), nn.BatchNorm1d(512), nn.ReLU(), nn.Conv1d(512,1024,1), nn.Sigmoid())
        self.fp4 = FeaturePropagation(in_channel=1024+512, mlp=[512,512])
        self.fp3 = FeaturePropagation(in_channel=512+256, mlp=[256,256])
        self.fp2 = FeaturePropagation(in_channel=256+128, mlp=[128,128])
        self.fp1 = FeaturePropagation(in_channel=128+input_channels, mlp=[128,128])
        self.seg_head = nn.Sequential(nn.Conv1d(128,128,1), nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(0.5), nn.Conv1d(128,2,1))
    def forward(self, features):
        xyz = features[:,:,:3].contiguous()
        extra_feats = features[:,:,3:].contiguous()
        l0_xyz, l0_points = xyz, extra_feats
        l1_xyz, l1_points, _ = self.sa1(l0_xyz, l0_points)
        l2_xyz, l2_points, _ = self.sa2(l1_xyz, l1_points)
        l3_xyz, l3_points, _ = self.sa3(l2_xyz, l2_points)
        l4_xyz, l4_points, _ = self.sa4(l3_xyz, l3_points)
        attn_input = l4_points.permute(0,2,1)
        attn_weights = self.attention(attn_input)
        l4_points = (l4_points.permute(0,2,1) * attn_weights).permute(0,2,1)
        l3_points = self.fp4(l3_xyz, l4_xyz, l3_points, l4_points)
        l2_points = self.fp3(l2_xyz, l3_xyz, l2_points, l3_points)
        l1_points = self.fp2(l1_xyz, l2_xyz, l1_points, l2_points)
        l0_points = self.fp1(l0_xyz, l1_xyz, l0_points, l1_points)
        x = l0_points.permute(0,2,1)
        x = self.seg_head(x)
        x = x.permute(0,2,1)
        return x


# ╔══════════════════════════════════════════════════════════════╗
# ║           PASSO 1: RECALIBRAR BATCHNORM                     ║
# ╚══════════════════════════════════════════════════════════════╝

def reset_bn_stats(model):
    """Reseta running stats de todos os BatchNorms."""
    for module in model.modules():
        if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d)):
            module.running_mean.zero_()
            module.running_var.fill_(1.0)
            module.num_batches_tracked.zero_()


def sample_stratified_batch(npz_files, batch_size, num_points, flange_ratio, rng):
    """Cria um batch estratificado como no treino."""
    batch = []
    for _ in range(batch_size):
        file_idx = rng.integers(0, len(npz_files))
        data = np.load(npz_files[file_idx])
        features = data["features"]
        flange_idx = data["flange_idx"]
        equip_idx = data["equip_idx"]

        n_f = len(flange_idx)
        n_e = len(equip_idx)

        desired_f = max(1, int(num_points * flange_ratio))
        desired_e = num_points - desired_f

        if n_f > 0:
            sf = rng.choice(flange_idx, desired_f, replace=(n_f < desired_f))
        else:
            sf = np.array([], dtype=np.int64)
            desired_e = num_points

        se = rng.choice(equip_idx, desired_e, replace=(n_e < desired_e))
        indices = np.concatenate([sf, se])
        rng.shuffle(indices)

        batch.append(features[indices])

    return torch.from_numpy(np.stack(batch).astype(np.float32))


def recalibrate_batchnorm(model, npz_files, n_batches=500, batch_size=2,
                           num_points=8192, flange_ratio=0.3, device="cuda"):
    """
    Recalibra as running stats do BatchNorm.

    Estratégia:
      1. Reseta todas as running stats
      2. Coloca o modelo em train() (BN usa batch stats e atualiza running stats)
      3. Faz N forward passes SEM backprop (só atualiza BN stats)
      4. Volta para eval()

    Resultado: running_mean e running_var convergem para valores estáveis
    baseados na distribuição real dos dados.
    """
    print("\n" + "=" * 60)
    print("PASSO 1: RECALIBRANDO BATCHNORM")
    print("=" * 60)

    # Resetar stats
    reset_bn_stats(model)
    print(f"  ✓ Running stats resetadas")

    # Train mode (BN calcula e acumula stats)
    model.train()
    print(f"  Rodando {n_batches} batches (batch_size={batch_size})...")

    rng = np.random.default_rng(SEED)
    t0 = time.time()

    with torch.no_grad():
        for i in range(n_batches):
            batch = sample_stratified_batch(
                npz_files, batch_size, num_points, flange_ratio, rng
            ).to(device)

            # Forward pass — atualiza running stats do BN
            _ = model(batch)

            if (i + 1) % 100 == 0:
                print(f"    Batch {i+1}/{n_batches}")

    elapsed = time.time() - t0
    print(f"  Recalibração: {elapsed:.1f}s")

    # Voltar para eval
    model.eval()

    # Verificar novas stats
    print(f"\n  Novas BatchNorm stats (amostra):")
    for name, module in model.named_modules():
        if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d)):
            rm = module.running_mean
            rv = module.running_var
            if "attention" in name or "seg_head" in name or "fp4" in name:
                print(f"    {name:40s} mean=[{rm.min():.3f}, {rm.max():.3f}]  "
                      f"var=[{rv.min():.3f}, {rv.max():.3f}]  "
                      f"batches={module.num_batches_tracked.item()}")

    # Teste rápido
    print(f"\n  Teste rápido pós-calibração:")
    data = np.load(npz_files[0])
    features = data["features"]
    labels = data["labels"]
    flange_idx = data["flange_idx"]
    equip_idx = data["equip_idx"]

    # Amostra estratificada
    desired_f = max(1, int(num_points * flange_ratio))
    desired_e = num_points - desired_f
    sf = rng.choice(flange_idx, desired_f, replace=(len(flange_idx) < desired_f))
    se = rng.choice(equip_idx, desired_e, replace=(len(equip_idx) < desired_e))
    idx = np.concatenate([sf, se])
    rng.shuffle(idx)

    feat_t = torch.from_numpy(features[idx].astype(np.float32)).unsqueeze(0).to(device)
    lab_s = labels[idx]

    with torch.no_grad():
        logits = model(feat_t).squeeze(0).cpu().numpy()

    pred = logits.argmax(axis=1)
    n_pred_f = int((pred == 1).sum())
    n_gt_f = int((lab_s == 1).sum())
    tp = int(((pred == 1) & (lab_s == 1)).sum())
    fp = int(((pred == 1) & (lab_s == 0)).sum())
    fn = int(((pred == 0) & (lab_s == 1)).sum())
    fl_iou = 100 * tp / max(tp + fp + fn, 1)

    print(f"    Logits equip:  mean={logits[:,0].mean():.3f}")
    print(f"    Logits flange: mean={logits[:,1].mean():.3f}")
    print(f"    Predições: equip={int((pred==0).sum()):,}, flange={n_pred_f:,}")
    print(f"    Ground truth: equip={int((lab_s==0).sum()):,}, flange={n_gt_f:,}")
    print(f"    Flange IoU: {fl_iou:.1f}%")

    if fl_iou > 20:
        print(f"    ✓ RECALIBRAÇÃO FUNCIONOU!")
    else:
        print(f"    ⚠ Flange IoU ainda baixo — modelo pode precisar de retreino")

    # Salvar modelo corrigido
    torch.save(model.state_dict(), FIXED_MODEL_PATH)
    print(f"\n  Modelo corrigido salvo: {FIXED_MODEL_PATH}")

    return model


# ╔══════════════════════════════════════════════════════════════╗
# ║           PASSO 2: INFERÊNCIA COM MODELO CORRIGIDO           ║
# ╚══════════════════════════════════════════════════════════════╝

def inference_full_cloud(model, features, num_points=8192, num_votes=10,
                         overlap_factor=3.0, device="cuda"):
    """Inferência com votação (modelo já em eval com BN corrigido)."""
    model.eval()
    N = len(features)
    num_classes = 2

    vote_logits = np.zeros((N, num_classes), dtype=np.float64)
    vote_counts = np.zeros(N, dtype=np.int32)

    total_samples = max(num_votes, int(np.ceil(N / num_points * overlap_factor)))
    rng = np.random.default_rng(42)

    with torch.no_grad():
        for vote_idx in range(total_samples):
            indices = rng.choice(N, size=min(num_points, N), replace=(N < num_points))
            feat_tensor = torch.from_numpy(
                features[indices].astype(np.float32)
            ).unsqueeze(0).to(device)

            logits = model(feat_tensor).squeeze(0).cpu().numpy()
            vote_logits[indices] += logits
            vote_counts[indices] += 1

    # Cobrir pontos sem votos
    uncovered = np.where(vote_counts == 0)[0]
    if len(uncovered) > 0:
        n_extra = int(np.ceil(len(uncovered) / num_points))
        for i in range(n_extra):
            start = i * num_points
            end = min(start + num_points, len(uncovered))
            batch_idx = uncovered[start:end]
            if len(batch_idx) < num_points:
                extra = rng.choice(N, size=num_points - len(batch_idx), replace=False)
                full_idx = np.concatenate([batch_idx, extra])
            else:
                full_idx = batch_idx
            feat_tensor = torch.from_numpy(
                features[full_idx].astype(np.float32)
            ).unsqueeze(0).to(device)
            with torch.no_grad():
                logits = model(feat_tensor).squeeze(0).cpu().numpy()
            vote_logits[full_idx] += logits
            vote_counts[full_idx] += 1

    predictions = np.argmax(vote_logits, axis=1).astype(np.int64)

    avg_logits = vote_logits / np.maximum(vote_counts[:, None], 1).astype(np.float64)
    exp_logits = np.exp(avg_logits - avg_logits.max(axis=1, keepdims=True))
    probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)
    confidence = probs.max(axis=1)

    return predictions, confidence, vote_counts


# ╔══════════════════════════════════════════════════════════════╗
# ║           PASSO 3: MÉTRICAS E VISUALIZAÇÕES                  ║
# ╚══════════════════════════════════════════════════════════════╝

def compute_metrics(gt, pred):
    cm = confusion_matrix(gt, pred, labels=[0, 1])
    ious = np.diag(cm) / (cm.sum(1) + cm.sum(0) - np.diag(cm) + 1e-8)
    precision = np.diag(cm) / (cm.sum(0) + 1e-8)
    recall = np.diag(cm) / (cm.sum(1) + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)
    return {
        "cm": cm,
        "oa": 100.0 * cm.trace() / max(cm.sum(), 1),
        "miou": 100.0 * np.nanmean(ious),
        "class_iou": dict(zip(CLASS_NAMES, 100 * ious)),
        "class_precision": dict(zip(CLASS_NAMES, 100 * precision)),
        "class_recall": dict(zip(CLASS_NAMES, 100 * recall)),
        "class_f1": dict(zip(CLASS_NAMES, 100 * f1)),
    }


def print_metrics(results, name=""):
    print(f"\n  ── Resultados {name} ──")
    print(f"  OA:   {results['oa']:.2f}%")
    print(f"  mIoU: {results['miou']:.2f}%")
    for cls in CLASS_NAMES:
        print(f"  {cls:12s}  IoU={results['class_iou'][cls]:5.1f}%  "
              f"P={results['class_precision'][cls]:5.1f}%  "
              f"R={results['class_recall'][cls]:5.1f}%  "
              f"F1={results['class_f1'][cls]:5.1f}%")


def _subsample(xyz, colors, max_pts):
    if len(xyz) <= max_pts:
        return xyz, colors
    idx = np.random.default_rng(42).choice(len(xyz), max_pts, replace=False)
    return xyz[idx], colors[idx]


def labels_to_colors(labels):
    colors = np.zeros((len(labels), 3))
    colors[labels == 0] = COLOR_EQUIPMENT
    colors[labels == 1] = COLOR_FLANGE
    return colors


def errors_to_colors(gt, pred):
    colors = np.tile(COLOR_CORRECT, (len(gt), 1))
    colors[(gt == 1) & (pred == 0)] = COLOR_ERROR_FN
    colors[(gt == 0) & (pred == 1)] = COLOR_ERROR_FP
    return colors


def _setup_ax(ax, xyz, title="", elev=25, azim=45):
    ax.set_title(title, fontsize=14, fontweight='bold', pad=10)
    ax.set_xlabel("X", fontsize=9, labelpad=-2)
    ax.set_ylabel("Y", fontsize=9, labelpad=-2)
    ax.set_zlabel("Z", fontsize=9, labelpad=-2)
    ax.tick_params(labelsize=7)
    ax.view_init(elev=elev, azim=azim)
    ranges = xyz.max(axis=0) - xyz.min(axis=0)
    max_range = max(ranges) / 2.0
    mid = (xyz.max(axis=0) + xyz.min(axis=0)) / 2.0
    ax.set_xlim(mid[0] - max_range, mid[0] + max_range)
    ax.set_ylim(mid[1] - max_range, mid[1] + max_range)
    ax.set_zlim(mid[2] - max_range, mid[2] + max_range)
    ax.xaxis.pane.set_alpha(0.05)
    ax.yaxis.pane.set_alpha(0.05)
    ax.zaxis.pane.set_alpha(0.05)


def render_comparison(xyz, gt, pred, save_path, elev=25, azim=45):
    fig = plt.figure(figsize=(22, 7))

    ax1 = fig.add_subplot(131, projection='3d')
    xyz_r, c = _subsample(xyz, labels_to_colors(gt), MAX_POINTS_RENDER)
    ax1.scatter(xyz_r[:,0], xyz_r[:,1], xyz_r[:,2], c=c, s=POINT_SIZE, alpha=0.8, edgecolors='none')
    _setup_ax(ax1, xyz_r, "Ground Truth", elev, azim)

    ax2 = fig.add_subplot(132, projection='3d')
    xyz_r, c = _subsample(xyz, labels_to_colors(pred), MAX_POINTS_RENDER)
    ax2.scatter(xyz_r[:,0], xyz_r[:,1], xyz_r[:,2], c=c, s=POINT_SIZE, alpha=0.8, edgecolors='none')
    _setup_ax(ax2, xyz_r, "Prediction", elev, azim)

    ax3 = fig.add_subplot(133, projection='3d')
    xyz_r, c = _subsample(xyz, errors_to_colors(gt, pred), MAX_POINTS_RENDER)
    ax3.scatter(xyz_r[:,0], xyz_r[:,1], xyz_r[:,2], c=c, s=POINT_SIZE, alpha=0.8, edgecolors='none')
    _setup_ax(ax3, xyz_r, "Error Map", elev, azim)

    legend1 = [Patch(facecolor=COLOR_EQUIPMENT, label='Equipment'),
               Patch(facecolor=COLOR_FLANGE, label='Flange')]
    fig.legend(handles=legend1, loc='upper center', ncol=2, fontsize=11,
               framealpha=0.9, bbox_to_anchor=(0.37, 0.98))

    legend2 = [Patch(facecolor=COLOR_CORRECT, label='Correct'),
               Patch(facecolor=COLOR_ERROR_FN, label='FN (missed flange)'),
               Patch(facecolor=COLOR_ERROR_FP, label='FP (false alarm)')]
    fig.legend(handles=legend2, loc='upper center', ncol=3, fontsize=10,
               framealpha=0.9, bbox_to_anchor=(0.82, 0.98))

    plt.tight_layout(rect=[0, 0, 1, 0.93])
    plt.savefig(save_path, dpi=DPI, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  Salvo: {save_path}")


def render_multi_angle(xyz, pred, save_path, name=""):
    angles = [(25, 45, "Vista 1"), (25, 135, "Vista 2"), (25, 225, "Vista 3"),
              (90, 0, "Top"), (0, 0, "Front"), (0, 90, "Side")]
    fig = plt.figure(figsize=(20, 13))
    fig.suptitle(f"Segmentation — {name}", fontsize=16, fontweight='bold', y=0.98)
    xyz_r, c = _subsample(xyz, labels_to_colors(pred), MAX_POINTS_RENDER)
    for i, (elev, azim, title) in enumerate(angles):
        ax = fig.add_subplot(2, 3, i+1, projection='3d')
        ax.scatter(xyz_r[:,0], xyz_r[:,1], xyz_r[:,2], c=c, s=POINT_SIZE*0.8, alpha=0.8, edgecolors='none')
        _setup_ax(ax, xyz_r, title, elev, azim)
    legend = [Patch(facecolor=COLOR_EQUIPMENT, label='Equipment'),
              Patch(facecolor=COLOR_FLANGE, label='Flange')]
    fig.legend(handles=legend, loc='lower center', ncol=2, fontsize=12, framealpha=0.9)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(save_path, dpi=DPI, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  Salvo: {save_path}")


def render_cm(results, save_path, name=""):
    fig, ax = plt.subplots(figsize=(6, 5))
    cm = results['cm']
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
    im = ax.imshow(cm_norm, cmap='Blues', vmin=0, vmax=1)
    for i in range(2):
        for j in range(2):
            color = 'white' if cm_norm[i,j] > 0.5 else 'black'
            ax.text(j, i, f'{cm[i,j]:,}\n({100*cm_norm[i,j]:.1f}%)',
                    ha='center', va='center', color=color, fontsize=13, fontweight='bold')
    ax.set_xticks([0,1]); ax.set_yticks([0,1])
    ax.set_xticklabels(CLASS_NAMES, fontsize=12)
    ax.set_yticklabels(CLASS_NAMES, fontsize=12)
    ax.set_xlabel('Predicted', fontsize=13); ax.set_ylabel('True', fontsize=13)
    ax.set_title(f'Confusion Matrix — {name}', fontsize=14, fontweight='bold')
    plt.colorbar(im, ax=ax, shrink=0.8)
    plt.tight_layout()
    plt.savefig(save_path, dpi=DPI, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  Salvo: {save_path}")


def save_ply(xyz, colors_rgb, save_path):
    N = len(xyz)
    c = (np.clip(colors_rgb, 0, 1) * 255).astype(np.uint8)
    with open(save_path, 'w') as f:
        f.write(f"ply\nformat ascii 1.0\nelement vertex {N}\n"
                f"property float x\nproperty float y\nproperty float z\n"
                f"property uchar red\nproperty uchar green\nproperty uchar blue\n"
                f"end_header\n")
        for i in range(N):
            f.write(f"{xyz[i,0]:.6f} {xyz[i,1]:.6f} {xyz[i,2]:.6f} "
                    f"{c[i,0]} {c[i,1]} {c[i,2]}\n")
    print(f"  PLY: {save_path} ({os.path.getsize(save_path)/1024/1024:.1f} MB)")


# ╔══════════════════════════════════════════════════════════════╗
# ║                          MAIN                               ║
# ╚══════════════════════════════════════════════════════════════╝

def main():
    t_start = time.time()

    if not os.path.exists(MODEL_PATH):
        print(f"ERRO: {MODEL_PATH} não encontrado"); sys.exit(1)
    if not os.path.exists(DATA_DIR):
        print(f"ERRO: {DATA_DIR} não encontrado"); sys.exit(1)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Carregar modelo
    print("\nCarregando modelo...")
    model = PointNet2SegModel(num_classes=2, input_channels=8).to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE, weights_only=False))
    print("Modelo carregado.")

    # Listar arquivos
    npz_files = sorted(glob(os.path.join(DATA_DIR, "*.npz")))
    print(f"Arquivos: {len(npz_files)}")

    # ── PASSO 1: RECALIBRAR BATCHNORM ──
    model = recalibrate_batchnorm(
        model, npz_files,
        n_batches=CALIBRATION_BATCHES,
        batch_size=CALIBRATION_BATCH_SIZE,
        num_points=NUM_POINTS,
        flange_ratio=FLANGE_RATIO,
        device=DEVICE
    )

    # ── PASSO 2: INFERÊNCIA EM TODOS OS ARQUIVOS ──
    print(f"\n{'='*60}")
    print("PASSO 2: INFERÊNCIA")
    print(f"{'='*60}")

    all_gt = []
    all_pred = []
    per_file = []

    for fi, npz_path in enumerate(npz_files):
        name = os.path.splitext(os.path.basename(npz_path))[0]
        print(f"\n{'─'*50}")
        print(f"[{fi+1}/{len(npz_files)}] {name}")

        out_dir = os.path.join(OUTPUT_DIR, name)
        os.makedirs(out_dir, exist_ok=True)

        data = np.load(npz_path)
        features = data["features"]
        labels = data["labels"]
        xyz = data["xyz_centered"]
        n_f = int((labels == 1).sum())
        n_e = int((labels == 0).sum())
        print(f"  Pontos: {len(labels):,} (flange={n_f:,}, equip={n_e:,})")

        t_inf = time.time()
        pred, conf, counts = inference_full_cloud(
            model, features, NUM_POINTS, NUM_VOTES, OVERLAP_FACTOR, DEVICE
        )
        print(f"  Inferência: {time.time()-t_inf:.1f}s")

        res = compute_metrics(labels, pred)
        print_metrics(res, name)
        all_gt.append(labels)
        all_pred.append(pred)
        per_file.append((name, res))

        # Visualizações
        print(f"  Gerando figuras...")
        render_comparison(xyz, labels, pred, os.path.join(out_dir, f"{name}_comparison.png"))
        render_multi_angle(xyz, pred, os.path.join(out_dir, f"{name}_multi_angle.png"), name)
        render_cm(res, os.path.join(out_dir, f"{name}_cm.png"), name)
        save_ply(xyz, labels_to_colors(pred), os.path.join(out_dir, f"{name}_pred.ply"))
        save_ply(xyz, errors_to_colors(labels, pred), os.path.join(out_dir, f"{name}_errors.ply"))

    # ── PASSO 3: RESULTADOS GLOBAIS ──
    print(f"\n{'='*60}")
    print("RESULTADOS GLOBAIS")
    print(f"{'='*60}")

    all_gt = np.concatenate(all_gt)
    all_pred = np.concatenate(all_pred)
    global_res = compute_metrics(all_gt, all_pred)
    print_metrics(global_res, "GLOBAL")

    render_cm(global_res, os.path.join(OUTPUT_DIR, "global_cm.png"), "All Equipment")

    # Relatório
    report_path = os.path.join(OUTPUT_DIR, "inference_report.txt")
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("INFERENCE REPORT — PointNet++ Flange Segmentation v5 (BN Fixed)\n")
        f.write(f"{'='*70}\n\n")
        f.write(f"Model: {MODEL_PATH}\n")
        f.write(f"Fixed model: {FIXED_MODEL_PATH}\n")
        f.write(f"Calibration batches: {CALIBRATION_BATCHES}\n\n")

        f.write(f"{'─'*70}\n")
        f.write(f"{'Equipment':<25s} {'OA':>6s} {'mIoU':>6s} {'Eq IoU':>7s} "
                f"{'Fl IoU':>7s} {'Fl P':>6s} {'Fl R':>6s} {'Fl F1':>6s}\n")
        f.write(f"{'─'*70}\n")

        for name, res in per_file:
            f.write(f"{name:<25s} "
                    f"{res['oa']:>5.1f}% "
                    f"{res['miou']:>5.1f}% "
                    f"{res['class_iou']['equipment']:>6.1f}% "
                    f"{res['class_iou']['flange']:>6.1f}% "
                    f"{res['class_precision']['flange']:>5.1f}% "
                    f"{res['class_recall']['flange']:>5.1f}% "
                    f"{res['class_f1']['flange']:>5.1f}%\n")

        f.write(f"{'─'*70}\n")
        f.write(f"{'GLOBAL':<25s} "
                f"{global_res['oa']:>5.1f}% "
                f"{global_res['miou']:>5.1f}% "
                f"{global_res['class_iou']['equipment']:>6.1f}% "
                f"{global_res['class_iou']['flange']:>6.1f}% "
                f"{global_res['class_precision']['flange']:>5.1f}% "
                f"{global_res['class_recall']['flange']:>5.1f}% "
                f"{global_res['class_f1']['flange']:>5.1f}%\n")

        f.write(f"\nConfusion Matrix:\n{global_res['cm']}\n")

    print(f"\nRelatório: {report_path}")
    print(f"Tempo total: {(time.time()-t_start)/60:.1f} min")
    print(f"Saída: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
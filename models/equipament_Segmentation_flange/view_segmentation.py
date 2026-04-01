"""
INFERÊNCIA + VISUALIZAÇÃO — PointNet++ Segmentação de Flanges (v5)
===================================================================
Script corrigido para inferência consistente com o treinamento.

PROBLEMA DAS VERSÕES ANTERIORES:
  - Re-normalizavam features por chunk (8192 pts) ao invés de usar
    as features já normalizadas do .npz
  - Resultado: distribuição diferente do treino → modelo colapsa
    para classe majoritária (equipment) → 0% flange IoU

CORREÇÃO:
  - Carrega features DIRETAMENTE do .npz (já normalizadas por nuvem inteira)
  - NÃO re-normaliza, NÃO re-calcula normais/curvatura
  - Votação com cobertura total: cada ponto recebe múltiplos votos
  - Visualizações de qualidade para artigo científico

USO:
  python inference_and_visualize_v5.py

  Ou editando as variáveis abaixo:
    DATA_DIR    → pasta com os .npz pré-processados
    MODEL_PATH  → caminho do best_segmentation_v5.pth
    OUTPUT_DIR  → pasta de saída para figuras e PLYs
"""

import os
import sys
import time
import copy
import numpy as np
from glob import glob

import torch
import torch.nn as nn
import torch.nn.functional as F

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from mpl_toolkits.mplot3d import Axes3D

from sklearn.metrics import confusion_matrix, classification_report


# ╔══════════════════════════════════════════════════════════════╗
# ║                      CONFIGURAÇÃO                           ║
# ╚══════════════════════════════════════════════════════════════╝

# ── EDITE ESTES CAMINHOS ──
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

DATA_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, "../../DATA_PREPROCESSED_v5"))
MODEL_PATH = os.path.join(SCRIPT_DIR, "best_segmentation_v5.pth")
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "article_figures")

# Inferência
NUM_POINTS = 8192           # Mesmo do treino
NUM_VOTES = 10              # Número de passes de votação por nuvem
OVERLAP_FACTOR = 3.0        # Multiplica votos para garantir cobertura
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Visualização
FIGSIZE_SINGLE = (8, 8)     # Tamanho de cada vista
FIGSIZE_COMPARISON = (20, 8) # Comparação lado a lado
DPI = 300                    # Resolução para artigo
POINT_SIZE = 0.3             # Tamanho dos pontos no scatter
MAX_POINTS_RENDER = 200_000  # Limite para renderização (performance)

# Cores para o artigo (colorblind-friendly)
COLOR_EQUIPMENT = np.array([0.2, 0.4, 0.8])    # Azul
COLOR_FLANGE = np.array([0.9, 0.3, 0.1])       # Vermelho-laranja
COLOR_ERROR_FN = np.array([1.0, 0.0, 0.0])     # Vermelho (flange não detectado)
COLOR_ERROR_FP = np.array([1.0, 0.8, 0.0])     # Amarelo (falso positivo)
COLOR_CORRECT = np.array([0.85, 0.85, 0.85])   # Cinza claro (acerto)

CLASS_NAMES = ["equipment", "flange"]


print(f"Device: {DEVICE}")
print(f"Modelo: {MODEL_PATH}")
print(f"Dados: {DATA_DIR}")
print(f"Saída: {OUTPUT_DIR}")


# ╔══════════════════════════════════════════════════════════════╗
# ║              MODELO (cópia exata do treino)                  ║
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
# ║                   INFERÊNCIA COM VOTAÇÃO                     ║
# ╚══════════════════════════════════════════════════════════════╝

def inference_full_cloud(model, features, num_points=8192, num_votes=10,
                         overlap_factor=3.0, device="cuda"):
    """
    Inferência com votação para cobertura total da nuvem de pontos.

    CRÍTICO: as features vêm DIRETAS do .npz — NÃO re-normalizar!
    O .npz já contém features normalizadas por nuvem inteira (mesmo
    processamento do treino).

    Args:
        model: PointNet2SegModel treinado
        features: (N, 11) array — features do .npz (já normalizadas)
        num_points: pontos por amostra (deve ser igual ao treino)
        num_votes: passes de votação
        overlap_factor: multiplicador de cobertura
        device: cuda ou cpu

    Returns:
        predictions: (N,) array de predições por ponto
        confidence: (N,) array de confiança (fração de votos na classe ganha)
        vote_counts: (N,) quantidade de votos que cada ponto recebeu
    """
    model.eval()
    N = len(features)
    num_classes = 2

    # Acumuladores de votos (logits somados, não argmax)
    vote_logits = np.zeros((N, num_classes), dtype=np.float64)
    vote_counts = np.zeros(N, dtype=np.int32)

    # Calcular número total de amostras para cobertura
    samples_for_coverage = max(num_votes, int(np.ceil(N / num_points * overlap_factor)))
    total_samples = max(samples_for_coverage, num_votes)

    print(f"    Votação: {total_samples} amostras para {N:,} pontos "
          f"(cobertura estimada: {min(100, 100 * total_samples * num_points / N):.0f}%)")

    rng = np.random.default_rng(42)

    with torch.no_grad():
        for vote_idx in range(total_samples):
            # Amostragem aleatória SEM estratificação (como no eval do treino)
            indices = rng.choice(N, size=min(num_points, N),
                                replace=(N < num_points))

            feat_sample = features[indices]
            feat_tensor = torch.from_numpy(feat_sample.astype(np.float32))
            feat_tensor = feat_tensor.unsqueeze(0).to(device)  # (1, num_points, 11)

            logits = model(feat_tensor)  # (1, num_points, 2)
            logits_np = logits.squeeze(0).cpu().numpy()  # (num_points, 2)

            # Acumular logits (mais estável que acumular argmax)
            vote_logits[indices] += logits_np
            vote_counts[indices] += 1

    # Pontos sem cobertura
    uncovered = vote_counts == 0
    n_uncovered = int(uncovered.sum())

    if n_uncovered > 0:
        print(f"    ⚠ {n_uncovered:,} pontos sem cobertura ({100*n_uncovered/N:.1f}%)")
        # Cobrir pontos restantes com passes extras
        uncovered_idx = np.where(uncovered)[0]
        n_extra = int(np.ceil(len(uncovered_idx) / num_points))
        print(f"    Fazendo {n_extra} passes extras para cobrir...")

        for i in range(n_extra):
            start = i * num_points
            end = min(start + num_points, len(uncovered_idx))
            batch_idx = uncovered_idx[start:end]

            # Se batch menor que num_points, completar com pontos aleatórios
            if len(batch_idx) < num_points:
                extra = rng.choice(N, size=num_points - len(batch_idx), replace=False)
                full_idx = np.concatenate([batch_idx, extra])
            else:
                full_idx = batch_idx

            feat_sample = features[full_idx]
            feat_tensor = torch.from_numpy(feat_sample.astype(np.float32))
            feat_tensor = feat_tensor.unsqueeze(0).to(device)

            with torch.no_grad():
                logits = model(feat_tensor)
                logits_np = logits.squeeze(0).cpu().numpy()

            vote_logits[full_idx] += logits_np
            vote_counts[full_idx] += 1

    # Predição final: argmax dos logits acumulados
    predictions = np.argmax(vote_logits, axis=1).astype(np.int64)

    # Confiança: softmax dos logits médios
    avg_logits = vote_logits / np.maximum(vote_counts[:, None], 1).astype(np.float64)
    exp_logits = np.exp(avg_logits - avg_logits.max(axis=1, keepdims=True))
    probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)
    confidence = probs.max(axis=1)

    # Verificação final
    still_uncovered = int((vote_counts == 0).sum())
    if still_uncovered > 0:
        print(f"    ⚠ AVISO: {still_uncovered} pontos ainda sem cobertura!")
    else:
        print(f"    ✓ Cobertura: 100% ({N:,} pontos)")

    return predictions, confidence, vote_counts


# ╔══════════════════════════════════════════════════════════════╗
# ║                      MÉTRICAS                               ║
# ╚══════════════════════════════════════════════════════════════╝

def compute_metrics(gt, pred, class_names):
    """Computa métricas detalhadas para relatório."""
    cm = confusion_matrix(gt, pred, labels=[0, 1])

    # IoU por classe
    ious = np.diag(cm) / (cm.sum(1) + cm.sum(0) - np.diag(cm) + 1e-8)
    miou = np.nanmean(ious) * 100
    oa = 100.0 * cm.trace() / max(cm.sum(), 1)

    # Precision, Recall, F1 por classe
    precision = np.diag(cm) / (cm.sum(0) + 1e-8)
    recall = np.diag(cm) / (cm.sum(1) + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)

    results = {
        "cm": cm,
        "oa": oa,
        "miou": miou,
        "class_iou": {name: 100 * iou for name, iou in zip(class_names, ious)},
        "class_precision": {name: 100 * p for name, p in zip(class_names, precision)},
        "class_recall": {name: 100 * r for name, r in zip(class_names, recall)},
        "class_f1": {name: 100 * f for name, f in zip(class_names, f1)},
    }
    return results


def print_metrics(results, name=""):
    """Imprime métricas formatadas."""
    print(f"\n  ── Resultados {name} ──")
    print(f"  Overall Accuracy: {results['oa']:.2f}%")
    print(f"  mIoU:             {results['miou']:.2f}%")
    for cls in CLASS_NAMES:
        iou = results['class_iou'][cls]
        p = results['class_precision'][cls]
        r = results['class_recall'][cls]
        f1 = results['class_f1'][cls]
        print(f"  {cls:12s}  IoU={iou:5.1f}%  P={p:5.1f}%  R={r:5.1f}%  F1={f1:5.1f}%")
    print(f"  Confusion Matrix:")
    cm = results['cm']
    print(f"    [[{cm[0,0]:>8,}  {cm[0,1]:>8,}]")
    print(f"     [{cm[1,0]:>8,}  {cm[1,1]:>8,}]]")


# ╔══════════════════════════════════════════════════════════════╗
# ║              VISUALIZAÇÕES PARA ARTIGO                       ║
# ╚══════════════════════════════════════════════════════════════╝

def _subsample_for_render(xyz, colors, max_points):
    """Subsample para renderização (matplotlib fica lento com muitos pontos)."""
    if len(xyz) <= max_points:
        return xyz, colors
    rng = np.random.default_rng(42)
    idx = rng.choice(len(xyz), max_points, replace=False)
    return xyz[idx], colors[idx]


def _setup_3d_axis(ax, xyz, title="", elev=25, azim=45):
    """Configura eixo 3D de forma consistente para artigo."""
    ax.set_title(title, fontsize=14, fontweight='bold', pad=10)
    ax.set_xlabel("X", fontsize=9, labelpad=-2)
    ax.set_ylabel("Y", fontsize=9, labelpad=-2)
    ax.set_zlabel("Z", fontsize=9, labelpad=-2)
    ax.tick_params(labelsize=7)
    ax.view_init(elev=elev, azim=azim)

    # Aspect ratio igual
    ranges = xyz.max(axis=0) - xyz.min(axis=0)
    max_range = max(ranges) / 2.0
    mid = (xyz.max(axis=0) + xyz.min(axis=0)) / 2.0
    ax.set_xlim(mid[0] - max_range, mid[0] + max_range)
    ax.set_ylim(mid[1] - max_range, mid[1] + max_range)
    ax.set_zlim(mid[2] - max_range, mid[2] + max_range)
    ax.xaxis.pane.set_alpha(0.05)
    ax.yaxis.pane.set_alpha(0.05)
    ax.zaxis.pane.set_alpha(0.05)


def labels_to_colors(labels, color_equip=COLOR_EQUIPMENT, color_flange=COLOR_FLANGE):
    """Converte labels em cores RGB."""
    colors = np.zeros((len(labels), 3))
    colors[labels == 0] = color_equip
    colors[labels == 1] = color_flange
    return colors


def errors_to_colors(gt, pred):
    """Mapa de erros: cinza=acerto, vermelho=FN, amarelo=FP."""
    colors = np.tile(COLOR_CORRECT, (len(gt), 1))
    fn_mask = (gt == 1) & (pred == 0)  # Flange não detectado
    fp_mask = (gt == 0) & (pred == 1)  # Falso positivo
    colors[fn_mask] = COLOR_ERROR_FN
    colors[fp_mask] = COLOR_ERROR_FP
    return colors


def render_comparison(xyz, gt_labels, pred_labels, save_path,
                      elev=25, azim=45, max_points=MAX_POINTS_RENDER):
    """
    Figura de comparação: GT | Predição | Erros (lado a lado).
    Formato ideal para artigo.
    """
    fig = plt.figure(figsize=(22, 7))

    # Ground Truth
    ax1 = fig.add_subplot(131, projection='3d')
    xyz_r, gt_colors = _subsample_for_render(xyz, labels_to_colors(gt_labels), max_points)
    ax1.scatter(xyz_r[:, 0], xyz_r[:, 1], xyz_r[:, 2],
                c=gt_colors, s=POINT_SIZE, alpha=0.8, edgecolors='none')
    _setup_3d_axis(ax1, xyz_r, "Ground Truth", elev, azim)

    # Predição
    ax2 = fig.add_subplot(132, projection='3d')
    xyz_r, pred_colors = _subsample_for_render(xyz, labels_to_colors(pred_labels), max_points)
    ax2.scatter(xyz_r[:, 0], xyz_r[:, 1], xyz_r[:, 2],
                c=pred_colors, s=POINT_SIZE, alpha=0.8, edgecolors='none')
    _setup_3d_axis(ax2, xyz_r, "Prediction", elev, azim)

    # Mapa de erros
    ax3 = fig.add_subplot(133, projection='3d')
    xyz_r, err_colors = _subsample_for_render(xyz, errors_to_colors(gt_labels, pred_labels), max_points)
    ax3.scatter(xyz_r[:, 0], xyz_r[:, 1], xyz_r[:, 2],
                c=err_colors, s=POINT_SIZE, alpha=0.8, edgecolors='none')
    _setup_3d_axis(ax3, xyz_r, "Error Map", elev, azim)

    # Legenda
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=COLOR_EQUIPMENT, label='Equipment'),
        Patch(facecolor=COLOR_FLANGE, label='Flange'),
    ]
    fig.legend(handles=legend_elements, loc='upper center',
               ncol=2, fontsize=11, framealpha=0.9,
               bbox_to_anchor=(0.37, 0.98))

    error_legend = [
        Patch(facecolor=COLOR_CORRECT, label='Correct'),
        Patch(facecolor=COLOR_ERROR_FN, label='FN (missed flange)'),
        Patch(facecolor=COLOR_ERROR_FP, label='FP (false alarm)'),
    ]
    fig.legend(handles=error_legend, loc='upper center',
               ncol=3, fontsize=10, framealpha=0.9,
               bbox_to_anchor=(0.82, 0.98))

    plt.tight_layout(rect=[0, 0, 1, 0.93])
    plt.savefig(save_path, dpi=DPI, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()
    print(f"  Salvo: {save_path}")


def render_multi_angle(xyz, pred_labels, save_path, name="",
                       max_points=MAX_POINTS_RENDER):
    """
    Múltiplos ângulos de visualização — bom para artigo mostrar cobertura 3D.
    """
    angles = [
        (25, 45, "Vista 1 (45°)"),
        (25, 135, "Vista 2 (135°)"),
        (25, 225, "Vista 3 (225°)"),
        (90, 0, "Vista superior"),
        (0, 0, "Vista frontal"),
        (0, 90, "Vista lateral"),
    ]

    fig = plt.figure(figsize=(20, 13))
    fig.suptitle(f"Segmentation Result — {name}", fontsize=16, fontweight='bold', y=0.98)

    xyz_r, colors = _subsample_for_render(xyz, labels_to_colors(pred_labels), max_points)

    for i, (elev, azim, title) in enumerate(angles):
        ax = fig.add_subplot(2, 3, i + 1, projection='3d')
        ax.scatter(xyz_r[:, 0], xyz_r[:, 1], xyz_r[:, 2],
                   c=colors, s=POINT_SIZE * 0.8, alpha=0.8, edgecolors='none')
        _setup_3d_axis(ax, xyz_r, title, elev, azim)

    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=COLOR_EQUIPMENT, label='Equipment'),
        Patch(facecolor=COLOR_FLANGE, label='Flange'),
    ]
    fig.legend(handles=legend_elements, loc='lower center',
               ncol=2, fontsize=12, framealpha=0.9)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(save_path, dpi=DPI, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()
    print(f"  Salvo: {save_path}")


def render_confidence_map(xyz, confidence, pred_labels, save_path,
                          elev=25, azim=45, max_points=MAX_POINTS_RENDER):
    """
    Mapa de confiança — mostra onde o modelo tem mais/menos certeza.
    Útil para artigo como análise qualitativa.
    """
    fig = plt.figure(figsize=(18, 7))

    # Vista por confiança geral
    ax1 = fig.add_subplot(131, projection='3d')
    xyz_r, conf_r = _subsample_for_render(xyz, confidence, max_points)
    sc = ax1.scatter(xyz_r[:, 0], xyz_r[:, 1], xyz_r[:, 2],
                     c=conf_r, cmap='RdYlGn', s=POINT_SIZE, alpha=0.8,
                     vmin=0.5, vmax=1.0, edgecolors='none')
    _setup_3d_axis(ax1, xyz_r, "Confidence (all points)", elev, azim)
    plt.colorbar(sc, ax=ax1, shrink=0.6, label='Confidence')

    # Confiança só nos flanges preditos
    flange_mask = pred_labels == 1
    if flange_mask.sum() > 0:
        ax2 = fig.add_subplot(132, projection='3d')
        f_xyz = xyz[flange_mask]
        f_conf = confidence[flange_mask]
        f_xyz_r, f_conf_r = _subsample_for_render(f_xyz, f_conf, max_points // 3)
        sc2 = ax2.scatter(f_xyz_r[:, 0], f_xyz_r[:, 1], f_xyz_r[:, 2],
                          c=f_conf_r, cmap='RdYlGn', s=POINT_SIZE * 3,
                          alpha=0.9, vmin=0.5, vmax=1.0, edgecolors='none')
        _setup_3d_axis(ax2, f_xyz_r, "Confidence (predicted flanges)", elev, azim)
        plt.colorbar(sc2, ax=ax2, shrink=0.6, label='Confidence')

    # Histograma de confiança
    ax3 = fig.add_subplot(133)
    ax3.hist(confidence[pred_labels == 0], bins=50, alpha=0.7,
             label='Equipment', color=COLOR_EQUIPMENT, density=True)
    if flange_mask.sum() > 0:
        ax3.hist(confidence[pred_labels == 1], bins=50, alpha=0.7,
                 label='Flange', color=COLOR_FLANGE, density=True)
    ax3.set_xlabel('Confidence', fontsize=12)
    ax3.set_ylabel('Density', fontsize=12)
    ax3.set_title('Confidence Distribution', fontsize=14, fontweight='bold')
    ax3.legend(fontsize=11)
    ax3.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=DPI, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()
    print(f"  Salvo: {save_path}")


def save_colored_ply(xyz, colors_rgb, save_path):
    """
    Salva PLY colorido para visualização em CloudCompare/MeshLab.
    colors_rgb: (N, 3) com valores em [0, 1]
    """
    N = len(xyz)
    colors_uint8 = (np.clip(colors_rgb, 0, 1) * 255).astype(np.uint8)

    header = (
        "ply\n"
        "format ascii 1.0\n"
        f"element vertex {N}\n"
        "property float x\n"
        "property float y\n"
        "property float z\n"
        "property uchar red\n"
        "property uchar green\n"
        "property uchar blue\n"
        "end_header\n"
    )

    with open(save_path, 'w') as f:
        f.write(header)
        for i in range(N):
            f.write(f"{xyz[i,0]:.6f} {xyz[i,1]:.6f} {xyz[i,2]:.6f} "
                    f"{colors_uint8[i,0]} {colors_uint8[i,1]} {colors_uint8[i,2]}\n")

    size_mb = os.path.getsize(save_path) / (1024 * 1024)
    print(f"  PLY salvo: {save_path} ({size_mb:.1f} MB)")


def render_confusion_matrix(results, save_path, name=""):
    """Matriz de confusão com formatação para artigo."""
    fig, ax = plt.subplots(figsize=(6, 5))

    cm = results['cm']
    cm_normalized = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    im = ax.imshow(cm_normalized, cmap='Blues', vmin=0, vmax=1)

    for i in range(2):
        for j in range(2):
            count = cm[i, j]
            pct = 100 * cm_normalized[i, j]
            text_color = 'white' if cm_normalized[i, j] > 0.5 else 'black'
            ax.text(j, i, f'{count:,}\n({pct:.1f}%)',
                    ha='center', va='center', color=text_color,
                    fontsize=13, fontweight='bold')

    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(CLASS_NAMES, fontsize=12)
    ax.set_yticklabels(CLASS_NAMES, fontsize=12)
    ax.set_xlabel('Predicted Label', fontsize=13)
    ax.set_ylabel('True Label', fontsize=13)
    ax.set_title(f'Confusion Matrix — {name}', fontsize=14, fontweight='bold')

    plt.colorbar(im, ax=ax, shrink=0.8, label='Proportion')
    plt.tight_layout()
    plt.savefig(save_path, dpi=DPI, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()
    print(f"  Salvo: {save_path}")


# ╔══════════════════════════════════════════════════════════════╗
# ║                          MAIN                               ║
# ╚══════════════════════════════════════════════════════════════╝

def main():
    t_start = time.time()

    # ── Verificações ──
    if not os.path.exists(MODEL_PATH):
        print(f"ERRO: Modelo não encontrado: {MODEL_PATH}")
        sys.exit(1)
    if not os.path.exists(DATA_DIR):
        print(f"ERRO: Dados não encontrados: {DATA_DIR}")
        sys.exit(1)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # ── Carregar modelo ──
    print("\nCarregando modelo...")
    model = PointNet2SegModel(num_classes=2, input_channels=8).to(DEVICE)
    state_dict = torch.load(MODEL_PATH, map_location=DEVICE)
    model.load_state_dict(state_dict)
    model.eval()
    print("Modelo carregado.")

    # ── Listar arquivos ──
    npz_files = sorted(glob(os.path.join(DATA_DIR, "*.npz")))
    print(f"Arquivos: {len(npz_files)}")

    # ── Acumuladores globais ──
    all_gt = []
    all_pred = []
    per_file_results = []

    # ── Processar cada nuvem ──
    for file_idx, npz_path in enumerate(npz_files):
        name = os.path.splitext(os.path.basename(npz_path))[0]
        print(f"\n{'─'*50}")
        print(f"[{file_idx+1}/{len(npz_files)}] {name}")

        # Criar subpasta
        file_output_dir = os.path.join(OUTPUT_DIR, name)
        os.makedirs(file_output_dir, exist_ok=True)

        # ── Carregar .npz (features JÁ NORMALIZADAS) ──
        data = np.load(npz_path)
        features = data["features"]        # (N, 11) — PRONTO, não re-normalizar!
        labels = data["labels"]             # (N,) — ground truth
        xyz_centered = data["xyz_centered"] # (N, 3) — para visualização

        n_total = len(labels)
        n_flange = int((labels == 1).sum())
        n_equip = int((labels == 0).sum())
        print(f"  Pontos: {n_total:,} (flange={n_flange:,}, equip={n_equip:,})")

        # ── Inferência ──
        print(f"  Inferência (votação)...")
        t_inf = time.time()

        predictions, confidence, vote_counts = inference_full_cloud(
            model, features,
            num_points=NUM_POINTS,
            num_votes=NUM_VOTES,
            overlap_factor=OVERLAP_FACTOR,
            device=DEVICE
        )

        elapsed_inf = time.time() - t_inf
        print(f"  Inferência: {elapsed_inf:.1f}s")

        # ── Métricas ──
        results = compute_metrics(labels, predictions, CLASS_NAMES)
        print_metrics(results, name)

        all_gt.append(labels)
        all_pred.append(predictions)
        per_file_results.append((name, results))

        # ── Visualizações ──
        print(f"  Gerando visualizações...")

        # 1. Comparação GT vs Pred vs Erros
        render_comparison(
            xyz_centered, labels, predictions,
            os.path.join(file_output_dir, f"{name}_comparison.png"),
            elev=25, azim=45
        )

        # 2. Multi-ângulo
        render_multi_angle(
            xyz_centered, predictions,
            os.path.join(file_output_dir, f"{name}_multi_angle.png"),
            name=name
        )

        # 3. Mapa de confiança
        render_confidence_map(
            xyz_centered, confidence, predictions,
            os.path.join(file_output_dir, f"{name}_confidence.png")
        )

        # 4. Matriz de confusão individual
        render_confusion_matrix(
            results,
            os.path.join(file_output_dir, f"{name}_cm.png"),
            name=name
        )

        # 5. PLY coloridos (para CloudCompare/MeshLab)
        save_colored_ply(
            xyz_centered, labels_to_colors(labels),
            os.path.join(file_output_dir, f"{name}_gt.ply")
        )
        save_colored_ply(
            xyz_centered, labels_to_colors(predictions),
            os.path.join(file_output_dir, f"{name}_pred.ply")
        )
        save_colored_ply(
            xyz_centered, errors_to_colors(labels, predictions),
            os.path.join(file_output_dir, f"{name}_errors.ply")
        )

    # ── Métricas globais ──
    print(f"\n{'='*60}")
    print("RESULTADOS GLOBAIS")
    print(f"{'='*60}")

    all_gt = np.concatenate(all_gt)
    all_pred = np.concatenate(all_pred)

    global_results = compute_metrics(all_gt, all_pred, CLASS_NAMES)
    print_metrics(global_results, "GLOBAL")

    # Matriz de confusão global
    render_confusion_matrix(
        global_results,
        os.path.join(OUTPUT_DIR, "global_confusion_matrix.png"),
        name="All Equipment"
    )

    # ── Relatório por equipamento ──
    report_path = os.path.join(OUTPUT_DIR, "inference_report.txt")
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("INFERENCE REPORT — PointNet++ Flange Segmentation v5\n")
        f.write(f"{'='*60}\n\n")
        f.write(f"Model: {MODEL_PATH}\n")
        f.write(f"Data: {DATA_DIR}\n")
        f.write(f"NUM_POINTS: {NUM_POINTS}\n")
        f.write(f"NUM_VOTES: {NUM_VOTES}\n")
        f.write(f"OVERLAP_FACTOR: {OVERLAP_FACTOR}\n\n")

        f.write(f"{'─'*60}\n")
        f.write(f"{'Equipment':<25s} {'OA':>6s} {'mIoU':>6s} {'Eq IoU':>7s} "
                f"{'Fl IoU':>7s} {'Fl P':>6s} {'Fl R':>6s} {'Fl F1':>6s}\n")
        f.write(f"{'─'*60}\n")

        for name, res in per_file_results:
            f.write(
                f"{name:<25s} "
                f"{res['oa']:>5.1f}% "
                f"{res['miou']:>5.1f}% "
                f"{res['class_iou']['equipment']:>6.1f}% "
                f"{res['class_iou']['flange']:>6.1f}% "
                f"{res['class_precision']['flange']:>5.1f}% "
                f"{res['class_recall']['flange']:>5.1f}% "
                f"{res['class_f1']['flange']:>5.1f}%\n"
            )

        f.write(f"{'─'*60}\n")
        f.write(
            f"{'GLOBAL':<25s} "
            f"{global_results['oa']:>5.1f}% "
            f"{global_results['miou']:>5.1f}% "
            f"{global_results['class_iou']['equipment']:>6.1f}% "
            f"{global_results['class_iou']['flange']:>6.1f}% "
            f"{global_results['class_precision']['flange']:>5.1f}% "
            f"{global_results['class_recall']['flange']:>5.1f}% "
            f"{global_results['class_f1']['flange']:>5.1f}%\n"
        )

        f.write(f"\n{'='*60}\n")
        f.write(f"Global Confusion Matrix:\n{global_results['cm']}\n")

    print(f"\nRelatório salvo: {report_path}")

    t_total = time.time() - t_start
    print(f"\nTempo total: {t_total/60:.1f} minutos")
    print(f"Saída em: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
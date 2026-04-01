"""
INFERÊNCIA + VISUALIZAÇÃO — v6 (GroupNorm)
============================================
Script de inferência para o modelo v6 treinado com GroupNorm.
Gera visualizações de qualidade para artigo científico.

USO:
  python inference_v6.py
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
MODEL_PATH = os.path.join(SCRIPT_DIR, "best_segmentation_v6.pth")
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "article_figures_v6")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_POINTS = 8192
NUM_VOTES = 10
OVERLAP_FACTOR = 3.0
GN_NUM_GROUPS = 16

DPI = 300
POINT_SIZE = 0.3
MAX_RENDER = 200_000

COLOR_EQUIP = np.array([0.2, 0.4, 0.8])
COLOR_FLANGE = np.array([0.9, 0.3, 0.1])
COLOR_FN = np.array([1.0, 0.0, 0.0])
COLOR_FP = np.array([1.0, 0.8, 0.0])
COLOR_OK = np.array([0.85, 0.85, 0.85])

CLASS_NAMES = ["equipment", "flange"]


# ╔══════════════════════════════════════════════════════════════╗
# ║                      MODELO (GroupNorm)                      ║
# ╚══════════════════════════════════════════════════════════════╝

def make_norm_layer(num_channels, dim="1d"):
    num_groups = min(GN_NUM_GROUPS, num_channels)
    while num_channels % num_groups != 0:
        num_groups -= 1
    return nn.GroupNorm(num_groups=num_groups, num_channels=num_channels)

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
        self.norms = nn.ModuleList()
        for out_ch in mlp:
            self.convs.append(nn.Conv2d(last_ch, out_ch, 1))
            self.norms.append(make_norm_layer(out_ch, dim="2d"))
            last_ch = out_ch
    def forward(self, xyz, points):
        if self.npoint is not None:
            new_xyz, new_points, fps_idx = sample_and_group(self.npoint, self.nsample, xyz, points)
        else:
            new_xyz, new_points = sample_and_group_all(xyz, points)
            fps_idx = None
        new_points = new_points.permute(0, 3, 1, 2)
        for conv, norm in zip(self.convs, self.norms):
            new_points = F.relu(norm(conv(new_points)))
        new_points = torch.max(new_points, dim=-1)[0]
        new_points = new_points.permute(0, 2, 1)
        return new_xyz, new_points, fps_idx

class FeaturePropagation(nn.Module):
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
            dists, idx = dists[:,:,:3], idx[:,:,:3]
            dist_recip = 1.0 / (dists + 1e-8)
            weight = dist_recip / torch.sum(dist_recip, dim=2, keepdim=True)
            interpolated = torch.sum(index_points(points2, idx) * weight.unsqueeze(-1), dim=2)
        if points1 is not None:
            new_points = torch.cat([points1, interpolated], dim=-1)
        else:
            new_points = interpolated
        new_points = new_points.permute(0, 2, 1)
        for conv, norm in zip(self.convs, self.norms):
            new_points = F.relu(norm(conv(new_points)))
        return new_points.permute(0, 2, 1)

class PointNet2SegModel(nn.Module):
    def __init__(self, num_classes=2, input_channels=8):
        super().__init__()
        self.sa1 = SetAbstraction(2048, 32, 3+input_channels, [64,64,128])
        self.sa2 = SetAbstraction(512, 64, 3+128, [128,128,256])
        self.sa3 = SetAbstraction(128, 128, 3+256, [256,256,512])
        self.sa4 = SetAbstraction(None, None, 3+512, [512,512,1024])
        self.attention = nn.Sequential(
            nn.Conv1d(1024,512,1), make_norm_layer(512), nn.ReLU(),
            nn.Conv1d(512,1024,1), nn.Sigmoid())
        self.fp4 = FeaturePropagation(1024+512, [512,512])
        self.fp3 = FeaturePropagation(512+256, [256,256])
        self.fp2 = FeaturePropagation(256+128, [128,128])
        self.fp1 = FeaturePropagation(128+input_channels, [128,128])
        self.seg_head = nn.Sequential(
            nn.Conv1d(128,128,1), make_norm_layer(128), nn.ReLU(),
            nn.Dropout(0.5), nn.Conv1d(128,num_classes,1))
    def forward(self, features):
        xyz = features[:,:,:3].contiguous()
        extra = features[:,:,3:].contiguous()
        l0_xyz, l0_pts = xyz, extra
        l1_xyz, l1_pts, _ = self.sa1(l0_xyz, l0_pts)
        l2_xyz, l2_pts, _ = self.sa2(l1_xyz, l1_pts)
        l3_xyz, l3_pts, _ = self.sa3(l2_xyz, l2_pts)
        l4_xyz, l4_pts, _ = self.sa4(l3_xyz, l3_pts)
        a = l4_pts.permute(0,2,1)
        l4_pts = (l4_pts.permute(0,2,1) * self.attention(a)).permute(0,2,1)
        l3_pts = self.fp4(l3_xyz, l4_xyz, l3_pts, l4_pts)
        l2_pts = self.fp3(l2_xyz, l3_xyz, l2_pts, l3_pts)
        l1_pts = self.fp2(l1_xyz, l2_xyz, l1_pts, l2_pts)
        l0_pts = self.fp1(l0_xyz, l1_xyz, l0_pts, l1_pts)
        x = self.seg_head(l0_pts.permute(0,2,1))
        return x.permute(0,2,1)


# ╔══════════════════════════════════════════════════════════════╗
# ║                   INFERÊNCIA                                 ║
# ╚══════════════════════════════════════════════════════════════╝

def inference_full(model, features, num_points=8192, num_votes=10,
                   overlap_factor=3.0, device="cuda"):
    model.eval()
    N = len(features)
    vote_logits = np.zeros((N, 2), dtype=np.float64)
    vote_counts = np.zeros(N, dtype=np.int32)
    total = max(num_votes, int(np.ceil(N / num_points * overlap_factor)))
    rng = np.random.default_rng(42)

    with torch.no_grad():
        for _ in range(total):
            idx = rng.choice(N, min(num_points, N), replace=(N < num_points))
            ft = torch.from_numpy(features[idx].astype(np.float32)).unsqueeze(0).to(device)
            logits = model(ft).squeeze(0).cpu().numpy()
            vote_logits[idx] += logits
            vote_counts[idx] += 1

    # Cobrir pontos sem votos
    uncov = np.where(vote_counts == 0)[0]
    if len(uncov) > 0:
        for i in range(int(np.ceil(len(uncov) / num_points))):
            s, e = i * num_points, min((i+1) * num_points, len(uncov))
            bidx = uncov[s:e]
            if len(bidx) < num_points:
                bidx = np.concatenate([bidx, rng.choice(N, num_points - len(bidx), replace=False)])
            ft = torch.from_numpy(features[bidx].astype(np.float32)).unsqueeze(0).to(device)
            with torch.no_grad():
                logits = model(ft).squeeze(0).cpu().numpy()
            vote_logits[bidx] += logits
            vote_counts[bidx] += 1

    pred = np.argmax(vote_logits, axis=1)
    avg = vote_logits / np.maximum(vote_counts[:, None], 1)
    exp = np.exp(avg - avg.max(axis=1, keepdims=True))
    conf = (exp / exp.sum(axis=1, keepdims=True)).max(axis=1)
    return pred, conf, vote_counts


# ╔══════════════════════════════════════════════════════════════╗
# ║                   MÉTRICAS E VISUALIZAÇÃO                    ║
# ╚══════════════════════════════════════════════════════════════╝

def compute_metrics(gt, pred):
    cm = confusion_matrix(gt, pred, labels=[0,1])
    ious = np.diag(cm) / (cm.sum(1) + cm.sum(0) - np.diag(cm) + 1e-8)
    prec = np.diag(cm) / (cm.sum(0) + 1e-8)
    rec = np.diag(cm) / (cm.sum(1) + 1e-8)
    f1 = 2 * prec * rec / (prec + rec + 1e-8)
    return {"cm": cm, "oa": 100*cm.trace()/max(cm.sum(),1),
            "miou": 100*np.nanmean(ious),
            "iou": dict(zip(CLASS_NAMES, 100*ious)),
            "prec": dict(zip(CLASS_NAMES, 100*prec)),
            "rec": dict(zip(CLASS_NAMES, 100*rec)),
            "f1": dict(zip(CLASS_NAMES, 100*f1))}

def print_metrics(r, name=""):
    print(f"\n  ── {name} ──")
    print(f"  OA={r['oa']:.2f}%  mIoU={r['miou']:.2f}%")
    for c in CLASS_NAMES:
        print(f"  {c:12s} IoU={r['iou'][c]:5.1f}% P={r['prec'][c]:5.1f}% "
              f"R={r['rec'][c]:5.1f}% F1={r['f1'][c]:5.1f}%")

def _sub(xyz, c, mx):
    if len(xyz) <= mx: return xyz, c
    idx = np.random.default_rng(42).choice(len(xyz), mx, replace=False)
    return xyz[idx], c[idx] if c.ndim > 1 else c[idx]

def lbl2col(labels):
    c = np.zeros((len(labels), 3))
    c[labels==0] = COLOR_EQUIP; c[labels==1] = COLOR_FLANGE
    return c

def err2col(gt, pred):
    c = np.tile(COLOR_OK, (len(gt), 1))
    c[(gt==1)&(pred==0)] = COLOR_FN; c[(gt==0)&(pred==1)] = COLOR_FP
    return c

def _ax3d(ax, xyz, title="", elev=25, azim=45):
    ax.set_title(title, fontsize=14, fontweight='bold', pad=10)
    ax.set_xlabel("X", fontsize=9, labelpad=-2)
    ax.set_ylabel("Y", fontsize=9, labelpad=-2)
    ax.set_zlabel("Z", fontsize=9, labelpad=-2)
    ax.tick_params(labelsize=7)
    ax.view_init(elev=elev, azim=azim)
    r = xyz.max(0)-xyz.min(0); mr = max(r)/2; mid = (xyz.max(0)+xyz.min(0))/2
    ax.set_xlim(mid[0]-mr, mid[0]+mr); ax.set_ylim(mid[1]-mr, mid[1]+mr)
    ax.set_zlim(mid[2]-mr, mid[2]+mr)
    ax.xaxis.pane.set_alpha(0.05); ax.yaxis.pane.set_alpha(0.05); ax.zaxis.pane.set_alpha(0.05)

def render_comparison(xyz, gt, pred, path, elev=25, azim=45):
    fig = plt.figure(figsize=(22, 7))
    for i, (labels, title) in enumerate([(gt,"Ground Truth"),(pred,"Prediction")]):
        ax = fig.add_subplot(1,3,i+1, projection='3d')
        xr, cr = _sub(xyz, lbl2col(labels), MAX_RENDER)
        ax.scatter(xr[:,0],xr[:,1],xr[:,2], c=cr, s=POINT_SIZE, alpha=0.8, edgecolors='none')
        _ax3d(ax, xr, title, elev, azim)
    ax3 = fig.add_subplot(133, projection='3d')
    xr, cr = _sub(xyz, err2col(gt, pred), MAX_RENDER)
    ax3.scatter(xr[:,0],xr[:,1],xr[:,2], c=cr, s=POINT_SIZE, alpha=0.8, edgecolors='none')
    _ax3d(ax3, xr, "Error Map", elev, azim)
    fig.legend(handles=[Patch(facecolor=COLOR_EQUIP, label='Equipment'),
                        Patch(facecolor=COLOR_FLANGE, label='Flange')],
               loc='upper center', ncol=2, fontsize=11, bbox_to_anchor=(0.37,0.98))
    fig.legend(handles=[Patch(facecolor=COLOR_OK, label='Correct'),
                        Patch(facecolor=COLOR_FN, label='FN'), Patch(facecolor=COLOR_FP, label='FP')],
               loc='upper center', ncol=3, fontsize=10, bbox_to_anchor=(0.82,0.98))
    plt.tight_layout(rect=[0,0,1,0.93])
    plt.savefig(path, dpi=DPI, bbox_inches='tight', facecolor='white'); plt.close()
    print(f"  Fig: {path}")

def render_multi(xyz, pred, path, name=""):
    angles = [(25,45,"Vista 1"),(25,135,"Vista 2"),(25,225,"Vista 3"),
              (90,0,"Top"),(0,0,"Front"),(0,90,"Side")]
    fig = plt.figure(figsize=(20,13))
    fig.suptitle(f"Segmentation — {name}", fontsize=16, fontweight='bold', y=0.98)
    xr, cr = _sub(xyz, lbl2col(pred), MAX_RENDER)
    for i,(e,a,t) in enumerate(angles):
        ax = fig.add_subplot(2,3,i+1, projection='3d')
        ax.scatter(xr[:,0],xr[:,1],xr[:,2], c=cr, s=POINT_SIZE*0.8, alpha=0.8, edgecolors='none')
        _ax3d(ax, xr, t, e, a)
    fig.legend(handles=[Patch(facecolor=COLOR_EQUIP, label='Equipment'),
                        Patch(facecolor=COLOR_FLANGE, label='Flange')],
               loc='lower center', ncol=2, fontsize=12)
    plt.tight_layout(rect=[0,0.03,1,0.95])
    plt.savefig(path, dpi=DPI, bbox_inches='tight', facecolor='white'); plt.close()
    print(f"  Fig: {path}")

def save_ply(xyz, colors, path):
    c = (np.clip(colors, 0, 1)*255).astype(np.uint8)
    with open(path, 'w') as f:
        f.write(f"ply\nformat ascii 1.0\nelement vertex {len(xyz)}\n"
                f"property float x\nproperty float y\nproperty float z\n"
                f"property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n")
        for i in range(len(xyz)):
            f.write(f"{xyz[i,0]:.6f} {xyz[i,1]:.6f} {xyz[i,2]:.6f} {c[i,0]} {c[i,1]} {c[i,2]}\n")
    print(f"  PLY: {path} ({os.path.getsize(path)/1024/1024:.1f} MB)")


# ╔══════════════════════════════════════════════════════════════╗
# ║                          MAIN                               ║
# ╚══════════════════════════════════════════════════════════════╝

def main():
    t0 = time.time()
    if not os.path.exists(MODEL_PATH):
        print(f"ERRO: {MODEL_PATH} não encontrado"); sys.exit(1)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("Carregando modelo v6 (GroupNorm)...")
    model = PointNet2SegModel(num_classes=2, input_channels=8).to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE, weights_only=False))
    model.eval()
    print("OK")

    npz_files = sorted(glob(os.path.join(DATA_DIR, "*.npz")))
    print(f"Arquivos: {len(npz_files)}")

    all_gt, all_pred, per_file = [], [], []

    for fi, path in enumerate(npz_files):
        name = os.path.splitext(os.path.basename(path))[0]
        print(f"\n{'─'*50}\n[{fi+1}/{len(npz_files)}] {name}")

        od = os.path.join(OUTPUT_DIR, name)
        os.makedirs(od, exist_ok=True)

        data = np.load(path)
        features, labels, xyz = data["features"], data["labels"], data["xyz_centered"]
        nf, ne = int((labels==1).sum()), int((labels==0).sum())
        print(f"  Pontos: {len(labels):,} (f={nf:,}, e={ne:,})")

        t1 = time.time()
        pred, conf, counts = inference_full(model, features, NUM_POINTS, NUM_VOTES, OVERLAP_FACTOR, DEVICE)
        print(f"  Inferência: {time.time()-t1:.1f}s  Cobertura: {int((counts>0).sum()):,}/{len(counts):,}")

        r = compute_metrics(labels, pred)
        print_metrics(r, name)
        all_gt.append(labels); all_pred.append(pred); per_file.append((name, r))

        render_comparison(xyz, labels, pred, os.path.join(od, f"{name}_comparison.png"))
        render_multi(xyz, pred, os.path.join(od, f"{name}_multi_angle.png"), name)
        save_ply(xyz, lbl2col(pred), os.path.join(od, f"{name}_pred.ply"))
        save_ply(xyz, err2col(labels, pred), os.path.join(od, f"{name}_errors.ply"))

    # Global
    print(f"\n{'='*60}\nGLOBAL\n{'='*60}")
    gr = compute_metrics(np.concatenate(all_gt), np.concatenate(all_pred))
    print_metrics(gr, "GLOBAL")

    # Relatório
    rp = os.path.join(OUTPUT_DIR, "inference_report.txt")
    with open(rp, 'w', encoding='utf-8') as f:
        f.write("INFERENCE REPORT — v6 (GroupNorm)\n" + "="*60 + "\n\n")
        f.write(f"{'Equipment':<25s} {'OA':>6s} {'mIoU':>6s} {'Eq':>6s} {'Fl':>6s} "
                f"{'FlP':>6s} {'FlR':>6s} {'FlF1':>6s}\n" + "─"*60 + "\n")
        for name, r in per_file:
            f.write(f"{name:<25s} {r['oa']:>5.1f}% {r['miou']:>5.1f}% "
                    f"{r['iou']['equipment']:>5.1f}% {r['iou']['flange']:>5.1f}% "
                    f"{r['prec']['flange']:>5.1f}% {r['rec']['flange']:>5.1f}% "
                    f"{r['f1']['flange']:>5.1f}%\n")
        f.write("─"*60 + "\n")
        f.write(f"{'GLOBAL':<25s} {gr['oa']:>5.1f}% {gr['miou']:>5.1f}% "
                f"{gr['iou']['equipment']:>5.1f}% {gr['iou']['flange']:>5.1f}% "
                f"{gr['prec']['flange']:>5.1f}% {gr['rec']['flange']:>5.1f}% "
                f"{gr['f1']['flange']:>5.1f}%\n")
        f.write(f"\nCM:\n{gr['cm']}\n")
    print(f"\nRelatório: {rp}")
    print(f"Tempo: {(time.time()-t0)/60:.1f} min")

if __name__ == "__main__":
    main()
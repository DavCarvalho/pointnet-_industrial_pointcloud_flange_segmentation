"""
TRAIN v8 — PointNet++ para segmentação de flanges, alinhado ao artigo.

Mudanças em relação à v7:
  1. Augmentation é um parâmetro (--aug none|light|heavy). Padrão: none, a
     configuração adotada no artigo. Na v7 o treino principal era sempre heavy.
  2. 10 features por ponto (as 8 da v7 + |n_z| + z relativo), como no artigo.
  3. BUG corrigido na augmentation: na v7, point dropout e oclusão de octante
     trocavam a geometria de um ponto pela de outro mas mantinham o rótulo
     antigo (um ponto de tubo podia ficar rotulado como flange). Agora o
     rótulo acompanha o ponto copiado.
  4. Validação 100% determinística (âncora E subamostra com semente).
  5. Avalia o conjunto de TESTE no fim e grava as métricas (na v7 o teste era
     separado mas nunca avaliado por este script).
  6. Semente por linha de comando, para repetir com 3 sementes.
  7. Checkpoint guarda configuração e split; a inferência lê tudo dele.
  8. topk no lugar de argsort/sort (mesmo resultado, menos memória) e
     distâncias com clamp ≥ 0 (evita peso negativo na interpolação).

Uso:
  python train_v8.py                       # none, seed 42
  python train_v8.py --aug light --seed 7
"""
import os, sys, json, time, random, argparse
from glob import glob
import numpy as np, pandas as pd
import torch, torch.nn as nn, torch.nn.functional as F, torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm.auto import tqdm
from common_v8 import (NUM_FEATS, FEATURE_NAMES, VOXEL_SIZE_M, NORMAL_RADIUS_M, CURV_RADIUS_M,
                       XYZ, NRM, VERT, CURV, RGB, INT, point_features, confusion, metrics_from_cm)

# ───────── CONFIG ─────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR   = os.path.abspath(os.path.join(SCRIPT_DIR, "../DATA_PREPROCESSED_v8"))
DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")

NUM_POINTS    = 8192
BATCH_SIZE    = 4
NUM_EPOCHS    = 150
LR, WD        = 1e-3, 1e-5
EARLY_STOP    = 30
CHUNK_RADII_M = [0.5, 1.0, 2.0]
EVAL_RADIUS_M = 1.0          # validação, teste e inferência
FLANGE_ANCHOR_PROB  = 0.5
CLASS_WEIGHT_FLANGE = 8.0
FOCAL_GAMMA   = 2.0
GN_GROUPS     = 16
EPOCH_TRAIN, EPOCH_VALID, TEST_CHUNKS = 300, 80, 400


def set_seed(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ───────── POINTNET++ ─────────
def make_norm(ch):
    g = min(GN_GROUPS, ch)
    while ch % g != 0:
        g -= 1
    return nn.GroupNorm(g, ch)


def square_dist(a, b):
    d = -2 * torch.matmul(a, b.transpose(1, 2)) + (a ** 2).sum(-1, keepdim=True) + (b ** 2).sum(-1).unsqueeze(1)
    return d.clamp_min(0)


def index_points(pts, idx):
    B = pts.shape[0]
    bi = torch.arange(B, device=pts.device).view(B, *((1,) * (idx.dim() - 1)))
    return pts[bi, idx, :]


def fps(xyz, npoint):
    """Farthest Point Sampling: sempre o ponto mais longe dos já escolhidos."""
    B, N, _ = xyz.shape
    npoint = min(npoint, N)
    cent = torch.zeros(B, npoint, dtype=torch.long, device=xyz.device)
    dist = torch.full((B, N), 1e10, device=xyz.device)
    far = torch.randint(0, N, (B,), device=xyz.device)
    bi = torch.arange(B, device=xyz.device)
    for i in range(npoint):
        cent[:, i] = far
        c = xyz[bi, far, :].view(B, 1, 3)
        dist = torch.minimum(dist, ((xyz - c) ** 2).sum(-1))
        far = dist.argmax(-1)
    return cent


def sample_and_group(npoint, nsample, xyz, pts):
    """FPS → k vizinhos mais próximos → coordenadas relativas + features."""
    N = xyz.shape[1]
    new_xyz = index_points(xyz, fps(xyz, npoint))
    gidx = square_dist(new_xyz, xyz).topk(min(nsample, N), dim=-1, largest=False)[1]
    gxyz = index_points(xyz, gidx) - new_xyz.unsqueeze(2)
    return new_xyz, torch.cat([gxyz, index_points(pts, gidx)], dim=-1)


def group_all(xyz, pts):
    B, N, C = xyz.shape
    return torch.zeros(B, 1, C, device=xyz.device), torch.cat([xyz, pts], dim=-1).view(B, 1, N, -1)


class SA(nn.Module):
    """Set Abstraction: amostra, agrupa, MLP compartilhado, max-pool."""
    def __init__(self, npoint, nsample, in_ch, mlp):
        super().__init__()
        self.npoint, self.nsample = npoint, nsample
        self.convs, self.norms = nn.ModuleList(), nn.ModuleList()
        for o in mlp:
            self.convs.append(nn.Conv2d(in_ch, o, 1)); self.norms.append(make_norm(o)); in_ch = o

    def forward(self, xyz, pts):
        if self.npoint is None:
            new_xyz, g = group_all(xyz, pts)
        else:
            new_xyz, g = sample_and_group(self.npoint, self.nsample, xyz, pts)
        g = g.permute(0, 3, 1, 2)                          # (B, C, npoint, nsample)
        for c, n in zip(self.convs, self.norms):
            g = F.relu(n(c(g)))
        return new_xyz, g.max(-1)[0].permute(0, 2, 1)      # (B, npoint, C')


class FP(nn.Module):
    """Feature Propagation: interpola por 3 vizinhos, concatena a skip, MLP."""
    def __init__(self, in_ch, mlp):
        super().__init__()
        self.convs, self.norms = nn.ModuleList(), nn.ModuleList()
        for o in mlp:
            self.convs.append(nn.Conv1d(in_ch, o, 1)); self.norms.append(make_norm(o)); in_ch = o

    def forward(self, xyz1, xyz2, p1, p2):
        N, S = xyz1.shape[1], xyz2.shape[1]
        if S == 1:
            interp = p2.repeat(1, N, 1)
        else:
            d, idx = square_dist(xyz1, xyz2).topk(min(3, S), dim=-1, largest=False)
            w = 1.0 / (d + 1e-8)
            w = w / w.sum(-1, keepdim=True)
            interp = (index_points(p2, idx) * w.unsqueeze(-1)).sum(2)
        x = torch.cat([p1, interp], -1).permute(0, 2, 1)
        for c, n in zip(self.convs, self.norms):
            x = F.relu(n(c(x)))
        return x.permute(0, 2, 1)


class PointNet2Seg(nn.Module):
    def __init__(self, num_classes=2, in_feats=NUM_FEATS):
        super().__init__()
        self.sa1 = SA(2048, 32, 3 + in_feats, [64, 64, 128])
        self.sa2 = SA(512, 64, 3 + 128, [128, 128, 256])
        self.sa3 = SA(128, 128, 3 + 256, [256, 256, 512])
        self.sa4 = SA(None, None, 3 + 512, [512, 512, 1024])
        self.fp4 = FP(1024 + 512, [512, 512])
        self.fp3 = FP(512 + 256, [256, 256])
        self.fp2 = FP(256 + 128, [128, 128])
        self.fp1 = FP(128 + in_feats, [128, 128])
        self.head = nn.Sequential(nn.Conv1d(128, 128, 1), make_norm(128), nn.ReLU(), nn.Dropout(0.5),
                                  nn.Conv1d(128, num_classes, 1))

    def forward(self, x):                                  # x: (B, N, 3 + in_feats)
        xyz, extra = x[:, :, :3].contiguous(), x[:, :, 3:].contiguous()
        l1x, l1p = self.sa1(xyz, extra)
        l2x, l2p = self.sa2(l1x, l1p)
        l3x, l3p = self.sa3(l2x, l2p)
        l4x, l4p = self.sa4(l3x, l3p)
        l3p = self.fp4(l3x, l4x, l3p, l4p)
        l2p = self.fp3(l2x, l3x, l2p, l3p)
        l1p = self.fp2(l1x, l2x, l1p, l2p)
        l0p = self.fp1(xyz, l1x, extra, l1p)
        return self.head(l0p.permute(0, 2, 1)).permute(0, 2, 1)   # (B, N, 2) logits


# ───────── LOSS ─────────
def focal_loss(logits, target, weight, gamma=FOCAL_GAMMA):
    ce = F.cross_entropy(logits, target, reduction="none")
    pt = torch.exp(-ce)
    return (weight[target] * (1 - pt) ** gamma * ce).mean()


def lovasz_softmax(probs, target):
    losses = []
    for c in range(probs.shape[1]):
        fg = (target == c).float()
        if fg.sum() == 0:
            continue
        err, perm = (fg - probs[:, c]).abs().sort(0, descending=True)
        fg_s = fg[perm]
        gts = fg_s.sum()
        jac = 1.0 - (gts - fg_s.cumsum(0)) / (gts + (1 - fg_s).cumsum(0))
        jac = torch.cat([jac[:1], jac[1:] - jac[:-1]])
        losses.append(torch.dot(err, jac))
    return sum(losses) / len(losses) if losses else probs.sum() * 0.0


def combined_loss(logits, target, weight):
    logits, target = logits.reshape(-1, logits.shape[-1]), target.reshape(-1)
    return focal_loss(logits, target, weight) + lovasz_softmax(F.softmax(logits, dim=-1), target)


# ───────── AUGMENTATION (opera em feats (N, 13) e labels (N,)) ─────────
def _rot_z(f, rng):
    t = rng.uniform(0, 2 * np.pi); c, s = np.cos(t), np.sin(t)
    R = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], np.float32)
    f[:, XYZ] = f[:, XYZ] @ R.T; f[:, NRM] = f[:, NRM] @ R.T


def aug_light(f, lab, rng):
    """Rotação livre em Z, jitter σ=0.003, dropout de RGB (p=0.3)."""
    _rot_z(f, rng)
    f[:, XYZ] += rng.normal(0, 0.003, (len(f), 3)).astype(np.float32)
    if rng.random() < 0.3:
        f[:, RGB] = 0
    return f, lab


def aug_heavy(f, lab, rng):
    """Domain randomization da v7, com o rótulo acompanhando os pontos copiados."""
    _rot_z(f, rng)
    for axis in (0, 1):                                    # inclinação ±8.6°
        a = rng.uniform(-0.15, 0.15); ca, sa = np.cos(a), np.sin(a)
        R = (np.array([[1, 0, 0], [0, ca, -sa], [0, sa, ca]], np.float32) if axis == 0 else
             np.array([[ca, 0, sa], [0, 1, 0], [-sa, 0, ca]], np.float32))
        f[:, XYZ] = f[:, XYZ] @ R.T; f[:, NRM] = f[:, NRM] @ R.T
    f[:, XYZ] *= rng.uniform(0.85, 1.15, 3).astype(np.float32)
    f[:, XYZ] += rng.normal(0, rng.uniform(0.002, 0.015), (len(f), 3)).astype(np.float32)
    n_out = int(len(f) * rng.uniform(0.005, 0.02))
    if n_out:
        oi = rng.choice(len(f), n_out, replace=False)
        f[oi, XYZ] += rng.normal(0, 0.05, (n_out, 3)).astype(np.float32)

    def replace_rows(mask):                                # copia linha inteira + rótulo
        keep, drop = np.flatnonzero(~mask), np.flatnonzero(mask)
        if len(keep) and len(drop):
            src = rng.choice(keep, len(drop), replace=True)
            f[drop] = f[src]; lab[drop] = lab[src]

    replace_rows(rng.random(len(f)) < rng.uniform(0.0, 0.5))       # point dropout
    if rng.random() < 0.3:                                         # oclusão de octante
        sx, sy, sz = rng.choice([-1, 1], 3)
        occ = (f[:, 0] * sx > 0) & (f[:, 1] * sy > 0) & (f[:, 2] * sz > 0)
        if 0.05 < occ.mean() < 0.4:
            replace_rows(occ)
    f[:, VERT] = np.abs(f[:, NRM][:, 2])                           # |n_z| após as rotações
    if rng.random() < 0.2: f[:, RGB] = 0
    if rng.random() < 0.2: f[:, INT] = 0
    if rng.random() < 0.1: f[:, CURV] = 0
    return f, lab


AUGMENTS = {"none": None, "light": aug_light, "heavy": aug_heavy}


# ───────── DATASET ─────────
def load_npz(fp):
    d = np.load(fp)
    xyz = d["xyz"]
    return {"xyz": xyz, "labels": d["labels"],
            "feats": point_features(xyz, d["normals"], d["curvature"], d["rgb"], d["intensity"])}


def make_chunk(d, anchor, radius, num_points, rng):
    """Pontos no raio → subamostra de num_points → (p − âncora)/raio + features."""
    xyz = d["xyz"]
    d2 = ((xyz - anchor) ** 2).sum(1)
    idx = np.flatnonzero(d2 < radius * radius)
    if len(idx) < 100:                                     # raio quase vazio: mais próximos
        k = min(len(xyz), num_points)
        idx = np.argpartition(d2, k - 1)[:k]
    sel = rng.choice(idx, num_points, replace=len(idx) < num_points)
    feats = np.concatenate([(xyz[sel] - anchor) / radius, d["feats"][sel]], axis=1).astype(np.float32)
    return feats, sel


class ChunkDataset(Dataset):
    def __init__(self, files, epoch_size, train, seed, aug="none"):
        self.files, self.size, self.train, self.seed = files, epoch_size, train, seed
        self.aug = AUGMENTS[aug] if train else None
        self.rng = np.random.default_rng(seed)
        self.cache = {}

    def __len__(self):
        return self.size

    def _get(self, fp):
        if fp not in self.cache:
            self.cache[fp] = load_npz(fp)
        return self.cache[fp]

    def __getitem__(self, i):
        if self.train:
            rng = self.rng
            d = self._get(self.files[rng.integers(len(self.files))])
            radius = float(rng.choice(CHUNK_RADII_M))
            fl = np.flatnonzero(d["labels"] == 1)
            if len(fl) and rng.random() < FLANGE_ANCHOR_PROB:
                anchor = d["xyz"][rng.choice(fl)]
            else:
                anchor = d["xyz"][rng.integers(len(d["xyz"]))]
        else:                                              # determinístico: depende só de i
            rng = np.random.default_rng(self.seed + 1000 + i)
            d = self._get(self.files[i % len(self.files)])
            radius = EVAL_RADIUS_M
            anchor = d["xyz"][rng.integers(len(d["xyz"]))]
        feats, sel = make_chunk(d, anchor, radius, NUM_POINTS, rng)
        lab = d["labels"][sel].astype(np.int64)
        if self.aug is not None:
            feats, lab = self.aug(feats, lab.copy(), rng)
        return torch.from_numpy(feats), torch.from_numpy(lab)


# ───────── LAÇO ─────────
def run_epoch(model, loader, opt, cw, train):
    model.train(train)
    cm, tot, n = np.zeros((2, 2), np.int64), 0.0, 0
    with torch.set_grad_enabled(train):
        for f, l in tqdm(loader, leave=False, desc="train" if train else "eval"):
            f, l = f.to(DEVICE), l.to(DEVICE)
            logits = model(f)
            loss = combined_loss(logits, l, cw)
            if train:
                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
            tot += loss.item() * l.size(0); n += l.size(0)
            cm += confusion(l.cpu().numpy(), logits.argmax(-1).cpu().numpy())
    return tot / max(n, 1), metrics_from_cm(cm)


def split_files(files, seed):
    """Split por EQUIPAMENTO: 20% teste, 15% validação, resto treino."""
    idx = np.arange(len(files)); np.random.default_rng(seed).shuffle(idx)
    nt, nv = max(1, int(len(files) * 0.2)), max(1, int(len(files) * 0.15))
    return ([files[i] for i in idx[nt + nv:]], [files[i] for i in idx[nt:nt + nv]], [files[i] for i in idx[:nt]])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--aug", choices=list(AUGMENTS), default="none")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--epochs", type=int, default=NUM_EPOCHS)
    args = ap.parse_args()
    tag = f"{args.aug}_s{args.seed}"
    model_path = os.path.join(SCRIPT_DIR, f"best_v8_{tag}.pth")

    files = sorted(glob(os.path.join(DATA_DIR, "*.npz")))
    if not files:
        sys.exit(f"ERRO: nenhum .npz em {DATA_DIR}. Rode preprocess_v8.py primeiro.")
    train_f, valid_f, test_f = split_files(files, args.seed)
    print(f"Device {DEVICE} | aug={args.aug} seed={args.seed} | train={len(train_f)} valid={len(valid_f)} test={len(test_f)}")

    set_seed(args.seed)
    model = PointNet2Seg(2, NUM_FEATS).to(DEVICE)
    print(f"Parâmetros: {sum(p.numel() for p in model.parameters()):,} | features: {FEATURE_NAMES}")
    cw = torch.tensor([1.0, CLASS_WEIGHT_FLANGE], device=DEVICE)
    opt = optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)
    sch = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs, eta_min=1e-6)

    mk = lambda fs, n, tr: DataLoader(ChunkDataset(fs, n, tr, args.seed, args.aug), BATCH_SIZE, shuffle=False, num_workers=0)
    tr_ld, va_ld = mk(train_f, EPOCH_TRAIN, True), mk(valid_f, EPOCH_VALID, False)

    cfg = {"aug": args.aug, "seed": args.seed, "features": FEATURE_NAMES, "num_points": NUM_POINTS,
           "eval_radius_m": EVAL_RADIUS_M, "voxel_m": VOXEL_SIZE_M, "normal_radius_m": NORMAL_RADIUS_M,
           "curv_radius_m": CURV_RADIUS_M,
           "split": {k: [os.path.basename(f) for f in v] for k, v in
                     (("train", train_f), ("valid", valid_f), ("test", test_f))}}

    best, noimp, hist = -1.0, 0, []
    for e in range(1, args.epochs + 1):
        t0 = time.time()
        tl, tm = run_epoch(model, tr_ld, opt, cw, True)
        vl, vm = run_epoch(model, va_ld, opt, cw, False)
        sch.step()
        hist.append({"epoch": e, "train_loss": tl, "train_oa": tm["OA"], "train_miou": tm["mIoU"],
                     "val_loss": vl, "val_oa": vm["OA"], "val_miou": vm["mIoU"]})
        star = ""
        if vm["mIoU"] > best:
            best, noimp, star = vm["mIoU"], 0, " ★"
            torch.save({"model": model.state_dict(), "cfg": cfg, "epoch": e, "val_miou": best}, model_path)
        else:
            noimp += 1
        print(f"E{e:3d} | treino L={tl:.3f} mIoU={tm['mIoU']:.1f} | val L={vl:.3f} mIoU={vm['mIoU']:.1f} "
              f"| {time.time() - t0:.0f}s{star}")
        if noimp >= EARLY_STOP:
            print("Early stop."); break
    pd.DataFrame(hist).to_csv(os.path.join(SCRIPT_DIR, f"history_v8_{tag}.csv"), index=False)

    # Teste por chunks com o melhor modelo (mesmo protocolo da ablação do artigo).
    # A métrica de cena inteira sai de:  python inference_v8.py --eval test
    model.load_state_dict(torch.load(model_path, map_location=DEVICE)["model"])
    _, test_m = run_epoch(model, mk(test_f, TEST_CHUNKS, False), opt, cw, False)
    with open(os.path.join(SCRIPT_DIR, f"test_chunks_v8_{tag}.json"), "w") as fo:
        json.dump({"best_val_miou": best, **test_m}, fo, indent=2)
    print("TESTE (chunks, argmax): " + " | ".join(f"{k}={v:.2f}" for k, v in test_m.items()))
    print(f"Modelo: {model_path}")


if __name__ == "__main__":
    main()

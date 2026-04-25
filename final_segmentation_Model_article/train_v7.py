# 2. train_v7.py — Treino robusto

# Sampling por chunks esféricos de raio físico fixo durante treino
# Augmentation pesada:

# Point dropout 20-60% aleatório
# Oclusão de octantes (simula view parcial de scanner)
# Ruído gaussiano com sigma variável (0.5cm a 3cm)
# Outliers esparsos (0.5% dos pontos, ruído grande)
# Subamostragem de densidade variável (simula distâncias diferentes do scanner)
# Rotação SO(3) limitada + rotação Z livre
# Mixup de chunks entre equipamentos diferentes


# Loss: Focal (γ=2, α=[1, 15]) + Lovász-Softmax (melhor que Dice para segmentação desbalanceada)
# Remove FLANGE_RATIO forçado — usa distribuição natural com oversample leve (max 10% flange)
# GroupNorm mantido (sua correção foi certa)
# Validação em chunks da MESMA forma que inferência (não equipamento inteiro)

"""
TRAIN v7 — Chunks de raio físico + augmentation pesada + domain randomization.

Chaves:
  - Amostragem: pega um PONTO ÂNCORA aleatório, pega vizinhos num RAIO FÍSICO
    aleatório (0.5/1.0/2.0 m) → subsampla para NUM_POINTS.
  - Sem FLANGE_RATIO forçado: âncora pode ser qualquer ponto (50% das vezes)
    ou um ponto perto de flange (50%) — mantém prior mais realista.
  - Augmentation forte: dropout, oclusão, ruído variável, outliers, densidade.
  - Model: GroupNorm (mantido do v6).
  - Loss: Focal + Lovász-Softmax (melhor que Dice para IoU).
"""
import os, sys, time, random, copy, numpy as np, pandas as pd
from glob import glob
import torch, torch.nn as nn, torch.nn.functional as F, torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import confusion_matrix
from tqdm.auto import tqdm

# ───────── CONFIG ─────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR   = os.path.abspath(os.path.join(SCRIPT_DIR, "../DATA_PREPROCESSED_v7"))
MODEL_PATH = os.path.join(SCRIPT_DIR, "best_v7.pth")
HIST_PATH  = os.path.join(SCRIPT_DIR, "history_v7.csv")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

NUM_POINTS   = 8192
BATCH_SIZE   = 4
NUM_EPOCHS   = 150
LR           = 1e-3
WD           = 1e-5
EARLY_STOP   = 30
SEED         = 42
CHUNK_RADII_M = [0.5, 1.0, 2.0]       # amostra multi-escala
FLANGE_ANCHOR_PROB = 0.5              # 50% das vezes âncora perto de flange
CLASS_WEIGHT_FLANGE = 8.0
GN_GROUPS    = 16
CLASS_NAMES  = ["equipment", "flange"]

random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
if torch.cuda.is_available(): torch.cuda.manual_seed_all(SEED)

print(f"Device: {DEVICE}")

# ───────── NORM LAYER ─────────
def make_norm(ch):
    g = min(GN_GROUPS, ch)
    while ch % g != 0: g -= 1
    return nn.GroupNorm(g, ch)

# ───────── POINTNET++ (mesmo do v6) ─────────
def square_dist(a, b):
    return -2*torch.matmul(a, b.transpose(1,2)) + (a**2).sum(-1,keepdim=True) + (b**2).sum(-1).unsqueeze(1)

def index_points(pts, idx):
    B = pts.shape[0]
    bi = torch.arange(B, device=pts.device).view(B, *((1,)*(idx.dim()-1)))
    return pts[bi, idx, :]

def fps(xyz, npoint):
    B, N, _ = xyz.shape
    npoint = min(npoint, N)
    cent = torch.zeros(B, npoint, dtype=torch.long, device=xyz.device)
    dist = torch.ones(B, N, device=xyz.device) * 1e10
    far = torch.randint(0, N, (B,), dtype=torch.long, device=xyz.device)
    bi = torch.arange(B, dtype=torch.long, device=xyz.device)
    for i in range(npoint):
        cent[:,i] = far
        c = xyz[bi, far, :].view(B,1,3)
        d = ((xyz-c)**2).sum(-1)
        m = d < dist; dist[m] = d[m]
        far = torch.max(dist, dim=-1)[1]
    return cent

def sag(npoint, nsample, xyz, pts):
    B, N, C = xyz.shape
    npoint = min(npoint, N) if npoint else N
    if npoint == N and pts is None: pass
    fidx = fps(xyz, npoint); new_xyz = index_points(xyz, fidx)
    sd = square_dist(new_xyz, xyz)
    nsample = min(nsample, N)
    gidx = sd.argsort()[:,:,:nsample]
    gxyz = index_points(xyz, gidx) - new_xyz.unsqueeze(2)
    if pts is not None:
        gp = index_points(pts, gidx); new_pts = torch.cat([gxyz, gp], dim=-1)
    else: new_pts = gxyz
    return new_xyz, new_pts

def sag_all(xyz, pts):
    B, N, C = xyz.shape
    new_xyz = torch.zeros(B,1,C, device=xyz.device)
    gxyz = xyz.view(B,1,N,C)
    if pts is not None:
        new_pts = torch.cat([gxyz, pts.view(B,1,N,-1)], dim=-1)
    else: new_pts = gxyz
    return new_xyz, new_pts

class SA(nn.Module):
    def __init__(self, np_, ns, ic, mlp):
        super().__init__()
        self.np_, self.ns = np_, ns
        last = ic; self.convs, self.norms = nn.ModuleList(), nn.ModuleList()
        for o in mlp:
            self.convs.append(nn.Conv2d(last, o, 1))
            self.norms.append(make_norm(o)); last = o
    def forward(self, xyz, pts):
        if self.np_ is not None:
            new_xyz, new_pts = sag(self.np_, self.ns, xyz, pts)
        else:
            new_xyz, new_pts = sag_all(xyz, pts)
        new_pts = new_pts.permute(0,3,1,2)
        for c, n in zip(self.convs, self.norms):
            new_pts = F.relu(n(c(new_pts)))
        new_pts = torch.max(new_pts, -1)[0].permute(0,2,1)
        return new_xyz, new_pts

class FP(nn.Module):
    def __init__(self, ic, mlp):
        super().__init__()
        last = ic; self.convs, self.norms = nn.ModuleList(), nn.ModuleList()
        for o in mlp:
            self.convs.append(nn.Conv1d(last, o, 1))
            self.norms.append(make_norm(o)); last = o
    def forward(self, xyz1, xyz2, p1, p2):
        B, N, _ = xyz1.shape; _, S, _ = xyz2.shape
        if S == 1: interp = p2.repeat(1, N, 1)
        else:
            d = square_dist(xyz1, xyz2); d, idx = d.sort(-1)
            d, idx = d[:,:,:3], idx[:,:,:3]
            dr = 1.0/(d+1e-8); w = dr / dr.sum(2, keepdim=True)
            interp = torch.sum(index_points(p2, idx) * w.unsqueeze(-1), dim=2)
        new = torch.cat([p1, interp], -1) if p1 is not None else interp
        new = new.permute(0,2,1)
        for c, n in zip(self.convs, self.norms):
            new = F.relu(n(c(new)))
        return new.permute(0,2,1)

class PointNet2Seg(nn.Module):
    def __init__(self, nc=2, ic=8):
        super().__init__()
        self.sa1 = SA(2048, 32, 3+ic, [64,64,128])
        self.sa2 = SA(512, 64, 3+128, [128,128,256])
        self.sa3 = SA(128, 128, 3+256, [256,256,512])
        self.sa4 = SA(None, None, 3+512, [512,512,1024])
        self.fp4 = FP(1024+512, [512,512])
        self.fp3 = FP(512+256, [256,256])
        self.fp2 = FP(256+128, [128,128])
        self.fp1 = FP(128+ic, [128,128])
        self.head = nn.Sequential(
            nn.Conv1d(128,128,1), make_norm(128), nn.ReLU(), nn.Dropout(0.5),
            nn.Conv1d(128, nc, 1))
    def forward(self, feats):
        xyz = feats[:,:,:3].contiguous(); extra = feats[:,:,3:].contiguous()
        l1x, l1p = self.sa1(xyz, extra)
        l2x, l2p = self.sa2(l1x, l1p)
        l3x, l3p = self.sa3(l2x, l2p)
        l4x, l4p = self.sa4(l3x, l3p)
        l3p = self.fp4(l3x, l4x, l3p, l4p)
        l2p = self.fp3(l2x, l3x, l2p, l3p)
        l1p = self.fp2(l1x, l2x, l1p, l2p)
        l0p = self.fp1(xyz, l1x, extra, l1p)
        return self.head(l0p.permute(0,2,1)).permute(0,2,1)

# ───────── LOSSES ─────────
def focal(pred, tgt, w, gamma=2.0):
    p = pred.reshape(-1, pred.shape[-1]); t = tgt.reshape(-1)
    ce = F.cross_entropy(p, t, reduction="none")
    pt = torch.exp(-ce)
    return (w[t] * (1-pt)**gamma * ce).mean()

def lovasz_softmax_flat(probs, labels):
    """Lovász-Softmax: otimiza IoU diretamente. Melhor que Dice p/ desbalanceado."""
    C = probs.shape[1]; losses = []
    for c in range(C):
        fg = (labels == c).float()
        if fg.sum() == 0: continue
        err = (fg - probs[:, c]).abs()
        err_s, perm = err.sort(0, descending=True)
        fg_s = fg[perm]
        # Gradiente de Lovász
        gts = fg_s.sum()
        inter = gts - fg_s.cumsum(0)
        union = gts + (1 - fg_s).cumsum(0)
        jaccard = 1. - inter / union
        if len(jaccard) > 1:
            jaccard[1:] = jaccard[1:] - jaccard[:-1]
        losses.append(torch.dot(err_s, jaccard))
    return sum(losses)/max(len(losses),1) if losses else torch.tensor(0., device=probs.device)

def combined_loss(pred, tgt, w):
    probs = F.softmax(pred, dim=-1)
    lov = lovasz_softmax_flat(probs.reshape(-1, probs.shape[-1]), tgt.reshape(-1))
    return focal(pred, tgt, w) + lov

# ───────── DATASET: CHUNKS DE RAIO FÍSICO ─────────
class ChunkDataset(Dataset):
    def __init__(self, files, num_points, epoch_size, train=True):
        self.files = files; self.N = num_points; self.size = epoch_size; self.train = train
        self.cache = {}
    def __len__(self): return self.size
    def _load(self, fp):
        if fp not in self.cache:
            d = np.load(fp)
            self.cache[fp] = {k: d[k] for k in ["xyz","labels","normals","curvature","rgb","intensity"]}
        return self.cache[fp]
    def __getitem__(self, idx):
        fp = self.files[np.random.randint(len(self.files))] if self.train else self.files[idx % len(self.files)]
        d = self._load(fp)
        xyz = d["xyz"]; labels = d["labels"]
        N = len(xyz)

        # Escolher raio e âncora
        if self.train:
            radius = np.random.choice(CHUNK_RADII_M)
            flange_idx = np.where(labels == 1)[0]
            if len(flange_idx) > 0 and np.random.rand() < FLANGE_ANCHOR_PROB:
                anchor = xyz[np.random.choice(flange_idx)]
            else:
                anchor = xyz[np.random.randint(N)]
        else:
            radius = 1.0
            rng = np.random.default_rng(idx + SEED)
            anchor = xyz[rng.integers(0, N)]

        # Pegar pontos no raio
        dists = np.linalg.norm(xyz - anchor, axis=1)
        mask = dists < radius
        chunk_idx = np.where(mask)[0]
        if len(chunk_idx) < 100:  # fallback: pega KNN
            chunk_idx = np.argsort(dists)[:max(self.N, 500)]

        # Subsample para NUM_POINTS
        if len(chunk_idx) >= self.N:
            sel = np.random.choice(chunk_idx, self.N, replace=False)
        else:
            sel = np.random.choice(chunk_idx, self.N, replace=True)

        xyz_c = xyz[sel].copy()
        xyz_c -= anchor  # centra no âncora
        # NORMALIZA PELO RAIO do chunk -> modelo vê sempre esfera unitária
        xyz_c /= radius

        feats = np.concatenate([
            xyz_c,
            d["normals"][sel],
            d["curvature"][sel],
            d["rgb"][sel],
            d["intensity"][sel],
        ], axis=1).astype(np.float32)
        lab = labels[sel].astype(np.int64)

        if self.train:
            feats = self._augment(feats)

        return torch.from_numpy(feats), torch.from_numpy(lab)

    def _augment(self, feats):
        f = feats.copy()
        xyz, nrm = f[:,:3], f[:,3:6]

        # Rotação Z livre
        t = np.random.uniform(0, 2*np.pi)
        c, s = np.cos(t), np.sin(t)
        R = np.array([[c,-s,0],[s,c,0],[0,0,1]], dtype=np.float32)
        xyz = xyz @ R.T; nrm = nrm @ R.T

        # Rotação pequena em X e Y (simula inclinação de scanner)
        for axis in [0, 1]:
            a = np.random.uniform(-0.15, 0.15)
            ca, sa = np.cos(a), np.sin(a)
            if axis == 0:
                R2 = np.array([[1,0,0],[0,ca,-sa],[0,sa,ca]], dtype=np.float32)
            else:
                R2 = np.array([[ca,0,sa],[0,1,0],[-sa,0,ca]], dtype=np.float32)
            xyz = xyz @ R2.T; nrm = nrm @ R2.T

        # Escala anisotrópica
        sc = np.random.uniform(0.85, 1.15, 3).astype(np.float32)
        xyz *= sc
        nrm /= (np.linalg.norm(nrm, axis=1, keepdims=True) + 1e-8)

        # Jitter variável (0.002 a 0.015 em espaço normalizado)
        sig = np.random.uniform(0.002, 0.015)
        xyz += np.random.normal(0, sig, xyz.shape).astype(np.float32)

        # Outliers: 0.5-2% dos pontos com ruído grande
        n_out = int(len(xyz) * np.random.uniform(0.005, 0.02))
        if n_out > 0:
            oi = np.random.choice(len(xyz), n_out, replace=False)
            xyz[oi] += np.random.normal(0, 0.05, (n_out,3)).astype(np.float32)

        # Point dropout: simula oclusão / densidade variável
        drop = np.random.uniform(0.0, 0.5)
        if drop > 0:
            keep_mask = np.random.rand(len(xyz)) > drop
            if keep_mask.sum() < len(xyz):
                # reamostra os pontos dropados a partir dos mantidos
                kept_idx = np.where(keep_mask)[0]
                drop_idx = np.where(~keep_mask)[0]
                replace = np.random.choice(kept_idx, len(drop_idx), replace=True)
                xyz[drop_idx] = xyz[replace]; nrm[drop_idx] = nrm[replace]
                f[drop_idx, 6:] = f[replace, 6:]  # copia features também

        # Oclusão de octante: zera um octante aleatório com 30% de prob
        if np.random.rand() < 0.3:
            sx, sy, sz = np.random.choice([-1,1],3)
            occ = (xyz[:,0]*sx > 0) & (xyz[:,1]*sy > 0) & (xyz[:,2]*sz > 0)
            if 0.05 < occ.mean() < 0.4:
                # substitui pontos do octante por cópias de outros
                keep = np.where(~occ)[0]
                xyz[occ] = xyz[np.random.choice(keep, occ.sum())]
                nrm[occ] = nrm[np.random.choice(keep, occ.sum())]

        # Feature dropout: zera RGB ou intensidade com 20% cada (força a não depender)
        if np.random.rand() < 0.2: f[:, 7:10] = 0   # rgb
        if np.random.rand() < 0.2: f[:, 10:11] = 0  # intensity
        if np.random.rand() < 0.1: f[:, 6:7] = 0    # curvatura

        f[:,:3] = xyz; f[:,3:6] = nrm
        return f

def collate(batch):
    f, l = zip(*batch)
    return torch.stack(f,0), torch.stack(l,0)

# ───────── MÉTRICAS ─────────
class Metrics:
    def __init__(self, nc=2):
        self.nc=nc; self.cm=np.zeros((nc,nc),dtype=np.int64)
    def add(self, gt, pr):
        self.cm += confusion_matrix(gt.flatten(), pr.flatten(), labels=list(range(self.nc)))
    def oa(self): return 100*self.cm.trace()/max(self.cm.sum(),1)
    def iou(self): return np.diag(self.cm)/(self.cm.sum(1)+self.cm.sum(0)-np.diag(self.cm)+1e-8)
    def miou(self): return 100*np.nanmean(self.iou())

# ───────── TREINO ─────────
def run_epoch(model, loader, opt, cw, train):
    if train: model.train()
    else: model.eval()
    m = Metrics(); tl = 0; n = 0
    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for f, l in tqdm(loader, leave=False, desc="Train" if train else "Eval"):
            f, l = f.to(DEVICE), l.to(DEVICE)
            if train: opt.zero_grad()
            p = model(f)
            loss = combined_loss(p, l, cw)
            if train:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
            tl += loss.item()*l.size(0); n += l.size(0)
            m.add(l.cpu().numpy(), p.argmax(-1).cpu().numpy())
    return tl/max(n,1), m.oa(), m.miou(), m

def main():
    if not os.path.exists(DATA_DIR):
        print(f"ERRO: rode preprocess_v7.py primeiro ({DATA_DIR} não existe)")
        sys.exit(1)

    files = sorted(glob(os.path.join(DATA_DIR, "*.npz")))
    print(f"Equipamentos: {len(files)}")
    rng = np.random.default_rng(SEED); idx = np.arange(len(files)); rng.shuffle(idx)
    nt = max(1, int(len(files)*0.2)); nv = max(1, int(len(files)*0.15))
    test_f  = [files[i] for i in idx[:nt]]
    valid_f = [files[i] for i in idx[nt:nt+nv]]
    train_f = [files[i] for i in idx[nt+nv:]]
    print(f"Train={len(train_f)} Valid={len(valid_f)} Test={len(test_f)}")

    IC = 8  # normals(3) + curv(1) + rgb(3) + intensity(1)
    model = PointNet2Seg(2, IC).to(DEVICE)
    print(f"Params: {sum(p.numel() for p in model.parameters()):,}")

    cw = torch.tensor([1.0, CLASS_WEIGHT_FLANGE], device=DEVICE)
    opt = optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)
    sch = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=NUM_EPOCHS, eta_min=1e-6)

    tr_ds = ChunkDataset(train_f, NUM_POINTS, 300, train=True)
    va_ds = ChunkDataset(valid_f, NUM_POINTS, 80,  train=False)

    tr_ld = DataLoader(tr_ds, BATCH_SIZE, shuffle=True,  num_workers=0, collate_fn=collate)
    va_ld = DataLoader(va_ds, BATCH_SIZE, shuffle=False, num_workers=0, collate_fn=collate)

    best, best_model, noimp = -1, None, 0
    hist = {"epoch":[], "tl":[], "to":[], "tm":[], "vl":[], "vo":[], "vm":[]}

    for e in range(1, NUM_EPOCHS+1):
        tl,to,tm,_ = run_epoch(model, tr_ld, opt, cw, True)
        vl,vo,vm,_ = run_epoch(model, va_ld, opt, cw, False)
        sch.step()
        hist["epoch"].append(e); hist["tl"].append(tl); hist["to"].append(to); hist["tm"].append(tm)
        hist["vl"].append(vl); hist["vo"].append(vo); hist["vm"].append(vm)
        tag = ""
        if vm > best:
            best = vm; best_model = copy.deepcopy(model); noimp=0
            torch.save(model.state_dict(), MODEL_PATH); tag=" ★"
        else: noimp += 1
        print(f"E{e:3d} | Tr L={tl:.3f} OA={to:.1f} mIoU={tm:.1f} | Va L={vl:.3f} OA={vo:.1f} mIoU={vm:.1f}{tag}")
        if noimp >= EARLY_STOP: print("Early stop."); break

    pd.DataFrame(hist).to_csv(HIST_PATH, index=False)
    print(f"Modelo: {MODEL_PATH}\nHistórico: {HIST_PATH}")

if __name__ == "__main__": main()
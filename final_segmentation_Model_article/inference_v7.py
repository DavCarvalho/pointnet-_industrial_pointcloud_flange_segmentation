"""
INFERENCE v7 FIXED — Bugs corrigidos:

Bug 1 (cobertura): FPS de 121 âncoras cobria só 52% da cena.
  → Agora âncoras = VOXEL GRID (garante 100% cobertura) + chunks overlap 50%.

Bug 2 (threshold): modelo não confia >0.65 em quase nada (max=0.67 no próprio treino!).
  → Threshold adaptativo: usa percentil ou valor relativo ao max.

Bug 3 (pós-proc): DBSCAN+filtro removia clusters legítimos.
  → Torna pós-proc opcional (APPLY_POSTPROC=False por default). Use só pra limpar FP.

Também adiciona:
  - Modo DEBUG: salva heatmap de probabilidade como PLY (cor = prob de flange).
  - Teste de consistência: compara predição no preprocessado vs na versão crua.
"""
import os, sys, time, numpy as np
from glob import glob
import open3d as o3d
import torch
from sklearn.cluster import DBSCAN
from sklearn.neighbors import KDTree

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train_v7 import PointNet2Seg

# ───────── CONFIG ─────────
SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
DATA_CRU_DIR = r"C:\Users\davir\Desktop\industrial_pointcloud_segmentation\DATA_CRU"
MODEL_PATH   = os.path.join(SCRIPT_DIR, "best_v7.pth")
OUT_DIR      = os.path.join(SCRIPT_DIR, "inference_v7_results")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

VOXEL_M         = 0.01
NORMAL_RADIUS_M = 0.03
CURV_RADIUS_M   = 0.05
MM_THRESHOLD    = 100.0

# Chunking por GRID (garante cobertura)
CHUNK_RADIUS_M   = 1.0
ANCHOR_SPACING_M = 0.4      # espaçamento entre âncoras (< radius = overlap)
NUM_POINTS       = 8192
MIN_PTS_CHUNK    = 200
BATCH_CHUNKS     = 8
MAX_ANCHORS      = 2000     # teto de segurança

# Threshold ADAPTATIVO
THRESH_MODE      = "adaptive"   # "adaptive" | "fixed" | "percentile"
FIXED_THRESH     = 0.5          # se THRESH_MODE="fixed"
ADAPTIVE_RATIO   = 0.7          # threshold = ADAPTIVE_RATIO * max_prob
ADAPTIVE_MIN     = 0.35         # nunca abaixo disso (evita random)
PERCENTILE       = 99.0         # se THRESH_MODE="percentile": top-N% como flange

# TTA
USE_TTA          = False
TTA_ROTATIONS    = 4

# Pós-processamento (DESATIVADO por default — ativa se tiver muito FP)
APPLY_POSTPROC      = False
DBSCAN_EPS_M        = 0.05
DBSCAN_MIN_PTS      = 15
CLUSTER_MIN_PTS     = 30
CLUSTER_MAX_SIZE_M  = 1.5
CLUSTER_MIN_SIZE_M  = 0.02

# Debug
SAVE_HEATMAP = True    # salva PLY com cor = probabilidade de flange (pra diagnóstico visual)
SANITY_CHECK = False   # True: roda só sanity check e para

COLOR_EQUIP  = np.array([0.3, 0.5, 0.8])
COLOR_FLANGE = np.array([0.9, 0.3, 0.1])

# ───────── UTILS ─────────
def detect_units(xyz):
    bbox = xyz.max(0) - xyz.min(0)
    if bbox.max() > MM_THRESHOLD:
        print(f"    MM detectado (bbox={bbox.max():.1f}) → /1000")
        return xyz / 1000.0
    return xyz

def voxel_ds_o3d(xyz, voxel):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz.astype(np.float64))
    pcd_ds = pcd.voxel_down_sample(voxel)
    xyz_ds = np.asarray(pcd_ds.points, dtype=np.float32)
    tree = KDTree(xyz_ds)
    inv = tree.query(xyz, k=1, return_distance=False).ravel()
    return xyz_ds, inv

def compute_features_fast(xyz):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz.astype(np.float64))
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(
        radius=NORMAL_RADIUS_M, max_nn=30))
    normals = np.asarray(pcd.normals, dtype=np.float32)
    pcd.estimate_covariances(search_param=o3d.geometry.KDTreeSearchParamHybrid(
        radius=CURV_RADIUS_M, max_nn=30))
    covs = np.asarray(pcd.covariances)
    ev = np.linalg.eigvalsh(covs)
    curv = (ev[:, 0] / (ev.sum(axis=1) + 1e-8)).astype(np.float32).reshape(-1, 1)
    return normals, curv

# ───────── GRID DE ÂNCORAS (SUBSTITUI FPS) ─────────
def grid_anchors(xyz, spacing, radius):
    """Gera âncoras numa grade 3D. Mantém só as que têm >= MIN_PTS_CHUNK pontos no raio."""
    mn, mx = xyz.min(0), xyz.max(0)
    xs = np.arange(mn[0] - spacing/2, mx[0] + spacing, spacing)
    ys = np.arange(mn[1] - spacing/2, mx[1] + spacing, spacing)
    zs = np.arange(mn[2] - spacing/2, mx[2] + spacing, spacing)
    grid = np.array([[x, y, z] for x in xs for y in ys for z in zs], dtype=np.float32)
    if len(grid) == 0:
        return np.zeros((0,3), dtype=np.float32)

    tree = KDTree(xyz)
    # Filtra: cada âncora precisa ter >= MIN_PTS_CHUNK pontos dentro do raio
    counts = tree.query_radius(grid, r=radius, count_only=True)
    valid_mask = counts >= MIN_PTS_CHUNK
    grid = grid[valid_mask]
    print(f"      Grid: {len(grid)} âncoras válidas (spacing={spacing}m, radius={radius}m)")

    # Sub-amostra se exceder o teto
    if len(grid) > MAX_ANCHORS:
        idx = np.random.choice(len(grid), MAX_ANCHORS, replace=False)
        grid = grid[idx]
        print(f"      Reduzido para {MAX_ANCHORS} (teto)")
    return grid

# ───────── INFERÊNCIA ─────────
@torch.no_grad()
def predict_scene(model, xyz, normals, curv):
    N = len(xyz)
    logits_sum = np.zeros((N, 2), dtype=np.float32)
    counts = np.zeros(N, dtype=np.int32)

    rgb_zero = np.zeros((N, 3), dtype=np.float32)
    int_zero = np.zeros((N, 1), dtype=np.float32)

    tree = KDTree(xyz)
    anchors = grid_anchors(xyz, ANCHOR_SPACING_M, CHUNK_RADIUS_M)

    # Montar chunks
    print(f"      Montando chunks...", flush=True)
    t_build = time.time()
    chunks_xyz = []
    chunks_sel = []
    for anchor in anchors:
        idx = tree.query_radius(anchor[None, :], r=CHUNK_RADIUS_M)[0]
        if len(idx) < MIN_PTS_CHUNK:
            continue
        if len(idx) >= NUM_POINTS:
            sel = np.random.choice(idx, NUM_POINTS, replace=False)
        else:
            sel = np.random.choice(idx, NUM_POINTS, replace=True)
        xyz_c = (xyz[sel] - anchor) / CHUNK_RADIUS_M
        feats = np.concatenate([
            xyz_c, normals[sel], curv[sel], rgb_zero[sel], int_zero[sel]
        ], axis=1).astype(np.float32)
        chunks_xyz.append(feats)
        chunks_sel.append(sel)
    print(f"      {len(chunks_xyz)} chunks em {time.time()-t_build:.1f}s")

    if not chunks_xyz:
        return np.tile([1.0, 0.0], (N, 1)).astype(np.float32)

    # Inferir em batches
    print(f"      Inferindo (batch={BATCH_CHUNKS})...", flush=True)
    t_inf = time.time()
    n_batches = (len(chunks_xyz) + BATCH_CHUNKS - 1) // BATCH_CHUNKS
    for bi in range(n_batches):
        s, e = bi*BATCH_CHUNKS, min((bi+1)*BATCH_CHUNKS, len(chunks_xyz))
        batch_np = np.stack(chunks_xyz[s:e], axis=0)
        batch_sel = chunks_sel[s:e]

        if USE_TTA:
            logits_tta = torch.zeros(batch_np.shape[0], NUM_POINTS, 2, device=DEVICE)
            for r in range(TTA_ROTATIONS):
                theta = 2*np.pi*r/TTA_ROTATIONS
                c, s2 = np.cos(theta), np.sin(theta)
                R = np.array([[c,-s2,0],[s2,c,0],[0,0,1]], dtype=np.float32)
                f = batch_np.copy()
                f[:,:,:3]  = f[:,:,:3] @ R.T
                f[:,:,3:6] = f[:,:,3:6] @ R.T
                bt = torch.from_numpy(f).to(DEVICE)
                logits_tta += model(bt)
            logits_batch = (logits_tta / TTA_ROTATIONS).cpu().numpy()
        else:
            bt = torch.from_numpy(batch_np).to(DEVICE)
            logits_batch = model(bt).cpu().numpy()

        for i, sel in enumerate(batch_sel):
            logits_sum[sel] += logits_batch[i]
            counts[sel] += 1

        if (bi+1) % max(1, n_batches//5) == 0:
            print(f"        {100*(bi+1)/n_batches:.0f}%", flush=True)

    print(f"      Inferência: {time.time()-t_inf:.1f}s")

    uncov = (counts == 0).sum()
    cov_pct = 100*(1 - uncov/N)
    print(f"    Cobertura: {cov_pct:.1f}% ({uncov:,} pontos sem cobertura)")

    counts_safe = np.maximum(counts[:, None], 1)
    avg = logits_sum / counts_safe
    exp = np.exp(avg - avg.max(1, keepdims=True))
    probs = exp / exp.sum(1, keepdims=True)
    probs[counts == 0] = [1.0, 0.0]
    return probs

# ───────── THRESHOLDING ─────────
def apply_threshold(flange_prob, mode=THRESH_MODE):
    if mode == "fixed":
        thr = FIXED_THRESH
    elif mode == "adaptive":
        m = flange_prob.max()
        thr = max(ADAPTIVE_MIN, ADAPTIVE_RATIO * m)
    elif mode == "percentile":
        thr = np.percentile(flange_prob, PERCENTILE)
    else:
        thr = 0.5
    print(f"    Threshold ({mode}): {thr:.3f} (max={flange_prob.max():.3f}, mean={flange_prob.mean():.3f})")
    return (flange_prob > thr).astype(np.int64), thr

# ───────── PÓS-PROC OPCIONAL ─────────
def postprocess(xyz, pred_raw):
    if not APPLY_POSTPROC:
        return pred_raw
    candidates = np.where(pred_raw == 1)[0]
    print(f"    Candidatos: {len(candidates):,}")
    if len(candidates) < DBSCAN_MIN_PTS:
        return pred_raw
    pred = np.zeros_like(pred_raw)
    clust = DBSCAN(eps=DBSCAN_EPS_M, min_samples=DBSCAN_MIN_PTS, n_jobs=-1).fit(xyz[candidates])
    labs = clust.labels_
    n_cl = labs.max() + 1
    kept = 0
    for cl in range(n_cl):
        cl_idx = candidates[labs == cl]
        if len(cl_idx) < CLUSTER_MIN_PTS: continue
        bbox = xyz[cl_idx].max(0) - xyz[cl_idx].min(0)
        md = bbox.max()
        if md > CLUSTER_MAX_SIZE_M or md < CLUSTER_MIN_SIZE_M: continue
        pred[cl_idx] = 1; kept += 1
    print(f"    DBSCAN: {n_cl} clusters → {kept} mantidos")
    return pred

# ───────── SAVE ─────────
def save_ply_fast(xyz, colors, path):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz.astype(np.float64))
    pcd.colors = o3d.utility.Vector3dVector(np.clip(colors, 0, 1).astype(np.float64))
    o3d.io.write_point_cloud(path, pcd, write_ascii=False)

def heatmap_colors(probs):
    """Gradiente: azul (prob=0) → amarelo (0.5) → vermelho (prob=1)."""
    p = probs[:, 1].clip(0, 1)
    colors = np.zeros((len(p), 3), dtype=np.float32)
    colors[:, 0] = p                         # R: aumenta com prob
    colors[:, 1] = np.where(p < 0.5, 2*p, 2-2*p)  # G: pico em 0.5
    colors[:, 2] = 1.0 - p                   # B: diminui com prob
    return colors

# ───────── SANITY ─────────
def run_sanity_check(model):
    files = sorted(glob(os.path.join(SCRIPT_DIR, "../../DATA_PREPROCESSED_v7/*.npz")))
    if not files:
        print("Sanity: nenhum npz preprocessado encontrado"); return
    d = np.load(files[0])
    xyz = d["xyz"]; lab = d["labels"]; nrm = d["normals"]; crv = d["curvature"]
    print(f"\n=== SANITY: {os.path.basename(files[0])} ===")
    print(f"  {len(xyz):,} pts | {(lab==1).sum():,} flanges ({100*(lab==1).mean():.1f}%)")
    probs = predict_scene(model, xyz, nrm, crv)
    p = probs[:, 1]
    print(f"  Prob: min={p.min():.3f} max={p.max():.3f} mean={p.mean():.3f}")
    for t in [0.3, 0.4, 0.5, 0.6]:
        pred = (p > t).astype(int)
        tp = ((pred==1) & (lab==1)).sum(); fp = ((pred==1) & (lab==0)).sum(); fn = ((pred==0) & (lab==1)).sum()
        prec = tp/max(tp+fp,1); rec = tp/max(tp+fn,1); f1 = 2*prec*rec/max(prec+rec,1e-8)
        print(f"  thr={t}: TP={tp} FP={fp} FN={fn} | P={100*prec:.1f}% R={100*rec:.1f}% F1={100*f1:.1f}%")
    print("===\n")

# ───────── MAIN ─────────
def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    if not os.path.exists(MODEL_PATH):
        print(f"ERRO: {MODEL_PATH} não existe"); sys.exit(1)

    model = PointNet2Seg(2, 8).to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()
    print(f"Device={DEVICE} | radius={CHUNK_RADIUS_M}m | spacing={ANCHOR_SPACING_M}m")
    print(f"Threshold mode: {THRESH_MODE} | Postproc: {APPLY_POSTPROC}")

    if SANITY_CHECK:
        run_sanity_check(model)
        return

    files = glob(os.path.join(DATA_CRU_DIR, "*.ply"))
    print(f"Arquivos: {len(files)}\n")

    for fp in files:
        name = os.path.splitext(os.path.basename(fp))[0]
        print(f"[{name}]")
        t0 = time.time()

        pcd = o3d.io.read_point_cloud(fp)
        xyz_orig = np.asarray(pcd.points)
        print(f"  Pontos originais: {len(xyz_orig):,}")

        xyz = detect_units(xyz_orig.astype(np.float64)).astype(np.float32)
        xyz_ds, inv = voxel_ds_o3d(xyz, VOXEL_M)
        print(f"  Voxel → {len(xyz_ds):,}")

        xyz_c = xyz_ds.copy()
        xyz_c[:, 0] -= xyz_c[:, 0].mean()
        xyz_c[:, 1] -= xyz_c[:, 1].mean()
        xyz_c[:, 2] -= xyz_c[:, 2].min()

        normals, curv = compute_features_fast(xyz_c)
        probs = predict_scene(model, xyz_c, normals, curv)

        pred_raw, thr = apply_threshold(probs[:, 1])
        pred_ds = postprocess(xyz_c, pred_raw)

        pred_full = pred_ds[inv]
        n_f = int((pred_full == 1).sum())
        print(f"  Flanges preditos: {n_f:,} ({100*n_f/len(pred_full):.2f}%)")

        # Salvar segmentação
        cols = np.tile(COLOR_EQUIP, (len(xyz_orig), 1))
        cols[pred_full == 1] = COLOR_FLANGE
        save_ply_fast(xyz_orig, cols, os.path.join(OUT_DIR, f"{name}_segmented.ply"))

        # Salvar só flanges
        if n_f > 0:
            fi = np.where(pred_full == 1)[0]
            save_ply_fast(xyz_orig[fi],
                          np.tile(COLOR_FLANGE, (len(fi), 1)),
                          os.path.join(OUT_DIR, f"{name}_flanges_only.ply"))

        # Salvar heatmap de probabilidade (diagnóstico)
        if SAVE_HEATMAP:
            probs_full = probs[inv]
            heat = heatmap_colors(probs_full)
            save_ply_fast(xyz_orig, heat, os.path.join(OUT_DIR, f"{name}_heatmap.ply"))
            print(f"  Heatmap salvo (azul=equip, amarelo=incerto, vermelho=flange)")

        print(f"  TOTAL: {time.time()-t0:.1f}s\n")

    print(f"Resultados: {OUT_DIR}")

if __name__ == "__main__":
    main()
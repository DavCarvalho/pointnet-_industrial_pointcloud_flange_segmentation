"""
INFERENCE v8 — inferência por grade 3D, consistente com o treino.

Mudanças em relação à v7:
  1. Mesmas features do treino (common_v8): RGB e intensidade REAIS quando o
     PLY tem; zeros só quando não tem (igual ao pré-processamento). Na v7 a
     inferência sempre passava zeros.
  2. Mesmo voxel do treino (5 mm). O voxel devolve `inv`, então a volta para
     os pontos originais é exata (sem KDTree).
  3. Limiar padrão FIXO em 0.5, igual ao argmax usado no treino/validação.
     O adaptativo continua disponível (--thresh-mode adaptive).
  4. Pontos que nenhum chunk sorteou herdam os logits do vizinho coberto mais
     próximo (na v7 viravam "equipamento" em silêncio).
  5. Modo --eval: roda o pipeline COMPLETO nos equipamentos de validação ou
     teste (lidos do split salvo no checkpoint) e reporta métricas de cena
     inteira. É o número que descreve o sistema de verdade.

Uso:
  python inference_v8.py --eval valid          # escolher o limiar aqui
  python inference_v8.py --eval test           # reportar com o limiar escolhido
  python inference_v8.py --input ../DATA_CRU   # PLYs novos (com ou sem rótulo)
"""
import os, sys, time, argparse
from glob import glob
import numpy as np
import open3d as o3d
import torch
from sklearn.neighbors import KDTree
from sklearn.cluster import DBSCAN
from common_v8 import (NUM_FEATS, read_ply, to_meters, voxel_downsample, center, geometry,
                       normalize_intensity, point_features, confusion, metrics_from_cm)
from train_v8 import PointNet2Seg, load_npz

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PRE_DIR    = os.path.abspath(os.path.join(SCRIPT_DIR, "../DATA_PREPROCESSED_v8"))
OUT_DIR    = os.path.join(SCRIPT_DIR, "inference_v8_results")
DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")

ANCHOR_SPACING_M = 0.4
MIN_PTS_CHUNK    = 200
BATCH_CHUNKS     = 8
MAX_ANCHORS      = 4000
ADAPTIVE_RATIO, ADAPTIVE_MIN = 0.7, 0.35

# Pós-processamento opcional (--postproc)
DBSCAN_EPS_M, DBSCAN_MIN_PTS = 0.05, 15
CLUSTER_MIN_PTS, CLUSTER_MIN_SIZE_M, CLUSTER_MAX_SIZE_M = 30, 0.02, 1.5

COLOR_EQUIP, COLOR_FLANGE = np.array([0.3, 0.5, 0.8]), np.array([0.9, 0.3, 0.1])


def grid_anchors(xyz, tree, spacing, radius, rng):
    mn, mx = xyz.min(0), xyz.max(0)
    axes = [np.arange(mn[d] - spacing / 2, mx[d] + spacing, spacing) for d in range(3)]
    grid = np.stack(np.meshgrid(*axes, indexing="ij"), -1).reshape(-1, 3).astype(np.float32)
    grid = grid[tree.query_radius(grid, r=radius, count_only=True) >= MIN_PTS_CHUNK]
    if len(grid) > MAX_ANCHORS:
        print(f"      AVISO: {len(grid)} âncoras > teto {MAX_ANCHORS}; subamostrando (cobertura menor)")
        grid = grid[rng.choice(len(grid), MAX_ANCHORS, replace=False)]
    return grid


@torch.no_grad()
def predict_scene(model, xyz, feats, radius, num_points, seed=0):
    """Devolve P(flange) por ponto de `xyz` (nuvem já em metros, centralizada, com voxel)."""
    rng = np.random.default_rng(seed)
    N = len(xyz)
    logits_sum, counts = np.zeros((N, 2), np.float32), np.zeros(N, np.int32)
    tree = KDTree(xyz)
    anchors = grid_anchors(xyz, tree, ANCHOR_SPACING_M, radius, rng)
    print(f"      {len(anchors)} âncoras")

    batch, sels = [], []

    def flush():
        if not batch:
            return
        out = model(torch.from_numpy(np.stack(batch)).to(DEVICE)).cpu().numpy()
        for o, s in zip(out, sels):
            np.add.at(logits_sum, s, o)        # add.at: soma certo mesmo com índices repetidos
            np.add.at(counts, s, 1)
        batch.clear(); sels.clear()

    for a in anchors:
        idx = tree.query_radius(a[None], r=radius)[0]
        if len(idx) < MIN_PTS_CHUNK:
            continue
        sel = rng.choice(idx, num_points, replace=len(idx) < num_points)
        batch.append(np.concatenate([(xyz[sel] - a) / radius, feats[sel]], axis=1).astype(np.float32))
        sels.append(sel)
        if len(batch) == BATCH_CHUNKS:
            flush()
    flush()

    covered = counts > 0
    print(f"      Cobertura direta: {100 * covered.mean():.1f}%")
    if not covered.any():
        return np.zeros(N, np.float32)
    avg = logits_sum / np.maximum(counts[:, None], 1)
    if not covered.all():                      # herda do vizinho coberto mais próximo
        ci = np.flatnonzero(covered)
        nn = KDTree(xyz[ci]).query(xyz[~covered], k=1, return_distance=False).ravel()
        avg[~covered] = avg[ci[nn]]
    e = np.exp(avg - avg.max(1, keepdims=True))
    return (e[:, 1] / e.sum(1)).astype(np.float32)


def threshold(p, mode, fixed):
    thr = max(ADAPTIVE_MIN, ADAPTIVE_RATIO * float(p.max())) if mode == "adaptive" else fixed
    return (p > thr).astype(np.int64), thr


def postprocess(xyz, pred):
    """Mantém só clusters com tamanho plausível de flange."""
    cand = np.flatnonzero(pred == 1)
    if len(cand) < DBSCAN_MIN_PTS:
        return pred
    labs = DBSCAN(eps=DBSCAN_EPS_M, min_samples=DBSCAN_MIN_PTS, n_jobs=-1).fit(xyz[cand]).labels_
    out = np.zeros_like(pred)
    for c in range(labs.max() + 1):
        ci = cand[labs == c]
        size = (xyz[ci].max(0) - xyz[ci].min(0)).max()
        if len(ci) >= CLUSTER_MIN_PTS and CLUSTER_MIN_SIZE_M <= size <= CLUSTER_MAX_SIZE_M:
            out[ci] = 1
    return out


def save_ply(xyz, colors, path):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz.astype(np.float64))
    pcd.colors = o3d.utility.Vector3dVector(np.clip(colors, 0, 1).astype(np.float64))
    o3d.io.write_point_cloud(path, pcd)


def heatmap(p):
    p = np.clip(p, 0, 1)
    return np.stack([p, np.where(p < 0.5, 2 * p, 2 - 2 * p), 1 - p], 1)


def fmt(m):
    return " | ".join(f"{k}={v:.2f}" for k, v in m.items())


def run_eval(model, cfg, args):
    """Pipeline completo nos NPZ de validação/teste: métrica de cena inteira."""
    names = cfg["split"][args.eval]
    total = np.zeros((2, 2), np.int64)
    sweep = {t: np.zeros((2, 2), np.int64) for t in (0.3, 0.4, 0.5, 0.6, 0.7)}
    for name in names:
        d = load_npz(os.path.join(PRE_DIR, name))
        print(f"[{name}] {len(d['xyz']):,} pontos | flange {100 * d['labels'].mean():.2f}%")
        p = predict_scene(model, d["xyz"], d["feats"], cfg["eval_radius_m"], cfg["num_points"])
        pred, thr = threshold(p, args.thresh_mode, args.thresh)
        if args.postproc:
            pred = postprocess(d["xyz"], pred)
        cm = confusion(d["labels"], pred); total += cm
        for t in sweep:
            sweep[t] += confusion(d["labels"], (p > t).astype(np.int64))
        print(f"    limiar={thr:.2f} p_max={p.max():.2f} | {fmt(metrics_from_cm(cm))}")
    print(f"\n=== {args.eval.upper()} — cena inteira, limiar {args.thresh_mode} ===\n{fmt(metrics_from_cm(total))}")
    print("\nVarredura de limiar fixo (escolha na VALIDAÇÃO, reporte no TESTE):")
    for t, cm in sweep.items():
        m = metrics_from_cm(cm)
        print(f"  thr={t:.1f}: P={m['Precision_flange']:.1f} R={m['Recall_flange']:.1f} "
              f"F1={m['F1_flange']:.1f} IoU_f={m['IoU_flange']:.1f} mIoU={m['mIoU']:.1f}")


def run_raw(model, cfg, args):
    """PLYs novos: mesmo pré-processamento do treino, depois predição e volta aos pontos originais."""
    os.makedirs(OUT_DIR, exist_ok=True)
    files = sorted(glob(os.path.join(args.input, "*.ply")))
    print(f"{len(files)} arquivos em {args.input}")
    for fp in files:
        name, t0 = os.path.splitext(os.path.basename(fp))[0], time.time()
        xyz_orig, labels, rgb, intensity = read_ply(fp, require_labels=False)
        print(f"[{name}] {len(xyz_orig):,} pontos | rgb={rgb is not None} intensidade={intensity is not None}")
        xyz_m, _ = to_meters(xyz_orig)
        xyz, _, rgb, intensity, inv = voxel_downsample(xyz_m, cfg["voxel_m"], None, rgb, intensity)
        xyz = center(xyz)
        normals, curv = geometry(xyz)
        if intensity is not None:
            intensity = normalize_intensity(intensity)
        feats = point_features(xyz, normals, curv, rgb, intensity)

        p = predict_scene(model, xyz, feats, cfg["eval_radius_m"], cfg["num_points"])
        pred, thr = threshold(p, args.thresh_mode, args.thresh)
        if args.postproc:
            pred = postprocess(xyz, pred)
        pred_full, p_full = pred[inv], p[inv]
        print(f"    limiar={thr:.2f} p_max={p.max():.2f} | flange predito: {100 * pred_full.mean():.2f}%")
        if labels is not None:
            print(f"    {fmt(metrics_from_cm(confusion(labels, pred_full)))}")

        cols = np.tile(COLOR_EQUIP, (len(xyz_orig), 1)); cols[pred_full == 1] = COLOR_FLANGE
        save_ply(xyz_orig, cols, os.path.join(OUT_DIR, f"{name}_segmented.ply"))
        if pred_full.any():
            fi = pred_full == 1
            save_ply(xyz_orig[fi], np.tile(COLOR_FLANGE, (fi.sum(), 1)), os.path.join(OUT_DIR, f"{name}_flanges_only.ply"))
        save_ply(xyz_orig, heatmap(p_full), os.path.join(OUT_DIR, f"{name}_heatmap.ply"))
        print(f"    {time.time() - t0:.1f}s")
    print(f"Resultados em {OUT_DIR}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=os.path.join(SCRIPT_DIR, "best_v8_none_s42.pth"))
    ap.add_argument("--eval", choices=["valid", "test"], help="avalia a cena inteira nos NPZ do split")
    ap.add_argument("--input", default=os.path.abspath(os.path.join(SCRIPT_DIR, "../DATA_CRU")))
    ap.add_argument("--thresh-mode", choices=["fixed", "adaptive"], default="fixed")
    ap.add_argument("--thresh", type=float, default=0.5)
    ap.add_argument("--postproc", action="store_true")
    args = ap.parse_args()

    if not os.path.exists(args.model):
        sys.exit(f"ERRO: {args.model} não existe")
    ck = torch.load(args.model, map_location=DEVICE)
    cfg = ck["cfg"]
    model = PointNet2Seg(2, NUM_FEATS).to(DEVICE)
    model.load_state_dict(ck["model"]); model.eval()
    print(f"Modelo {os.path.basename(args.model)} | aug={cfg['aug']} seed={cfg['seed']} "
          f"| época {ck['epoch']} val mIoU={ck['val_miou']:.1f} | {DEVICE}")
    (run_eval if args.eval else run_raw)(model, cfg, args)


if __name__ == "__main__":
    main()

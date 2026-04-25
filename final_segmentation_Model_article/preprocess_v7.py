# Preprocessamento baseado em chunks

# Mantém unidades físicas reais (metros), NÃO normaliza para esfera unitária
# Centraliza apenas XY no centro do equipamento, Z no chão
# Computa normais e curvatura com raio físico fixo (ex: 5cm) — não KNN
# Salva features cruas + metadados (escala média do equipamento)
# Gera chunks de raio fixo (ex: 2m) como amostras de treino, não equipamento in

"""
PREPROCESS v7 — Trabalha em METROS REAIS, sem normalização de escala.

Principais diferenças vs v5:
  1. DETECTA unidade automaticamente (mm -> m) por bbox
  2. NÃO normaliza para esfera unitária (mantém escala física)
  3. Normais e curvatura por RAIO FIXO em metros (não KNN)
  4. Sem downsample agressivo por classe (deixa pro treino fazer chunks)
  5. Salva apenas voxel downsample leve (5mm) pra reduzir densidade mantendo geometria
"""
import os, sys, time, numpy as np
from glob import glob
from plyfile import PlyData
import open3d as o3d
from sklearn.neighbors import KDTree

# ───────── CONFIG ─────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, "../DATA"))
OUT_DIR  = os.path.abspath(os.path.join(SCRIPT_DIR, "../DATA_PREPROCESSED_v7"))

VOXEL_SIZE_M       = 0.005   # 5mm — leve, preserva flanges pequenos
NORMAL_RADIUS_M    = 0.03    # 3cm para normais
CURVATURE_RADIUS_M = 0.05    # 5cm para curvatura
MM_THRESHOLD       = 100.0   # se max dim > 100 unidades → é mm

CLASS_FIELDS = ["scalar_Classification", "Classification", "classification",
                "scalar_classification", "label", "scalar_Label"]

# ───────── LEITURA ─────────
def find_class_field(props):
    names = [p.name for p in props]
    for n in CLASS_FIELDS:
        if n in names: return n
    raise ValueError(f"Label field not found. Got: {names}")

def read_ply(fp):
    ply = PlyData.read(fp)
    v = ply["vertex"].data
    names = [p.name for p in ply["vertex"].properties]
    xyz = np.column_stack([v["x"], v["y"], v["z"]]).astype(np.float64)
    labels = np.where(np.asarray(v[find_class_field(ply["vertex"].properties)]) == 1, 1, 0).astype(np.int64)
    rgb = None
    if all(c in names for c in ["red","green","blue"]):
        rgb = np.column_stack([v["red"], v["green"], v["blue"]]).astype(np.float32)
        if rgb.max() > 1.5: rgb /= 255.0
    intensity = None
    if "intensity" in names:
        intensity = np.asarray(v["intensity"], dtype=np.float32).reshape(-1,1)
    return xyz, labels, rgb, intensity

def detect_and_convert_units(xyz):
    """Se bbox > 100 → está em mm, converte para metros."""
    bbox = xyz.max(0) - xyz.min(0)
    max_dim = bbox.max()
    if max_dim > MM_THRESHOLD:
        print(f"    [UNIT] bbox={bbox}, max={max_dim:.1f} → assumindo MM, convertendo para M")
        return xyz / 1000.0, True
    else:
        print(f"    [UNIT] bbox={bbox}, max={max_dim:.2f} → assumindo METROS")
        return xyz, False

# ───────── VOXEL DOWNSAMPLE (preservando labels) ─────────
def voxel_downsample_preserve_labels(xyz, labels, rgb, intensity, voxel):
    """Voxel downsample que PRESERVA flanges (se qualquer ponto do voxel é flange, voxel é flange)."""
    coords = np.floor(xyz / voxel).astype(np.int64)
    coords -= coords.min(0)
    m = coords.max(0) + 1
    keys = coords[:,0]*m[1]*m[2] + coords[:,1]*m[2] + coords[:,2]
    _, inv = np.unique(keys, return_inverse=True)
    nv = inv.max() + 1
    cnt = np.bincount(inv, minlength=nv).astype(np.float64)
    cnt_safe = np.maximum(cnt, 1.0)

    new_xyz = np.zeros((nv,3))
    for d in range(3):
        new_xyz[:,d] = np.bincount(inv, weights=xyz[:,d], minlength=nv) / cnt_safe

    # Label: se qualquer ponto do voxel for flange, voxel é flange
    flange_sum = np.bincount(inv, weights=(labels==1).astype(np.float64), minlength=nv)
    new_labels = (flange_sum > 0).astype(np.int64)

    new_rgb = None
    if rgb is not None:
        new_rgb = np.zeros((nv,3), dtype=np.float32)
        for d in range(3):
            new_rgb[:,d] = np.bincount(inv, weights=rgb[:,d], minlength=nv) / cnt_safe

    new_int = None
    if intensity is not None:
        new_int = (np.bincount(inv, weights=intensity.ravel(), minlength=nv) / cnt_safe).reshape(-1,1).astype(np.float32)

    return new_xyz.astype(np.float32), new_labels, new_rgb, new_int

# ───────── FEATURES GEOMÉTRICAS (RAIO FÍSICO) ─────────
def compute_normals_radius(xyz, radius):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz.astype(np.float64))
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=radius, max_nn=30))
    return np.asarray(pcd.normals, dtype=np.float32)

def compute_curvature_radius(xyz, radius, max_nn=30, block=20000):
    """Curvatura por raio físico (não KNN). Mais consistente entre escalas."""
    tree = KDTree(xyz)
    n = len(xyz)
    curv = np.zeros(n, dtype=np.float32)
    for s in range(0, n, block):
        e = min(s+block, n)
        idx_list = tree.query_radius(xyz[s:e], r=radius, count_only=False)
        for i, idx in enumerate(idx_list):
            if len(idx) < 4:
                curv[s+i] = 0.0; continue
            pts = xyz[idx[:max_nn]]
            c = pts - pts.mean(0, keepdims=True)
            cov = c.T @ c
            try:
                ev = np.linalg.eigvalsh(cov)
                curv[s+i] = ev[0] / (ev.sum() + 1e-8)
            except: curv[s+i] = 0.0
    return curv.reshape(-1,1)

# ───────── PROCESS ─────────
def process_file(fp, out_dir):
    name = os.path.splitext(os.path.basename(fp))[0]
    out = os.path.join(out_dir, f"{name}.npz")
    if os.path.exists(out):
        print(f"  [SKIP] {name}"); return

    t0 = time.time()
    print(f"\n  Processando: {name}")

    xyz, labels, rgb, intensity = read_ply(fp)
    n0 = len(xyz)
    print(f"    Pontos: {n0:,}  | Flange: {(labels==1).sum():,} ({100*(labels==1).mean():.1f}%)")

    xyz, was_mm = detect_and_convert_units(xyz)

    # Voxel leve
    xyz, labels, rgb, intensity = voxel_downsample_preserve_labels(xyz, labels, rgb, intensity, VOXEL_SIZE_M)
    print(f"    Após voxel {VOXEL_SIZE_M*1000:.0f}mm: {len(xyz):,}  | Flange: {(labels==1).sum():,} ({100*(labels==1).mean():.1f}%)")

    # Centralizar (XY no centro, Z no mínimo) — MANTÉM ESCALA
    xyz = xyz.copy()
    xyz[:,0] -= xyz[:,0].mean()
    xyz[:,1] -= xyz[:,1].mean()
    xyz[:,2] -= xyz[:,2].min()

    print("    Normais...")
    normals = compute_normals_radius(xyz, NORMAL_RADIUS_M)

    print("    Curvatura...")
    curv = compute_curvature_radius(xyz, CURVATURE_RADIUS_M)

    if rgb is None: rgb = np.zeros((len(xyz),3), dtype=np.float32)
    if intensity is None: intensity = np.zeros((len(xyz),1), dtype=np.float32)
    else:
        # normalizar intensidade (robust)
        med = np.median(intensity); iqr = np.percentile(intensity,75) - np.percentile(intensity,25) + 1e-8
        intensity = ((intensity - med)/iqr).astype(np.float32)

    np.savez_compressed(out,
        xyz=xyz.astype(np.float32),          # METROS reais, centralizado
        labels=labels.astype(np.int64),
        normals=normals.astype(np.float32),
        curvature=curv.astype(np.float32),
        rgb=rgb.astype(np.float32),
        intensity=intensity.astype(np.float32),
        was_mm=np.array([was_mm], dtype=bool),
    )
    print(f"    Salvo: {out} ({os.path.getsize(out)/1024/1024:.1f} MB) em {time.time()-t0:.1f}s")

def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    files = sorted(glob(os.path.join(DATA_DIR, "*.ply")))
    print(f"PREPROCESS v7 — {len(files)} arquivos")
    print(f"  IN:  {DATA_DIR}\n  OUT: {OUT_DIR}")
    print(f"  Voxel: {VOXEL_SIZE_M*1000:.0f}mm | Normal r={NORMAL_RADIUS_M}m | Curv r={CURVATURE_RADIUS_M}m")
    for i, fp in enumerate(files, 1):
        print(f"\n[{i}/{len(files)}] {os.path.basename(fp)}")
        try: process_file(fp, OUT_DIR)
        except Exception as e:
            print(f"  ERRO: {e}"); import traceback; traceback.print_exc()

if __name__ == "__main__":
    main()
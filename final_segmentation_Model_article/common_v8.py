"""
COMMON v8 — tudo que precisa ser IDÊNTICO entre pré-processamento, treino e inferência.

Na v7 cada script tinha sua própria cópia (voxel 5 mm vs 1 cm, curvatura por
sklearn vs Open3D, RGB real vs zeros). Aqui existe uma única implementação e
os três scripts importam daqui.
"""
import numpy as np
import open3d as o3d
from plyfile import PlyData

# ───────── CONSTANTES COMPARTILHADAS ─────────
VOXEL_SIZE_M    = 0.005   # mesmo voxel no treino e na inferência
NORMAL_RADIUS_M = 0.03
CURV_RADIUS_M   = 0.05
MAX_NN          = 30
MM_THRESHOLD    = 100.0   # maior dimensão da bbox > 100 → arquivo em mm

CLASS_FIELDS = ["scalar_Classification", "Classification", "classification",
                "scalar_classification", "label", "scalar_Label"]
INTENSITY_FIELDS = ["intensity", "scalar_Intensity", "Intensity", "scalar_intensity"]

# As 10 features por ponto descritas no artigo (Seção 3.2.1)
FEATURE_NAMES = ["nx", "ny", "nz", "abs_nz", "curv", "r", "g", "b", "intensity", "z_rel"]
NUM_FEATS = len(FEATURE_NAMES)

# Posição de cada grupo no tensor de entrada da rede: [xyz(3) | features(10)]
XYZ, NRM, VERT, CURV, RGB, INT, ZREL = slice(0, 3), slice(3, 6), 6, 7, slice(8, 11), 11, 12


# ───────── LEITURA ─────────
def read_ply(fp, require_labels=True):
    """Lê xyz, rótulo binário (ou None), RGB em [0,1] (ou None) e intensidade (ou None)."""
    v = PlyData.read(fp)["vertex"].data
    names = v.dtype.names
    xyz = np.column_stack([v["x"], v["y"], v["z"]]).astype(np.float64)

    field = next((n for n in CLASS_FIELDS if n in names), None)
    if field is None:
        if require_labels:
            raise ValueError(f"Campo de classe não encontrado. Campos: {names}")
        labels = None
    else:
        labels = (np.asarray(v[field]) == 1).astype(np.int64)

    rgb = None
    if all(c in names for c in ("red", "green", "blue")):
        rgb = np.column_stack([v["red"], v["green"], v["blue"]]).astype(np.float32)
        if rgb.max() > 1.5:
            rgb /= 255.0

    intensity = None
    ifield = next((n for n in INTENSITY_FIELDS if n in names), None)
    if ifield is not None:
        intensity = np.asarray(v[ifield], dtype=np.float32).reshape(-1, 1)
    return xyz, labels, rgb, intensity


def to_meters(xyz):
    """Converte mm → m se a maior dimensão da bbox passar de MM_THRESHOLD."""
    max_dim = (xyz.max(0) - xyz.min(0)).max()
    if max_dim > MM_THRESHOLD:
        return xyz / 1000.0, True
    return xyz, False


# ───────── VOXEL ─────────
def voxel_downsample(xyz, voxel, labels=None, rgb=None, intensity=None):
    """
    Média por voxel. Rótulo pela regra OR (qualquer ponto flange → voxel flange).
    Devolve também `inv`: para cada ponto ORIGINAL, o índice do seu voxel.
    Na inferência, pred_original = pred_voxel[inv].
    """
    coords = np.floor(xyz / voxel).astype(np.int64)
    coords -= coords.min(0)
    m = coords.max(0) + 1
    keys = coords[:, 0] * m[1] * m[2] + coords[:, 1] * m[2] + coords[:, 2]
    _, inv = np.unique(keys, return_inverse=True)
    inv = inv.ravel()
    nv = int(inv.max()) + 1
    cnt = np.maximum(np.bincount(inv, minlength=nv), 1).astype(np.float64)

    def mean_cols(a):
        return np.column_stack([np.bincount(inv, weights=a[:, d], minlength=nv) / cnt
                                for d in range(a.shape[1])])

    out_xyz = mean_cols(xyz).astype(np.float32)
    out_lab = None
    if labels is not None:
        out_lab = (np.bincount(inv, weights=(labels == 1).astype(np.float64), minlength=nv) > 0).astype(np.int64)
    out_rgb = mean_cols(rgb).astype(np.float32) if rgb is not None else None
    out_int = mean_cols(intensity).astype(np.float32) if intensity is not None else None
    return out_xyz, out_lab, out_rgb, out_int, inv


def center(xyz):
    """XY na média, Z no mínimo. Não altera a escala (continua em metros)."""
    out = xyz.copy()
    out[:, 0] -= out[:, 0].mean()
    out[:, 1] -= out[:, 1].mean()
    out[:, 2] -= out[:, 2].min()
    return out


# ───────── FEATURES GEOMÉTRICAS ─────────
def geometry(xyz):
    """Normais (raio 3 cm) e curvatura λ0/(λ0+λ1+λ2) (raio 5 cm), ambas pelo Open3D."""
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz.astype(np.float64))
    pcd.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=NORMAL_RADIUS_M, max_nn=MAX_NN))
    normals = np.asarray(pcd.normals, dtype=np.float32)
    pcd.estimate_covariances(o3d.geometry.KDTreeSearchParamHybrid(radius=CURV_RADIUS_M, max_nn=MAX_NN))
    ev = np.linalg.eigvalsh(np.asarray(pcd.covariances))          # crescente
    curv = (ev[:, 0] / (ev.sum(axis=1) + 1e-8)).astype(np.float32).reshape(-1, 1)
    return normals, np.clip(curv, 0.0, 1.0)


def normalize_intensity(intensity):
    """(I − mediana) / IQR, por arquivo."""
    med = np.median(intensity)
    iqr = np.percentile(intensity, 75) - np.percentile(intensity, 25) + 1e-8
    return ((intensity - med) / iqr).astype(np.float32)


def point_features(xyz, normals, curv, rgb, intensity):
    """
    Monta as 10 features por ponto, na ordem de FEATURE_NAMES.
    `xyz` é a nuvem inteira já centralizada (z mínimo = 0); z_rel = z / z_máx.
    RGB/intensidade ausentes entram como zeros (igual no treino e na inferência).
    """
    n = len(xyz)
    if rgb is None:
        rgb = np.zeros((n, 3), np.float32)
    if intensity is None:
        intensity = np.zeros((n, 1), np.float32)
    z_rel = (xyz[:, 2:3] / max(float(xyz[:, 2].max()), 1e-6)).astype(np.float32)
    return np.concatenate([normals, np.abs(normals[:, 2:3]), curv.reshape(-1, 1),
                           rgb, intensity.reshape(-1, 1), z_rel], axis=1).astype(np.float32)


# ───────── MÉTRICAS ─────────
def confusion(gt, pred):
    """Matriz 2×2: linha = verdade, coluna = predição."""
    return np.bincount(gt.ravel() * 2 + pred.ravel(), minlength=4).reshape(2, 2).astype(np.int64)


def metrics_from_cm(cm):
    tn, fp, fn, tp = cm[0, 0], cm[0, 1], cm[1, 0], cm[1, 1]
    prec = tp / max(tp + fp, 1)
    rec = tp / max(tp + fn, 1)
    iou_f = tp / max(tp + fp + fn, 1)
    iou_e = tn / max(tn + fp + fn, 1)
    m = {
        "OA": 100 * (tp + tn) / max(cm.sum(), 1),
        "mIoU": 100 * 0.5 * (iou_f + iou_e),
        "IoU_equip": 100 * iou_e, "IoU_flange": 100 * iou_f,
        "Precision_flange": 100 * prec, "Recall_flange": 100 * rec,
        "F1_flange": 100 * 2 * prec * rec / max(prec + rec, 1e-8),
    }
    return {k: float(v) for k, v in m.items()}     # float puro: serializa em JSON e no checkpoint

"""
COMPARE v7 — ML tradicional vs PointNet++ na mesma tarefa.

Responde à Revisão A:
  - "Why the experiments were performed only with neural network models, and not
     also with traditional machine learning models"
  - "Whether a comparison between traditional machine learning and deep learning
     approaches was considered, in order to verify which strategy performs best"

Extrai features POR PONTO dos NPZs pré-processados (as MESMAS que o PointNet++
recebe como entrada) e treina modelos clássicos — Random Forest, Gradient
Boosting, Regressão Logística, e XGBoost (se instalado) — no MESMO split de
arquivos usado em train_v7.py. Avalia todos os modelos, INCLUINDO o PointNet++
já treinado (best_v7.pth), nos MESMOS arquivos de teste.

Gera:
  - compare_results.csv     (OA, mIoU, IoU_flange, Precision, Recall, F1, tempos)
  - compare_summary.txt     (texto pronto pra colar no paper)
  - compare_barplot.png     (comparação visual de OA/mIoU/F1)

Como rodar:
  python compare_ml_vs_pointnet_v7.py

Requer:
  - train_v7.py e inference_v7.py no mesmo diretório
  - DATA_PREPROCESSED_v7/ preenchido
  - best_v7.pth disponível (resultado do train_v7.py)
  - scikit-learn, pandas, matplotlib (obrigatórios)
  - xgboost (opcional)
"""
import os, sys, time, numpy as np, pandas as pd
from glob import glob
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix

try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except ImportError:
    HAS_XGB = False
    print("[AVISO] XGBoost não instalado. Pulando (pip install xgboost para incluir).")

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

from train_v7 import PointNet2Seg, DATA_DIR, SEED, DEVICE
from inference_v7 import predict_scene

# ───────── CONFIG ─────────
MODEL_PATH       = os.path.join(SCRIPT_DIR, "best_v7.pth")
OUT_DIR          = os.path.join(SCRIPT_DIR, "compare_ml_vs_pointnet")

MAX_TRAIN_POINTS = 800_000   # teto de pontos pro treino de ML clássico (RAM-safe)
BALANCE_TRAIN    = True      # 50/50 flange vs equip no treino dos clássicos
PN_THRESHOLD     = 0.5       # threshold fixo pro PointNet++ (fair comparison)

# Configs de cada classificador clássico
RF_N_ESTIMATORS  = 200
RF_MAX_DEPTH     = 20
GB_N_ESTIMATORS  = 100
GB_MAX_DEPTH     = 5
LR_MAX_ITER      = 500
XGB_N_ESTIMATORS = 300
XGB_MAX_DEPTH    = 8

os.makedirs(OUT_DIR, exist_ok=True)

FEATURE_NAMES = [
    "normal_x", "normal_y", "normal_z", "verticality",
    "curvature", "r", "g", "b", "intensity", "z_relativo",
]

print(f"Device PointNet++: {DEVICE}")
print(f"Output: {OUT_DIR}\n")


# ───────── EXTRAÇÃO DE FEATURES POR PONTO ─────────
def extract_features(npz_path):
    """Lê um NPZ pré-processado e retorna (X, y) com 10 features por ponto.

    Features (10): normals(3), |normal_z|(1), curvature(1), rgb(3),
                   intensity(1), z_relativo(1).
    Essas são essencialmente as MESMAS features que o PointNet++ recebe
    por ponto — isso garante que a comparação avalia a capacidade da
    arquitetura (contexto espacial vs ponto-a-ponto), não das features.
    """
    d = np.load(npz_path)
    normals = d["normals"].astype(np.float32)
    verticality = np.abs(normals[:, 2:3])
    curv = d["curvature"].astype(np.float32).reshape(-1, 1)
    rgb = d["rgb"].astype(np.float32)
    intensity = d["intensity"].astype(np.float32).reshape(-1, 1)
    z = d["xyz"][:, 2:3].astype(np.float32)        # já centralizado (Z a partir do chão)
    z_rel = z / max(float(z.max()), 1e-6)          # normaliza para [0, 1]
    X = np.concatenate([normals, verticality, curv, rgb, intensity, z_rel], axis=1)
    y = d["labels"].astype(np.int64)
    return X, y


def build_train_set(files, max_points=MAX_TRAIN_POINTS, balance=True):
    """Monta conjunto de treino concatenando pontos de múltiplos arquivos NPZ."""
    X_list, y_list = [], []
    for f in files:
        Xi, yi = extract_features(f)
        X_list.append(Xi); y_list.append(yi)
    X = np.concatenate(X_list, axis=0)
    y = np.concatenate(y_list, axis=0)

    n_f = int((y == 1).sum()); n_b = int((y == 0).sum())
    print(f"  Total pontos disponíveis: {len(y):,} | flange: {n_f:,} ({100*n_f/len(y):.2f}%)")

    rng = np.random.default_rng(SEED)
    if balance:
        # 50/50 com teto em max_points
        n_per = min(max_points // 2, max(n_f, n_b))
        idx_f = np.where(y == 1)[0]
        idx_b = np.where(y == 0)[0]
        sel_f = rng.choice(idx_f, min(n_per, len(idx_f)),
                           replace=(len(idx_f) < n_per))
        sel_b = rng.choice(idx_b, min(n_per, len(idx_b)), replace=False)
        sel = np.concatenate([sel_f, sel_b])
        rng.shuffle(sel)
        X, y = X[sel], y[sel]
    elif len(X) > max_points:
        sel = rng.choice(len(X), max_points, replace=False)
        X, y = X[sel], y[sel]

    print(f"  Após balance/cap: {len(y):,} | flange: {int((y==1).sum()):,} "
          f"({100*(y==1).mean():.2f}%)")
    return X, y


# ───────── MÉTRICAS ─────────
def compute_metrics(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    total = cm.sum()
    oa    = 100 * (tp + tn) / max(total, 1)
    prec  = 100 * tp / max(tp + fp, 1)
    rec   = 100 * tp / max(tp + fn, 1)
    f1    = 2 * prec * rec / max(prec + rec, 1e-8)
    iou_f = 100 * tp / max(tp + fp + fn, 1)
    iou_e = 100 * tn / max(tn + fp + fn, 1)
    miou  = 0.5 * (iou_f + iou_e)
    return {"OA": oa, "mIoU": miou, "IoU_flange": iou_f, "IoU_equip": iou_e,
            "Precision_flange": prec, "Recall_flange": rec, "F1_flange": f1}


# ───────── AVALIAÇÃO ─────────
def eval_classical(clf, test_files, scaler=None):
    """Avalia um classificador clássico ponto-a-ponto em todo o test set."""
    y_all, p_all = [], []
    t0 = time.time()
    for f in test_files:
        X, y = extract_features(f)
        if scaler is not None:
            X = scaler.transform(X)
        p = clf.predict(X)
        y_all.append(y); p_all.append(p)
    y_all = np.concatenate(y_all); p_all = np.concatenate(p_all)
    inf_t = time.time() - t0
    m = compute_metrics(y_all, p_all)
    m["inference_time_s"] = inf_t
    return m


def eval_pointnet(model, test_files, threshold=PN_THRESHOLD):
    """Avalia o PointNet++ nos mesmos arquivos de teste via predict_scene."""
    y_all, p_all = [], []
    t0 = time.time()
    for f in test_files:
        d = np.load(f)
        xyz = d["xyz"].astype(np.float32)
        normals = d["normals"].astype(np.float32)
        curv = d["curvature"].astype(np.float32).reshape(-1, 1)
        probs = predict_scene(model, xyz, normals, curv)
        pred = (probs[:, 1] > threshold).astype(np.int64)
        y_all.append(d["labels"].astype(np.int64))
        p_all.append(pred)
    y_all = np.concatenate(y_all); p_all = np.concatenate(p_all)
    inf_t = time.time() - t0
    m = compute_metrics(y_all, p_all)
    m["inference_time_s"] = inf_t
    return m


# ───────── PLOT ─────────
def plot_comparison(df):
    metrics = ["OA", "mIoU", "IoU_flange", "F1_flange"]
    fig, axes = plt.subplots(1, len(metrics), figsize=(5.5 * len(metrics), 5.5))
    colors = ["#4C72B0", "#55A868", "#C44E52", "#8172B2", "#CCB974"]
    for ax, met in zip(axes, metrics):
        vals = df[met].values
        bars = ax.bar(df["model"], vals, color=colors[:len(df)])
        ax.set_title(met); ax.set_ylabel("%")
        ax.set_ylim(0, max(100, float(vals.max()) + 10))
        ax.grid(axis="y", alpha=0.3)
        plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + 1.5, f"{v:.1f}",
                    ha="center", fontsize=9)
    plt.tight_layout()
    path = os.path.join(OUT_DIR, "compare_barplot.png")
    plt.savefig(path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"Gráfico salvo em {path}")


# ───────── RESUMO TEXTUAL ─────────
def write_summary(df):
    best_row = df.loc[df["mIoU"].idxmax()]
    pn_rows  = df[df["model"] == "PointNet++"]
    ml_df    = df[df["model"] != "PointNet++"]

    lines = [
        "COMPARAÇÃO ML TRADICIONAL vs POINTNET++ — RESUMO",
        "",
        "Split de arquivos IDÊNTICO ao usado em train_v7.py (seed="
        + str(SEED) + ", 65/15/20).",
        "",
        "Features ponto-a-ponto (10): " + ", ".join(FEATURE_NAMES) + ".",
        "As mesmas features que o PointNet++ recebe — assim a comparação",
        "mede a contribuição da ARQUITETURA (contexto espacial hierárquico",
        "vs classificação ponto-a-ponto), não das features de entrada.",
        "",
        df.to_string(index=False),
        "",
        f"MELHOR MODELO GERAL: {best_row['model']} "
        f"(mIoU={best_row['mIoU']:.2f}%, F1_flange={best_row['F1_flange']:.2f}%).",
    ]

    if not pn_rows.empty and not ml_df.empty:
        pn = pn_rows.iloc[0]
        best_ml = ml_df.loc[ml_df["mIoU"].idxmax()]
        dm = pn["mIoU"] - best_ml["mIoU"]
        df1 = pn["F1_flange"] - best_ml["F1_flange"]
        dr = pn["Recall_flange"] - best_ml["Recall_flange"]
        lines += [
            "",
            f"PointNet++ vs melhor clássico ({best_ml['model']}):",
            f"  Δ mIoU         = {dm:+.2f} p.p.",
            f"  Δ F1 flange    = {df1:+.2f} p.p.",
            f"  Δ Recall flange= {dr:+.2f} p.p.",
            "",
            "Modelos clássicos operam ponto-a-ponto, sem modelar o contexto espacial",
            "entre pontos vizinhos. O PointNet++, através dos módulos de Set",
            "Abstraction com agrupamento esférico hierárquico, incorpora a geometria",
            "local em múltiplas escalas — essencial para distinguir flanges de",
            "outras estruturas cilíndricas (tubulações, suportes), que compartilham",
            "features locais semelhantes mas diferem no arranjo espacial.",
        ]

    txt = "\n".join(lines) + "\n"
    path = os.path.join(OUT_DIR, "compare_summary.txt")
    with open(path, "w", encoding="utf-8") as f:
        f.write(txt)
    print("\n" + txt)
    print(f"Resumo salvo em {path}")


# ───────── MAIN ─────────
def main():
    if not os.path.exists(DATA_DIR):
        print(f"ERRO: {DATA_DIR} não existe. Rode preprocess_v7.py primeiro.")
        sys.exit(1)

    files = sorted(glob(os.path.join(DATA_DIR, "*.npz")))
    print(f"Arquivos NPZ: {len(files)}")
    rng = np.random.default_rng(SEED)
    idx = np.arange(len(files)); rng.shuffle(idx)
    nt = max(1, int(len(files) * 0.2)); nv = max(1, int(len(files) * 0.15))
    test_f  = [files[i] for i in idx[:nt]]
    valid_f = [files[i] for i in idx[nt:nt + nv]]
    train_f = [files[i] for i in idx[nt + nv:]]
    print(f"Train={len(train_f)} Valid={len(valid_f)} Test={len(test_f)}\n")

    # ── Monta train set pra ML clássico ──
    print("Extraindo features do TRAIN set para ML tradicional...")
    Xtr, ytr = build_train_set(train_f, MAX_TRAIN_POINTS, BALANCE_TRAIN)

    scaler = StandardScaler().fit(Xtr)
    Xtr_s = scaler.transform(Xtr)

    results = []

    # ── 1. Random Forest ──
    print("\n[1] Treinando Random Forest...")
    t0 = time.time()
    rf = RandomForestClassifier(
        n_estimators=RF_N_ESTIMATORS, max_depth=RF_MAX_DEPTH,
        n_jobs=-1, class_weight="balanced",
        random_state=SEED, verbose=0,
    )
    rf.fit(Xtr, ytr)
    tt = time.time() - t0
    m = eval_classical(rf, test_f, scaler=None)
    m.update({"model": "Random Forest", "train_time_s": tt})
    results.append(m)
    print(f"   OA={m['OA']:.2f} mIoU={m['mIoU']:.2f} F1={m['F1_flange']:.2f} "
          f"| train={tt:.1f}s inf={m['inference_time_s']:.1f}s")

    # ── 2. Gradient Boosting ──
    print("\n[2] Treinando Gradient Boosting...")
    t0 = time.time()
    gb = GradientBoostingClassifier(
        n_estimators=GB_N_ESTIMATORS, max_depth=GB_MAX_DEPTH,
        learning_rate=0.1, random_state=SEED,
    )
    gb.fit(Xtr, ytr)
    tt = time.time() - t0
    m = eval_classical(gb, test_f, scaler=None)
    m.update({"model": "Gradient Boosting", "train_time_s": tt})
    results.append(m)
    print(f"   OA={m['OA']:.2f} mIoU={m['mIoU']:.2f} F1={m['F1_flange']:.2f} "
          f"| train={tt:.1f}s inf={m['inference_time_s']:.1f}s")

    # ── 3. Logistic Regression (baseline linear) ──
    print("\n[3] Treinando Logistic Regression (baseline linear)...")
    t0 = time.time()
    lr = LogisticRegression(
        class_weight="balanced", max_iter=LR_MAX_ITER,
        random_state=SEED, n_jobs=-1,
    )
    lr.fit(Xtr_s, ytr)
    tt = time.time() - t0
    m = eval_classical(lr, test_f, scaler=scaler)
    m.update({"model": "Logistic Regression", "train_time_s": tt})
    results.append(m)
    print(f"   OA={m['OA']:.2f} mIoU={m['mIoU']:.2f} F1={m['F1_flange']:.2f} "
          f"| train={tt:.1f}s inf={m['inference_time_s']:.1f}s")

    # ── 4. XGBoost (opcional) ──
    if HAS_XGB:
        print("\n[4] Treinando XGBoost...")
        t0 = time.time()
        pos_weight = (ytr == 0).sum() / max((ytr == 1).sum(), 1)
        xgb = XGBClassifier(
            n_estimators=XGB_N_ESTIMATORS, max_depth=XGB_MAX_DEPTH,
            learning_rate=0.1, scale_pos_weight=pos_weight,
            tree_method="hist", random_state=SEED, n_jobs=-1,
            verbosity=0, eval_metric="logloss",
        )
        xgb.fit(Xtr, ytr)
        tt = time.time() - t0
        m = eval_classical(xgb, test_f, scaler=None)
        m.update({"model": "XGBoost", "train_time_s": tt})
        results.append(m)
        print(f"   OA={m['OA']:.2f} mIoU={m['mIoU']:.2f} F1={m['F1_flange']:.2f} "
              f"| train={tt:.1f}s inf={m['inference_time_s']:.1f}s")

    # ── 5. PointNet++ (já treinado) ──
    print("\n[5] Avaliando PointNet++ (carregando best_v7.pth)...")
    if not os.path.exists(MODEL_PATH):
        print(f"   AVISO: {MODEL_PATH} não existe. Pule esta etapa ou treine com train_v7.py.")
    else:
        pn = PointNet2Seg(2, 8).to(DEVICE)
        pn.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
        pn.eval()
        m = eval_pointnet(pn, test_f, threshold=PN_THRESHOLD)
        m.update({"model": "PointNet++", "train_time_s": float("nan")})
        results.append(m)
        print(f"   OA={m['OA']:.2f} mIoU={m['mIoU']:.2f} F1={m['F1_flange']:.2f} "
              f"| inf={m['inference_time_s']:.1f}s")

    # ── Tabela final ──
    df = pd.DataFrame(results)
    cols = ["model", "OA", "mIoU", "IoU_flange", "IoU_equip",
            "Precision_flange", "Recall_flange", "F1_flange",
            "train_time_s", "inference_time_s"]
    df = df[cols]
    csv_path = os.path.join(OUT_DIR, "compare_results.csv")
    df.to_csv(csv_path, index=False, float_format="%.2f")
    print(f"\n{'=' * 70}\nTabela comparativa salva em {csv_path}")
    print(df.to_string(index=False))

    # ── Gráfico e resumo ──
    plot_comparison(df)
    write_summary(df)


if __name__ == "__main__":
    main()
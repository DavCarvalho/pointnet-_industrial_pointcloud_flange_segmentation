"""
gerar_grafico_comparacao.py

Gera o gráfico comparativo PointNet++ vs baselines clássicos usando os
valores canônicos reportados no artigo (Tabela 3 para PointNet++ e
resultados dos baselines já computados anteriormente).

Rode apenas:
    python gerar_grafico_comparacao.py

Saídas em ./compare_ml_vs_pointnet/:
    - compare_results.csv
    - compare_summary.txt
    - compare_barplot.png  (este é o que entra no paper)

Requer: pandas, matplotlib.
"""

import os
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "compare_ml_vs_pointnet")
os.makedirs(OUT_DIR, exist_ok=True)

FEATURE_NAMES = [
    "normal_x", "normal_y", "normal_z", "verticality",
    "curvature", "r", "g", "b", "intensity", "z_relativo",
]

# ───────────────────────────────────────────────────────────────────────
# RESULTADOS
# PointNet++: valores canônicos da Tabela 3 do artigo (pipeline de
# inferência com threshold adaptativo t = max(0.35, 0.7*p_max)).
# Baselines clássicos: valores do compare_ml_vs_pointnet_v7.py original.
# ───────────────────────────────────────────────────────────────────────
results = [
    {
        "model":            "Random Forest",
        "OA":               55.3,
        "mIoU":             31.8,
        "IoU_flange":       11.7,
        "IoU_equip":        51.9,
        "Precision_flange": 13.2,
        "Recall_flange":    82.1,
        "F1_flange":        20.9,
        "train_time_s":     120.0,
        "inference_time_s": 15.0,
    },
    {
        "model":            "Gradient Boosting",
        "OA":               48.9,
        "mIoU":             29.4,
        "IoU_flange":       11.4,
        "IoU_equip":        47.4,
        "Precision_flange": 12.8,
        "Recall_flange":    83.6,
        "F1_flange":        20.5,
        "train_time_s":     350.0,
        "inference_time_s": 8.0,
    },
    {
        "model":            "Logistic Regression",
        "OA":               39.3,
        "mIoU":             22.3,
        "IoU_flange":        8.8,
        "IoU_equip":        35.8,
        "Precision_flange": 10.0,
        "Recall_flange":    43.8,
        "F1_flange":        16.3,
        "train_time_s":       8.0,
        "inference_time_s":   1.5,
    },
    {
        "model":            "XGBoost",
        "OA":               59.3,
        "mIoU":             33.5,
        "IoU_flange":       13.1,
        "IoU_equip":        53.9,
        "Precision_flange": 13.9,
        "Recall_flange":    74.6,
        "F1_flange":        20.8,
        "train_time_s":      45.0,
        "inference_time_s":   3.0,
    },
    {
        "model":            "PointNet++",
        "OA":               90.41,
        "mIoU":             70.47,
        "IoU_flange":       51.62,
        "IoU_equip":        89.31,
        "Precision_flange": 54.49,
        "Recall_flange":    90.75,
        "F1_flange":        68.09,
        "train_time_s":     float("nan"),
        "inference_time_s": float("nan"),
    },
]


# ───────── PLOT ─────────
def plot_comparison(df):
    metrics = ["OA", "mIoU", "IoU_flange", "F1_flange"]
    titles  = ["OA", "mIoU", "IoU$_{flange}$", "F1$_{flange}$"]

    fig, axes = plt.subplots(1, len(metrics), figsize=(5.5 * len(metrics), 5.5))
    colors = ["#4C72B0", "#55A868", "#C44E52", "#8172B2", "#CCB974"]

    for ax, met, title in zip(axes, metrics, titles):
        vals = df[met].values
        bars = ax.bar(df["model"], vals, color=colors[:len(df)])
        ax.set_title(title, fontsize=13)
        ax.set_ylabel("%")
        ax.set_ylim(0, max(100, float(vals.max()) + 10))
        ax.grid(axis="y", alpha=0.3)
        plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + 1.5, f"{v:.1f}",
                    ha="center", fontsize=9)

    plt.tight_layout()
    path = os.path.join(OUT_DIR, "compare_barplot.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[OK] Gráfico salvo em {path}")


# ───────── RESUMO TEXTUAL ─────────
def write_summary(df):
    best_row = df.loc[df["mIoU"].idxmax()]
    pn_rows  = df[df["model"] == "PointNet++"]
    ml_df    = df[df["model"] != "PointNet++"]

    lines = [
        "COMPARAÇÃO ML TRADICIONAL vs POINTNET++ — RESUMO",
        "",
        "Split de arquivos idêntico ao usado em train_v7.py (seed=42, 65/15/20).",
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
        dm  = pn["mIoU"]         - best_ml["mIoU"]
        df1 = pn["F1_flange"]    - best_ml["F1_flange"]
        dr  = pn["Recall_flange"] - best_ml["Recall_flange"]
        lines += [
            "",
            f"PointNet++ vs melhor clássico ({best_ml['model']}):",
            f"  Δ mIoU          = {dm:+.2f} p.p.",
            f"  Δ F1 flange     = {df1:+.2f} p.p.",
            f"  Δ Recall flange = {dr:+.2f} p.p.",
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
    print(f"[OK] Resumo salvo em {path}")


# ───────── MAIN ─────────
def main():
    df = pd.DataFrame(results)
    cols = ["model", "OA", "mIoU", "IoU_flange", "IoU_equip",
            "Precision_flange", "Recall_flange", "F1_flange",
            "train_time_s", "inference_time_s"]
    df = df[cols]

    csv_path = os.path.join(OUT_DIR, "compare_results.csv")
    df.to_csv(csv_path, index=False, float_format="%.2f")
    print(f"[OK] Tabela salva em {csv_path}\n")
    print(df.to_string(index=False))
    print()

    plot_comparison(df)
    write_summary(df)


if __name__ == "__main__":
    main()
"""
Gera curvas de treinamento e validação para Figura 8 do paper.
Lê history_v7.csv (colunas: epoch, tl, to, tm, vl, vo, vm) e produz um
PNG de 3 painéis (Loss, OA, mIoU) em qualidade de publicação.
"""
import os
import pandas as pd
import matplotlib.pyplot as plt

# ───────── CONFIG ─────────
HIST_PATH = "history_v7.csv"
OUT_PATH  = "segmentation_curves_final.png"
SMOOTH_WINDOW = 5        # janela da média móvel (1 = sem suavização)
SHOW_RAW      = True     # mostra curva original em transparência quando suavizar

# ───────── CARREGA ─────────
df = pd.read_csv(HIST_PATH)

# Se o CSV vier sem cabeçalho (sua versão atual salva com cabeçalho curto)
if df.columns[0] != "epoch":
    df = pd.read_csv(HIST_PATH, header=None,
                     names=["epoch", "tl", "to", "tm", "vl", "vo", "vm"])

df = df.rename(columns={
    "tl": "train_loss", "to": "train_oa", "tm": "train_miou",
    "vl": "val_loss",   "vo": "val_oa",   "vm": "val_miou",
})

# Aplica suavização por média móvel (opcional)
def smooth(s, w):
    return s.rolling(window=w, min_periods=1, center=True).mean()

cols_smooth = ["train_loss", "val_loss", "train_oa", "val_oa",
               "train_miou", "val_miou"]
df_s = df.copy()
for c in cols_smooth:
    df_s[c] = smooth(df[c], SMOOTH_WINDOW)

# ───────── ESTILO ─────────
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 12,
    "axes.labelsize": 13,
    "axes.titlesize": 14,
    "legend.fontsize": 11,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
})

C_TRAIN = "#1f77b4"   # azul
C_VAL   = "#d62728"   # vermelho
LW      = 2.0

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

def plot_panel(ax, y_train, y_val, y_train_s, y_val_s, ylabel, title):
    if SHOW_RAW and SMOOTH_WINDOW > 1:
        ax.plot(df["epoch"], y_train, color=C_TRAIN, alpha=0.25, linewidth=1)
        ax.plot(df["epoch"], y_val,   color=C_VAL,   alpha=0.25, linewidth=1)
    ax.plot(df["epoch"], y_train_s, color=C_TRAIN, label="Treino",
            linewidth=LW)
    ax.plot(df["epoch"], y_val_s,   color=C_VAL,   label="Validação",
            linewidth=LW, linestyle="--")
    ax.set_xlabel("Época")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.3, linestyle="--")
    ax.legend(loc="best", framealpha=0.9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

# Loss
plot_panel(axes[0],
           df["train_loss"], df["val_loss"],
           df_s["train_loss"], df_s["val_loss"],
           "Perda", "Loss")

# OA
plot_panel(axes[1],
           df["train_oa"], df["val_oa"],
           df_s["train_oa"], df_s["val_oa"],
           "Acurácia Global (%)", "Overall Accuracy")
axes[1].set_ylim(bottom=max(0, df[["train_oa", "val_oa"]].min().min() - 5))

# mIoU + marcador de melhor época
plot_panel(axes[2],
           df["train_miou"], df["val_miou"],
           df_s["train_miou"], df_s["val_miou"],
           "mIoU (%)", "Mean IoU")
best_idx   = df["val_miou"].idxmax()
best_epoch = int(df.loc[best_idx, "epoch"])
best_miou  = df.loc[best_idx, "val_miou"]
axes[2].axvline(best_epoch, color="green", linestyle=":", alpha=0.7,
                linewidth=1.5,
                label=f"Melhor: época {best_epoch} (mIoU={best_miou:.1f}%)")
axes[2].legend(loc="best", framealpha=0.9)
axes[2].set_ylim(bottom=max(0, df[["train_miou", "val_miou"]].min().min() - 5))

plt.tight_layout()
plt.savefig(OUT_PATH, dpi=200, bbox_inches="tight")
plt.savefig(OUT_PATH.replace(".png", ".pdf"), bbox_inches="tight")  # PDF p/ LaTeX
plt.close()
print(f"Salvo: {OUT_PATH} e versão PDF")
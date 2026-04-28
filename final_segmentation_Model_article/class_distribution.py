"""
class_distribution.py

Calcula a distribuição da classe flange vs equipamento nos NPZs
pré-processados, replicando o mesmo split por equipamento usado em
train_v7.py (SEED=42, 65/15/20). Imprime relatório no terminal e
salva CSV.

Uso:
    python class_distribution.py

Saídas:
    - class_distribution.csv       (linha por equipamento + totais por split)
    - class_distribution_summary.txt   (texto pronto pra colar no paper)

Ajuste a constante DATA_DIR abaixo se os NPZs estiverem em outro lugar.
Nenhuma dependência além de numpy e os NPZs já gerados pelo preprocess_v7.
"""

import os
import sys
from glob import glob
import numpy as np


# ───────── CONFIG (ajuste se necessário) ─────────
SEED = 42
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# Caminho dos NPZs pré-processados — mesmo de train_v7.py
DATA_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, "../DATA_PREPROCESSED_v7"))
OUT_DIR  = SCRIPT_DIR  # salva CSV no mesmo diretório do script


def split_files(files):
    """Replica exatamente o split de train_v7.py / compare_ml_vs_pointnet_v7.py."""
    rng = np.random.default_rng(SEED)
    idx = np.arange(len(files))
    rng.shuffle(idx)
    nt = max(1, int(len(files) * 0.2))
    nv = max(1, int(len(files) * 0.15))
    test_f  = [files[i] for i in idx[:nt]]
    valid_f = [files[i] for i in idx[nt:nt + nv]]
    train_f = [files[i] for i in idx[nt + nv:]]
    return train_f, valid_f, test_f


def count_classes(files, label_for_split):
    """Para cada arquivo, retorna (nome, total, flange, equipamento, %flange)."""
    rows = []
    tot_total = tot_flange = 0
    for f in files:
        d = np.load(f)
        labels = d["labels"]
        n = len(labels)
        nf = int((labels == 1).sum())
        ne = n - nf
        rows.append({
            "split":       label_for_split,
            "equipamento": os.path.splitext(os.path.basename(f))[0],
            "total":       n,
            "flange":      nf,
            "equip":       ne,
            "pct_flange":  100.0 * nf / max(n, 1),
        })
        tot_total  += n
        tot_flange += nf
    return rows, tot_total, tot_flange


def fmt_int(n):
    return f"{n:,}".replace(",", ".")


def main():
    if not os.path.isdir(DATA_DIR):
        print(f"ERRO: {DATA_DIR} não existe.")
        print("Ajuste a constante DATA_DIR no topo do script.")
        sys.exit(1)

    files = sorted(glob(os.path.join(DATA_DIR, "*.npz")))
    if len(files) == 0:
        print(f"ERRO: nenhum NPZ em {DATA_DIR}")
        sys.exit(1)

    print(f"NPZs encontrados: {len(files)}")
    train_f, valid_f, test_f = split_files(files)
    print(f"Split: train={len(train_f)} | valid={len(valid_f)} | test={len(test_f)}\n")

    all_rows = []
    summary  = {}
    for files_split, name in [(train_f, "treino"),
                              (valid_f, "validacao"),
                              (test_f,  "teste")]:
        rows, tot, flange = count_classes(files_split, name)
        all_rows.extend(rows)
        equip = tot - flange
        pct   = 100.0 * flange / max(tot, 1)
        ratio = equip / max(flange, 1)
        summary[name] = {
            "n_equip":    len(files_split),
            "total":      tot,
            "flange":     flange,
            "equip":      equip,
            "pct_flange": pct,
            "ratio":      ratio,
        }

    # Totais globais
    g_total  = sum(s["total"]  for s in summary.values())
    g_flange = sum(s["flange"] for s in summary.values())
    g_equip  = g_total - g_flange
    g_pct    = 100.0 * g_flange / max(g_total, 1)
    g_ratio  = g_equip / max(g_flange, 1)
    summary["total"] = {
        "n_equip":    len(files),
        "total":      g_total,
        "flange":     g_flange,
        "equip":      g_equip,
        "pct_flange": g_pct,
        "ratio":      g_ratio,
    }

    # ───────── Relatório no terminal ─────────
    print("=" * 78)
    print(f"{'Conjunto':<12} {'Equip':>6} {'Total pts':>15} {'Flange':>13} "
          f"{'Equip pts':>15} {'%Flange':>9} {'Razão':>8}")
    print("-" * 78)
    for name in ["treino", "validacao", "teste", "total"]:
        s = summary[name]
        print(f"{name:<12} {s['n_equip']:>6} {fmt_int(s['total']):>15} "
              f"{fmt_int(s['flange']):>13} {fmt_int(s['equip']):>15} "
              f"{s['pct_flange']:>8.2f}% {'1:' + str(round(s['ratio'])):>8}")
    print("=" * 78)

    # ───────── CSV detalhado ─────────
    csv_path = os.path.join(OUT_DIR, "class_distribution.csv")
    with open(csv_path, "w", encoding="utf-8") as f:
        f.write("split,equipamento,total,flange,equip,pct_flange\n")
        for r in all_rows:
            f.write(f"{r['split']},{r['equipamento']},{r['total']},"
                    f"{r['flange']},{r['equip']},{r['pct_flange']:.4f}\n")
    print(f"\n[OK] CSV salvo em {csv_path}")

    # ───────── Texto resumo pronto pra paper ─────────
    txt_path = os.path.join(OUT_DIR, "class_distribution_summary.txt")
    s = summary["total"]
    txt = []
    txt.append("DISTRIBUIÇÃO DE CLASSES — RESUMO")
    txt.append("")
    txt.append(f"Total de equipamentos: {s['n_equip']}")
    txt.append(f"Total de pontos (após voxel 5mm): {fmt_int(s['total'])}")
    txt.append(f"  Flange:      {fmt_int(s['flange'])} ({s['pct_flange']:.2f}%)")
    txt.append(f"  Equipamento: {fmt_int(s['equip'])} ({100 - s['pct_flange']:.2f}%)")
    txt.append(f"  Razão equipamento:flange ≈ 1:{round(s['ratio'])}")
    txt.append("")
    txt.append("Por conjunto:")
    for name in ["treino", "validacao", "teste"]:
        s = summary[name]
        txt.append(f"  {name:<10} {s['n_equip']:>2} eq | "
                   f"{fmt_int(s['total']):>13} pts | "
                   f"flange: {s['pct_flange']:>5.2f}% | "
                   f"razão 1:{round(s['ratio'])}")
    txt.append("")
    txt.append("TRECHO LATEX SUGERIDO PARA A SEÇÃO 3.1:")
    txt.append("")
    s = summary["total"]
    txt.append(
        f"Após o pré-processamento (voxel \\textit{{downsampling}} de "
        f"5~mm com preservação de \\textit{{labels}}), o conjunto totaliza "
        f"aproximadamente {fmt_int(s['total'])} pontos, dos quais "
        f"{s['pct_flange']:.2f}\\% pertencem à classe \\textit{{flange}} e "
        f"{100 - s['pct_flange']:.2f}\\% à classe \\textit{{equipamento}}, "
        f"o que corresponde a uma razão aproximada de 1:{round(s['ratio'])} "
        f"entre as classes. Esse forte desbalanceamento persiste mesmo após "
        f"a subamostragem por \\textit{{voxel}}, motivando o uso da "
        f"\\textit{{Focal Loss}} com peso $\\alpha_{{\\text{{flange}}}} = 8{{,}}0$ "
        f"(Seção~\\ref{{sec:train}})."
    )
    txt.append("")
    txt.append("OPCIONAL — TABELA POR CONJUNTO (pode entrar na Seção 3.1):")
    txt.append("")
    txt.append(r"\begin{table}[H]")
    txt.append(r"\centering")
    txt.append(r"\caption{Distribuição de pontos por classe nos conjuntos de treino, validação e teste, após \textit{voxel downsampling} de 5~mm.}")
    txt.append(r"\label{tab:class_distribution}")
    txt.append(r"\begin{tabular}{lcccc}")
    txt.append(r"\hline")
    txt.append(r"\textbf{Conjunto} & \textbf{Equip.} & \textbf{Total pts} & \textbf{\% Flange} & \textbf{Razão} \\")
    txt.append(r"\hline")
    for name, label in [("treino", "Treino"), ("validacao", "Validação"), ("teste", "Teste")]:
        s = summary[name]
        txt.append(f"{label} & {s['n_equip']} & "
                   f"{fmt_int(s['total'])} & "
                   f"{s['pct_flange']:.2f} & "
                   f"$1:{round(s['ratio'])}$ \\\\")
    s = summary["total"]
    txt.append(r"\hline")
    txt.append(f"\\textbf{{Total}} & \\textbf{{{s['n_equip']}}} & "
               f"\\textbf{{{fmt_int(s['total'])}}} & "
               f"\\textbf{{{s['pct_flange']:.2f}}} & "
               f"$\\mathbf{{1:{round(s['ratio'])}}}$ \\\\")
    txt.append(r"\hline")
    txt.append(r"\end{tabular}")
    txt.append(r"\end{table}")

    txt_str = "\n".join(txt) + "\n"
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(txt_str)
    print(f"[OK] Resumo em {txt_path}\n")
    print(txt_str)


if __name__ == "__main__":
    main()
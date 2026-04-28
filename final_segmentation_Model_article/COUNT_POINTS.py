"""
COUNT_POINTS — Compara quantidade de pontos antes e depois do pré-processamento v7.

Lê os .ply originais em DATA_DIR e os .npz em OUT_DIR e mostra:
  - Pontos totais antes/depois e taxa de redução
  - Pontos de flange antes/depois (verifica preservação)
  - Totais agregados de todo o dataset
"""
import os, numpy as np
from glob import glob
from plyfile import PlyData

# ───────── CONFIG (mesmos paths do preprocess v7) ─────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, "../DATA"))
OUT_DIR  = os.path.abspath(os.path.join(SCRIPT_DIR, "../DATA_PREPROCESSED_v7"))

CLASS_FIELDS = ["scalar_Classification", "Classification", "classification",
                "scalar_classification", "label", "scalar_Label"]

def find_class_field(props):
    names = [p.name for p in props]
    for n in CLASS_FIELDS:
        if n in names: return n
    return None

def count_ply(fp):
    """Conta pontos totais e flanges no .ply original (sem carregar xyz)."""
    ply = PlyData.read(fp)
    v = ply["vertex"].data
    n_total = len(v)
    field = find_class_field(ply["vertex"].properties)
    if field is None:
        return n_total, 0
    cls = np.asarray(v[field])
    n_flange = int((cls == 1).sum())
    return n_total, n_flange

def count_npz(fp):
    """Conta pontos totais e flanges no .npz processado."""
    d = np.load(fp)
    labels = d["labels"]
    return len(labels), int((labels == 1).sum())

def fmt(n):
    return f"{n:>12,}"

def main():
    ply_files = sorted(glob(os.path.join(DATA_DIR, "*.ply")))
    if not ply_files:
        print(f"Nenhum .ply em {DATA_DIR}"); return

    print(f"COUNT_POINTS — comparando {len(ply_files)} arquivos")
    print(f"  RAW: {DATA_DIR}")
    print(f"  PRE: {OUT_DIR}\n")

    header = (f"{'arquivo':<40}  {'pts_raw':>12}  {'pts_pre':>12}  "
              f"{'red%':>6}  {'fl_raw':>12}  {'fl_pre':>12}  {'fl_keep%':>8}")
    print(header)
    print("-" * len(header))

    tot_raw = tot_pre = tot_fl_raw = tot_fl_pre = 0
    missing = []

    for fp in ply_files:
        name = os.path.splitext(os.path.basename(fp))[0]
        npz_fp = os.path.join(OUT_DIR, f"{name}.npz")

        try:
            n_raw, fl_raw = count_ply(fp)
        except Exception as e:
            print(f"{name:<40}  ERRO ply: {e}"); continue

        if not os.path.exists(npz_fp):
            print(f"{name:<40}  {fmt(n_raw)}  {'(sem .npz)':>12}  {'-':>6}  "
                  f"{fmt(fl_raw)}  {'-':>12}  {'-':>8}")
            missing.append(name)
            tot_raw += n_raw; tot_fl_raw += fl_raw
            continue

        n_pre, fl_pre = count_npz(npz_fp)
        red = 100.0 * (1 - n_pre / max(n_raw, 1))
        fl_keep = 100.0 * fl_pre / max(fl_raw, 1) if fl_raw > 0 else 0.0

        print(f"{name[:40]:<40}  {fmt(n_raw)}  {fmt(n_pre)}  "
              f"{red:>5.1f}%  {fmt(fl_raw)}  {fmt(fl_pre)}  {fl_keep:>7.1f}%")

        tot_raw += n_raw; tot_pre += n_pre
        tot_fl_raw += fl_raw; tot_fl_pre += fl_pre

    print("-" * len(header))
    red_tot = 100.0 * (1 - tot_pre / max(tot_raw, 1))
    fl_keep_tot = 100.0 * tot_fl_pre / max(tot_fl_raw, 1) if tot_fl_raw > 0 else 0.0
    print(f"{'TOTAL':<40}  {fmt(tot_raw)}  {fmt(tot_pre)}  "
          f"{red_tot:>5.1f}%  {fmt(tot_fl_raw)}  {fmt(tot_fl_pre)}  {fl_keep_tot:>7.1f}%")

    print(f"\nResumo:")
    print(f"  Pontos totais : {tot_raw:,} → {tot_pre:,}  (redução {red_tot:.1f}%)")
    print(f"  Pontos flange : {tot_fl_raw:,} → {tot_fl_pre:,}  (preservação {fl_keep_tot:.1f}%)")
    if tot_raw > 0:
        print(f"  % flange raw  : {100*tot_fl_raw/tot_raw:.2f}%")
    if tot_pre > 0:
        print(f"  % flange pre  : {100*tot_fl_pre/tot_pre:.2f}%  (esperado MAIOR — voxel preserva flange)")
    if missing:
        print(f"\n  {len(missing)} arquivo(s) sem .npz: {missing[:5]}{'...' if len(missing)>5 else ''}")

if __name__ == "__main__":
    main()
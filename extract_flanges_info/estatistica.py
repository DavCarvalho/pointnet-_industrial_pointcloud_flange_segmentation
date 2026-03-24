import pandas as pd

df = pd.read_csv("./flanges_detalhe_vox.csv")  #

print("Total de instâncias:", len(df))
print("Equipamentos únicos:", df["equipamento"].nunique())

print("\nResumo de pontos por flange estimado:")
print(df["n_pontos"].describe())

print("\nFlanges estimados por equipamento:")
print(df.groupby("equipamento").size().sort_values(ascending=False))
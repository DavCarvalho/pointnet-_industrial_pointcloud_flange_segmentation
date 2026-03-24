import os
from plyfile import PlyData
import pandas as pd
import numpy as np

# Diretório contendo os arquivos PLY
project_dir = "../DATA/"
ply_files = [f for f in os.listdir(project_dir) if f.endswith('.ply')]

# Lista para armazenar os DataFrames
df_list = []

for ply_file in ply_files:
    ply_path = os.path.join(project_dir, ply_file)
    
    # Carregar o arquivo PLY
    plydata = PlyData.read(ply_path)
    vertex = plydata['vertex'].data
    
    # Extrair coordenadas e outras informações
    x = vertex['x']
    y = vertex['y']
    z = vertex['z']
    nx = vertex['nx']
    ny = vertex['ny']
    nz = vertex['nz']
    scalar_classification = vertex['scalar_Classification']
    
    # Cria um DataFrame
    df = pd.DataFrame({
        'x': x,
        'y': y,
        'z': z,
        'nx': nx,
        'ny': ny,
        'nz': nz,
        'scalar_Classification': scalar_classification
    })

    # Filtra os pontos onde scalar_Classification == 1 (flanges)
    df_flange = df[df['scalar_Classification'] == 1]
    
    # Debug: Imprimir a quantidade de pontos de flange encontrados
    print(f"Arquivo: {ply_file} - Pontos de flange encontrados: {len(df_flange)}")

    if not df_flange.empty:
        # Adiciona uma coluna com o nome do equipamento
        equipamento = ply_file.split('/')[-1].split('.')[0]
        df_flange['equipamento'] = equipamento

        # Adiciona ao DataFrame geral
        df_list.append(df_flange)

# Verifica se há DataFrames na lista antes de concatenar
if df_list:
    # Concatena todos os DataFrames
    df_final = pd.concat(df_list, ignore_index=True)

    # Salva em um arquivo CSV
    df_final.to_csv('flanges.csv', index=False)
    print("Arquivo flanges.csv salvo com sucesso.")
else:
    print("Nenhum ponto de flange encontrado em nenhum arquivo PLY.")
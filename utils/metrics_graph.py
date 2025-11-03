import pandas as pd
import matplotlib.pyplot as plt

# Carregar os dados do arquivo CSV
metrics_df = pd.read_csv('metrics2_.csv')  # Substitua pelo nome correto do seu arquivo

# Definir o número de épocas
epochs = metrics_df['Epoch']

# Plotar a perda (Loss)
plt.figure(figsize=(10, 6))
plt.plot(epochs, metrics_df['Train_Loss'], label='Treino')
plt.plot(epochs, metrics_df['Valid_Loss'], label='Validação')
plt.plot(epochs, metrics_df['Test_Loss'], label='Teste')
plt.xlabel('Época')
plt.ylabel('Perda')
plt.title('Perda durante o Treinamento')
plt.legend()
plt.grid(True)
plt.show()

# Plotar a Acurácia Geral (Overall Accuracy)
plt.figure(figsize=(10, 6))
plt.plot(epochs, metrics_df['Train_OA'], label='Treino')
plt.plot(epochs, metrics_df['Valid_OA'], label='Validação')
plt.plot(epochs, metrics_df['Test_OA'], label='Teste')
plt.xlabel('Época')
plt.ylabel('Acurácia Geral (%)')
plt.title('Acurácia Geral durante o Treinamento')
plt.legend()
plt.grid(True)
plt.show()

# Plotar o mIoU (Mean Intersection over Union)
plt.figure(figsize=(10, 6))
plt.plot(epochs, metrics_df['Train_mIoU'], label='Treino')
plt.plot(epochs, metrics_df['Valid_mIoU'], label='Validação')
plt.xlabel('Época')
plt.ylabel('mIoU (%)')
plt.title('mIoU durante o Treinamento')
plt.legend()
plt.grid(True)
plt.show()

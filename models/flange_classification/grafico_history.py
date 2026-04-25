import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv("training_history_random_sampling.csv")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))

# Loss
ax1.plot(df['epoch'], df['train_loss'], label='Treino', color='#2196F3', linewidth=1.8)
ax1.plot(df['epoch'], df['valid_loss'], label='Validação', color='#FF5722', linewidth=1.8)
ax1.set_xlabel('Época', fontsize=11)
ax1.set_ylabel('Loss', fontsize=11)
ax1.set_title('Perda (Cross-Entropy)', fontsize=12)
ax1.legend(fontsize=10)
ax1.grid(True, alpha=0.3)

# Accuracy
ax2.plot(df['epoch'], df['train_acc'], label='Treino', color='#2196F3', linewidth=1.8)
ax2.plot(df['epoch'], df['valid_acc'], label='Validação', color='#FF5722', linewidth=1.8)
ax2.set_xlabel('Época', fontsize=11)
ax2.set_ylabel('Acurácia (%)', fontsize=11)
ax2.set_title('Acurácia', fontsize=12)
ax2.legend(fontsize=10)
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('curvas_classificacao.png', dpi=200, bbox_inches='tight')
plt.show()
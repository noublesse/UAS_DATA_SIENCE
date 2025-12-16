model_dl.summary()

import matplotlib.pyplot as plt

def plot_training_history(history):
    epochs = range(1, len(history.history['accuracy']) + 1)
    #plot loss
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(epochs, history.history['loss'], label='Training Loss', color='blue')
    plt.plot(epochs, history.history['val_loss'], label='Validation Loss', color='red')

    best_val_loss = min(history.history['val_loss'])
    best_epoch_loss = history.history['val_loss'].index(best_val_loss) + 1

    plt.scatter(best_epoch_loss, best_val_loss, color='green', s=50, zorder=5,
                label=f'Best Weights (Epoch {best_epoch_loss})')

    plt.title('Training and Validation Loss per Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Loss (Categorical Crossentropy)')
    plt.legend()
    plt.grid(axis='y', linestyle='--')

    plt.subplot(1, 2, 2)
    plt.plot(epochs, history.history['accuracy'], label='Training Accuracy', color='blue')
    plt.plot(epochs, history.history['val_accuracy'], label='Validation Accuracy', color='red')

    best_val_acc = max(history.history['val_accuracy'])
    best_epoch_acc = history.history['val_accuracy'].index(best_val_acc) + 1

    plt.scatter(best_epoch_acc, best_val_acc, color='green', s=50, zorder=5,
                label=f'Best Accuracy (Epoch {best_epoch_acc})')

    plt.title('Training and Validation Accuracy per Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(axis='y', linestyle='--')

    plt.tight_layout()
    plt.show()

plot_training_history(history)

import matplotlib.pyplot as plt
import numpy as np

model_names = ['LR (Baseline)', 'LightGBM', 'MLP (Deep Learning)']
accuracy = [0.54, 0.79, 0.63]
weighted_f1 = [0.59, 0.77, 0.66]
recall_class_0 = [0.66, 0.17, 0.60]

x = np.arange(len(model_names))
width = 0.25

fig, ax = plt.subplots(figsize=(10, 6))

# Plotting Bars
rects1 = ax.bar(x - width, accuracy, width, label='Akurasi Keseluruhan', color='#4CAF50')
rects2 = ax.bar(x, weighted_f1, width, label='Weighted F1-Score', color='#FFC107')
rects3 = ax.bar(x + width, recall_class_0, width, label='Recall Kelas 0 (Buruk)', color='#F44336')

# Fungsi untuk menambahkan label di atas bar
def autolabel(rects):
    for rect in rects:
        height = rect.get_height()
        ax.annotate(f'{height:.2f}',
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=9)

autolabel(rects1)
autolabel(rects2)
autolabel(rects3)

ax.set_ylabel('Skor Metrik')
ax.set_title('Perbandingan Kinerja Tiga Model pada Test Set')
ax.set_xticks(x)
ax.set_xticklabels(model_names)
ax.legend(loc='lower center', bbox_to_anchor=(0.5, -0.2), ncol=3)
ax.grid(axis='y', linestyle='--')
ax.set_ylim(0, 1.0)

plt.tight_layout()
plt.show()
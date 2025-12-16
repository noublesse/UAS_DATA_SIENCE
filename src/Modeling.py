#@title 6.1 Model 1 — Baseline Model
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
import time
from sklearn.multiclass import OneVsRestClassifier

from sklearn.metrics import classification_report, confusion_matrix, f1_score
class_weights_dict = {
    0: 7.506172839506172,
    1: 0.43517382413087935,
    2: 1.7579512598099958
}
print(f"Menggunakan Bobot Kelas: {class_weights_dict}")
print("-" * 50)
start_time = time.time()

model_baseline = LogisticRegression(
    # Hyperparameter
    C=1.0,
    solver='lbfgs',
    max_iter=1000,
    penalty='l2',
    random_state=42,
    # Menerapkan Class Weights (Kritis untuk Imbalance Data)
    class_weight=class_weights_dict
)

model_baseline.fit(X_train, y_train)

end_time = time.time()
training_time_baseline = end_time - start_time
print(f"Pelatihan Selesai. Waktu Training: {training_time_baseline:.4f} detik")
# --- 2. PREDIKSI & EVALUASI ---
y_pred_baseline = model_baseline.predict(X_test)

# Laporan Klasifikasi
print(classification_report(y_test, y_pred_baseline, target_names=['0: Buruk', '1: Normal', '2: Baik']))

print(confusion_matrix(y_test, y_pred_baseline))
#@title 6.2 Model 2 — ML / Advanced Model
import lightgbm as lgb
import time
from sklearn.metrics import classification_report, confusion_matrix

start_time = time.time()

model_advanced = lgb.LGBMClassifier(
    objective='multiclass',
    metric='multi_logloss',
    n_estimators=500,
    learning_rate=0.05,
    num_leaves=31,
    random_state=42,
    n_jobs=-1,

    # is_unbalance=True akan menyesuaikan bobot secara internal
    is_unbalance=True,

)

# LightGBM secara default tidak memerlukan X_scaled, saya menggunakannya
model_advanced.fit(X_train, y_train)

end_time = time.time()
training_time_advanced = end_time - start_time
print(f"Pelatihan Selesai. Waktu Training: {training_time_advanced:.4f} detik")

# PREDIKSI & EVALUASI
y_pred_advanced = model_advanced.predict(X_test)

# Laporan Klasifikasi
print(classification_report(y_test, y_pred_advanced, target_names=['0: Buruk', '1: Normal', '2: Baik']))
print(confusion_matrix(y_test, y_pred_advanced))
#@title 6.3 Model 3 — Deep Learning Model
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.utils import to_categorical
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, accuracy_score
num_classes = 3
y_train_ohe = to_categorical(y_train, num_classes=num_classes)
y_test_ohe = to_categorical(y_test, num_classes=num_classes)
# CLASS WEIGHTS
class_weights = compute_class_weight(
    class_weight='balanced',
    classes=np.unique(y_train),
    y=y_train
)
class_weight_dict = dict(enumerate(class_weights))
print(f"Menggunakan Bobot Kelas Agresif: {class_weight_dict}")
print("-" * 50)
#  BUILD MLP MODEL
input_dim = X_train.shape[1]

model_dl = Sequential([
    Dense(128, activation='relu', input_shape=(input_dim,)),
    Dropout(0.3),

    Dense(64, activation='relu'),
    Dropout(0.3),

    Dense(num_classes, activation='softmax')
])
# --- 4. COMPILE ---
model_dl.compile(
    optimizer=Adam(learning_rate=0.001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# --- 5. TRAINING ---
early_stop = EarlyStopping(
    monitor='val_loss',
    patience=10,
    restore_best_weights=True
)

print("Memulai pelatihan model_dl (MLP 128-64 units) ")
start_time = time.time()
history = model_dl.fit(
    X_train,
    y_train_ohe,
    validation_split=0.2,
    epochs=100,
    batch_size=32,
    class_weight=class_weight_dict,
    callbacks=[early_stop],
    verbose=1 # Diubah ke 1 agar terlihat progress training
)

end_time = time.time()
training_time_dl_final = end_time - start_time
print(f"Pelatihan Selesai. Waktu Training FINAL: {training_time_dl_final:.4f} detik")
# BUILD MLP MODEL
input_dim = X_train.shape[1]

model_mlp_final = Sequential([
    Dense(128, activation='relu', input_shape=(input_dim,)),
    Dropout(0.3),

    Dense(64, activation='relu'),
    Dropout(0.3),

    Dense(num_classes, activation='softmax')
])
# EVALUASI DAN LAPORAN AKHIR
y_pred_probs_final = model_dl.predict(X_test, verbose=0)
y_pred_final = np.argmax(y_pred_probs_final, axis=1)

print("\n CLASSIFICATION REPORT ")
print(classification_report(
    y_test,
    y_pred_final,
    target_names=['0: Buruk', '1: Normal', '2: Baik']
))
print(f"\nTest Set Accuracy: {accuracy_score(y_test, y_pred_final):.4f}")
print(confusion_matrix(y_test, y_pred_final))
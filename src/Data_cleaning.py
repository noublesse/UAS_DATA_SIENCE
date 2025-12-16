#@title 5.1 Data Cleaning

#Menghapus Duplikat (Removing Duplicates)
n_duplicates = wine_quality.duplicated().sum()

if n_duplicates > 0:
    wine_quality_clean = wine_quality.copy()
    wine_quality_clean.drop_duplicates(inplace=True)
else:
    wine_quality_clean = wine_quality.copy()
    print("1. Tidak ada data duplikat untuk dihapus.")
    print(f"2. Pengecekan Missing Values: {wine_quality_clean.isnull().sum().sum()} (Tidak Ada)")
duplicates_after_cleaning = wine_quality_clean.duplicated().sum()
print(f"Jumlah duplikat (Setelah Cleaning): {duplicates_after_cleaning} baris.")
#@title 5.2: feature engineering (Creating New Features)

# Tujuan: Mengubah masalah 7-kelas (3-9) menjadi 3-kelas Klasifikasi Multi-kelas
# Definisi Kelas: 0 (Buruk: 0-4), 1 (Normal: 5-6), 2 (Baik: 7-10)
bins = [0, 4, 6, 10]
labels = [0, 1, 2]

wine_quality_clean['quality_class'] = pd.cut(
    wine_quality_clean['quality'],
    bins=bins,
    labels=labels,
    include_lowest=True,
    right=True
).astype(int)

# Menghapus variabel target original dari fitur input.
X_raw = wine_quality_clean.drop(['quality', 'quality_class'], axis=1)
y = wine_quality_clean['quality_class']
print(f"\n   X_raw (Fitur Input) shape: {X_raw.shape}")
print(f"   y (Target Klasifikasi) shape: {y.shape}")
print(wine_quality_clean.head())
#data transformasi
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils import class_weight
import tensorflow as tf
from tensorflow.keras.utils import to_categorical

#data tabular
# 1. Inisialisasi StandardScaler
scaler = StandardScaler()
# 2. Terapkan Scaling pada seluruh Himpunan Fitur X_raw
# Fitur wine_type (0/1) juga di-scale.
X_scaled_array = scaler.fit_transform(X_raw)
# 3. Konversi kembali ke DataFrame untuk kemudahan inspeksi
feature_names = X_raw.columns.tolist()
X_scaled = pd.DataFrame(X_scaled_array, columns=feature_names, index=X_raw.index)

print("1. Standardization (StandardScaler) Selesai Diterapkan pada seluruh X_raw.")
print(f"   X_scaled shape: {X_scaled.shape}")
print(f"   Contoh 5 Baris Pertama X_scaled:")
print(X_scaled.head())
from sklearn.model_selection import train_test_split
# --- 1. DATA SPLITTING (5.4) ---
# Membagi X_scaled dan y menjadi 80% train dan 20% test
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y  # Kritis: Mempertahankan rasio kelas
)
print("1. Data Splitting (80:20 Stratified Split) Selesai.")
print(f"   X_train shape: {X_train.shape} | Sampel Training: {X_train.shape[0]}")
print(f"   X_test shape: {X_test.shape} | Sampel Testing: {X_test.shape[0]}")
# DATA BALANCING
# Menghitung bobot kelas berdasarkan distribusi y_train
weights = class_weight.compute_class_weight(
    class_weight='balanced',
    classes=np.unique(y_train),
    y=y_train
)
class_weights_dict = dict(zip(np.unique(y_train), weights))

print("\n2. Class Weights Dihitung (untuk Model LR dan MLP).")
print(f"   Bobot Kelas (0:Buruk, 1:Normal, 2:Baik): {class_weights_dict}")
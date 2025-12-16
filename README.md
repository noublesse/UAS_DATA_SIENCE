# 📘 Judul Proyek
*(Isi judul proyek Anda di sini)*

## 👤 Informasi
- **Nama:** [Brandewa Pandu Asmara]  
- **Repo:** [https://github.com/noublesse/UAS_DATA_SIENCE.git]
- **Video:** [https://youtu.be/UJlUed-Hg5c?si=vg79_RYdhVW7FwCh]  

---

# 1. 🎯 Ringkasan Proyek
Domain: Industri Minuman (Kontrol Kualitas Anggur).

Data Preparation: Penggabungan dataset Red & White Wine, penanganan missing values, outlier capping, dan class weighting.

Model: Membangun 3 model yaitu Logistic Regression (Baseline), LightGBM (Advanced), dan Multi-Layer Perceptron (Deep Learning).

Hasil: Menangani trade-off antara Akurasi Global dan deteksi kelas minoritas (Recall).

---

# 2. 📄 Problem & Goals
**Problem Statements:**  
Problem Statements: - Penggabungan dataset anggur merah dan putih meningkatkan heterogenitas data, sehingga sulit menentukan batas kualitas secara linear.
Ketidakseimbangan kelas yang ekstrem (Kelas 0/Buruk hanya 4.4%) menyebabkan model cenderung mengabaikan sampel kualitas rendah yang krusial bagi kontrol kualitas.

**Goals:**  
Mengembangkan model yang mampu mencapai akurasi tinggi secara keseluruhan menggunakan algoritma boosting.
Memastikan model memiliki Recall yang kuat pada kelas "Buruk" (Kelas 0) untuk meminimalkan risiko produk cacat lolos ke pasar.  
Membangun model Machine Learning (termasuk Deep Learning) untuk klasifikasi kualitas anggur dengan target F1-Score rata-rata (weighted average) melebihi 0.75.

---
## 📁 Struktur Folder
```
project/
│
├── data/                   # Dataset (tidak di-commit, download manual)
│
├── notebooks/              # Jupyter notebooks
│   └── ML_Project.ipynb
│
├── src/                    # Source code
│   
├── models/                 # Saved models
│   ├── model_baseline.pkl
│   ├── model_rf.pkl
│   └── model_cnn.h5
│
├── images/                 # Visualizations
│   └── r
│
├── requirements.txt        # Dependencies
├── .gitignore
└── README.md
```
---

# 3. 📊 Dataset
- **Sumber:** [https://archive.ics.uci.edu/dataset/186/wine+quality]  
- **Jumlah Data:** [6,497 sampel (setelah penggabungan).]  
- **Tipe:** [Tabular]  

### Fitur Utama
Fitur,Deskripsi
volatile acidity,Kadar asam volatil (terkait dengan rasa masam/busuk).
alcohol,Persentase kandungan alkohol.
sulphates,Kadar aditif sulfat.
wine_type,Jenis anggur (Red/White).
quality,"Target: 0 (Buruk), 1 (Normal), 2 (Baik)."

---

# 4. 🔧 Data Preparation
Cleaning: Penanganan missing values dan capping outliers menggunakan metode IQR.

Transformasi: Target quality diubah menjadi 3 kelas (quality_class: 0, 1, 2). 2. Fitur wine_type: Menambahkan fitur biner (0=Merah, 1=Putih).
Untuk Mengatasi ketidakseimbangan kelas. Dan  memungkinkan model membedakan pola fisikokimia antara dua jenis anggur.
Disini saya Menggunakan pd.cut() dengan bins [0, 4, 6, 10]. Lalu Kolom wine_type dibuat saat integrasi data.

Splitting: Pembagian data menjadi 80% Train dan 20% Test (dengan Validation set untuk MLP).

Imbalance Handling: Penghitungan Class Weights secara otomatis untuk menyeimbangkan perhatian model pada kelas minoritas.

---

# 5. 🤖 Modeling
- **Model 1 – Baseline:** [aseline: Logistic Regression dengan class_weight='balanced'. Berfungsi sebagai pembanding performa linier.]  
- **Model 2 – Advanced ML:** [LightGBM (Gradient Boosting). Fokus pada optimalisasi Akurasi Global melalui pembelajaran non-linier berbasis pohon.]  
- **Model 3 – Deep Learning:** [Multi-Layer Perceptron (MLP). Arsitektur 128-64 units dengan Dropout dan Early Stopping untuk menangkap pola kompleks.]  

---

# 6. 🧪 Evaluation
**Metrik:** Accuracy / F1 / MAE / MSE (pilih sesuai tugas)

### Hasil Singkat
Model,Accuracy,Recall (Cls 0),Catatan
Baseline (LR),0.54,0.66,Terbaik untuk deteksi sampel Buruk.
Advanced (LGBM),0.79,0.17,"Akurasi tertinggi, tapi gagal di kelas minoritas."
Deep Learning,0.63,0.60,Keseimbangan antara akurasi dan recall.

---

# 7. 🏁 Kesimpulan
Model terbaik: Logistic Regression (untuk QC) atau LightGBM (untuk Akurasi).
Alasan: LR memberikan keamanan tertinggi bagi produsen karena mampu mendeteksi 66% anggur berkualitas buruk, meskipun akurasi totalnya rendah.
Insight penting: Penggabungan dataset merah dan putih meningkatkan kompleksitas (heterogenitas), sehingga klasifikasi multi-kelas memerlukan trade-off antara efisiensi global dan sensitivitas kelas minoritas.

---

# 8. 🔮 Future Work
- [x] Tambah data  
- [x] Tuning model  
- [x] Coba arsitektur DL lain  
- [ ] Deployment  

---

# 9. 🔁 Reproducibility
Gunakan environment:
Library Utama: scikit-learn==1.3.0, lightgbm==4.0.0, tensorflow==2.14.0.

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
## 📁 Struktur Direktori Proyek

```text
project/
│
├── data/
│   ├── wine+quality/              # Folder sumber dataset
│   ├── winequality-red.csv        # Dataset anggur merah
│   ├── winequality-white.csv      # Dataset anggur putih
│   ├── winequality.names          # Dokumentasi dataset
│   └── .gitkeep
│
├── images/                        # Hasil visualisasi dan plot
│   ├── Bar plot.png
│   ├── Boxplot.png
│   ├── Headmap.png
│   ├── Time series.png
│   ├── plotting bars.png
│   └── .gitkeep
│
├── models/                        # File model yang sudah dilatih (Saved Models)
│   ├── model_baseline_lr.pkl      # Logistic Regression (Model 1)
│   ├── model_advanced_lgbm.pkl    # LightGBM (Model 2)
│   ├── model_deep_learning_mlp.h5 # Deep Learning MLP (Model 3)
│   └── .gitkeep
│
├── notebooks/                     # File utama eksperimen
│   ├── uas.ipynb                  # Jupyter Notebook utama
│   └── .gitkeep
│
├── src/                           # Modul kode Python (Script)
│   ├── Data_cleaning.py           # Script pembersihan data
│   ├── Informasi_dataset.py       # Script statistik deskriptif
│   ├── Modeling.py                # Script algoritma model
│   ├── visualisasi.py             # Script fungsi plotting
│   └── .gitkeep
│
├── .gitignore                     
├── Cheklist Submit.md             
├── LICENSE                        
├── Laporan Proyek Machine Learning.md  
├── Laporan_DataScience.pdf       
└── requirements.txt              
```
---

# 3. 📊 Dataset
- **Sumber:** [https://archive.ics.uci.edu/dataset/186/wine+quality]  
- **Jumlah Data:** [6,497 sampel (setelah penggabungan).]  
- **Tipe:** [Tabular]  

### Fitur Utama
### 📊 Statistik Deskriptif Dataset

| Nama Fitur | Tipe Data | Deskripsi | Count | Mean (μ) | Std (σ) | Min | Max |
|:---|:---:|:---|:---:|:---:|:---:|:---:|:---:|
| **fixed acidity** | Float | Asam Tetap | 6497 | 7.215 | 1.296 | 3.8 | 15.9 |
| **volatile acidity** | Float | Asam Volatil | 6497 | 0.339 | 0.165 | 0.08 | 1.58 |
| **citric acid** | Float | Asam Sitrat | 6497 | 0.319 | 0.145 | 0.00 | 1.66 |
| **residual sugar** | Float | Sisa Gula | 6497 | 5.443 | 4.758 | 0.6 | 65.8 |
| **chlorides** | Float | Klorida | 6497 | 0.056 | 0.035 | 0.009 | 0.611 |
| **free sulfur dioxide** | Float | SO2 Bebas (Anti-oksidan) | 6497 | 30.525 | 17.769 | 1.0 | 289.0 |
| **total sulfur dioxide** | Float | SO2 Total (Stabilitas Anggur) | 6497 | 115.744 | 56.521 | 6.0 | 440.0 |
| **density** | Float | Kepadatan (Gula & Alkohol) | 6497 | 0.99469 | 0.003 | 0.98711 | 1.03898 |
| **pH** | Float | Tingkat Keasaman | 6497 | 3.218 | 0.161 | 2.72 | 4.01 |
| **sulphates** | Float | Sulfat | 6497 | 0.531 | 0.149 | 0.22 | 2.00 |
| **alcohol** | Float | Kadar Alkohol (%) | 6497 | 10.492 | 1.192 | 8.0 | 14.9 |
| **quality (TARGET)** | Integer | Skor Kualitas (Skala 3-9) | 6497 | 5.818 | 0.873 | 3 | 9 |
| **wine_type** | Biner | Jenis (Putih=0, Merah=1) | 6497 | 0.753 | 0.431 | 0 | 1 |
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
numpy==1.24.3
pandas==2.0.3
scikit-learn==1.3.0
matplotlib==3.7.2
seaborn==0.12.2

# Deep Learning Framework (pilih salah satu)
tensorflow==2.14.0  # atau
torch==2.1.0        # PyTorch

# Additional libraries (sesuaikan)
xgboost==1.7.6
lightgbm==4.0.0
opencv-python==4.8.0  # untuk computer vision
nltk==3.8.1           # untuk NLP
transformers==4.30.0  # untuk BERT, dll



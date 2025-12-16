#@title Informasi Dataset

import pandas as pd
import numpy as np

# Koreksi Pemuatan Data dengan Pemisah (sep=';') agar rapi
wine_red = pd.read_csv(
    '/content/drive/MyDrive/wine quality/winequality-red.csv',
    sep=';'
)
wine_white = pd.read_csv(
    '/content/drive/MyDrive/wine quality/winequality-white.csv',
    sep=';'
)

print(f"Data Anggur Merah (wine_red) dimuat: {wine_red.shape[0]} baris.")
print(f"Data Anggur Putih (wine_white) dimuat: {wine_white.shape[0]} baris.")
print(f"\nTotal data gabungan (wine_quality): {wine_quality.shape[0]} baris, {wine_quality.shape[1]} fitur.")

wine_quality = pd.concat([wine_red, wine_white], ignore_index=True)

wine_quality.info()

#kondisi data
#@title kondisi data
print("\n--- 1. Pengecekan Missing Values ---")
# Menghitung jumlah nilai NULL/NA per kolom
print(wine_quality.isnull().sum().sort_values(ascending=False))

# --- 2. Duplicate Data ---
n_duplicates = wine_quality.duplicated().sum()
print(f"\n--- 2. Pengecekan Duplicate Data ---")
print(f"Jumlah baris duplikat yang ditemukan: {n_duplicates} baris.")
# --- 3. Imbalanced Data (Distribusi Kelas Target) ---
print("\n--- 3. Pengecekan Imbalanced Data (Distribusi Skor Kualitas Original) ---")
# Menghitung frekuensi setiap skor kualitas (0-10)
target_counts = wine_quality['quality'].value_counts().sort_index()
print(target_counts)
# --- 4. Indikasi Outliers dan Noise ---
print("\n--- 4. Indikasi Outliers (Melalui Perbandingan Mean, Min, dan Max) ---")
print("Perhatikan Jangkauan Nilai (Min vs Max) dibandingkan Mean:")
print(wine_quality[['residual sugar', 'chlorides', 'total sulfur dioxide', 'alcohol']].describe().loc[['mean', 'min', 'max']])
#Exploratory Data Analysis (EDA) 
#@title Exploratory Data Analysis (EDA)

import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style("whitegrid")
# --- Visualisasi 1: Class Distribution Plot ---
plt.figure(figsize=(9, 6))
# Menggunakan warna tunggal (misalnya 'teal') untuk menghindari warning
sns.countplot(x='quality', data=wine_quality, color='teal')
plt.title('Visualisasi 1: Distribusi Frekuensi Skor Kualitas Anggur (0-10)', fontsize=14)
plt.xlabel('Skor Kualitas (quality)', fontsize=12)
plt.ylabel('Jumlah Sampel (Frekuensi)', fontsize=12)
plt.grid(axis='y', linestyle='--')
plt.show()
# --- Visualisasi 2: Correlation Heatmap ---
plt.figure(figsize=(10, 8))
corr_matrix = wine_quality.corr()
# Fokus pada 10 fitur yang memiliki korelasi tertinggi ke target 'quality'
top_corr_features = corr_matrix.index[abs(corr_matrix["quality"]).argsort()[-10:]]
sns.heatmap(wine_quality[top_corr_features].corr(),
            annot=True,
            cmap='coolwarm',
            fmt=".2f",
            linewidths=.5,
            cbar_kws={'label': 'Koefisien Korelasi'})
plt.title('Visualisasi 2: Heatmap Korelasi Antar Fitur dengan Korelasi Tertinggi ke Target', fontsize=14)
plt.yticks(rotation=0)
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig('eda_vis_2_correlation_heatmap.png')
plt.show()
# --- Visualisasi 3: Boxplot of Alcohol Content vs. Quality
plt.figure(figsize=(9, 6))
sns.boxplot(
    x='quality',
    y='alcohol',
    data=wine_quality,
    hue='quality',            # Variabel yang digunakan untuk membedakan warna
    palette='Set3',
    legend=False              # Menonaktifkan legend karena hue sama dengan x
)
plt.title('Visualisasi 3: Distribusi Kadar Alkohol Berdasarkan Skor Kualitas', fontsize=14)
plt.xlabel('Skor Kualitas (quality)', fontsize=12)
plt.ylabel('Kadar Alkohol (alcohol)', fontsize=12)
plt.grid(axis='y', linestyle='--')
plt.show()
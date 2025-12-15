# 📘 Judul Proyek
Klasifikasi Otomatis Abstrak Jurnal Ilmiah Menggunakan LSTM dan Support Vector Machine

## 👤 Informasi
- **Nama:** Adam Mahabayu Muhibbulloh  
- **Repo:** https://github.com/adammahabayu/UAS_Praktik_Teknik_2025_TRPL-5A_Data-Science 
- **Video:** [...]  

---

# 1. 🎯 Ringkasan Proyek
- Menyelesaikan permasalahan sesuai domain  
- Melakukan data preparation  
- Membangun 3 model: **Baseline**, **Advanced**, **Deep Learning**  
- Melakukan evaluasi dan menentukan model terbaik  

---

# 2. 📄 Problem & Goals
**Problem Statements:**  
- Proses klasifikasi jurnal secara manual memakan waktu lama dan tidak efisien seiring bertambahnya volume publikasi.
- Diperlukan sistem yang dapat membedakan topik jurnal yang memiliki istilah teknis serupa.

**Goals:**  
- Membangun pipeline *text preprocessing* yang efektif untuk data abstrak ilmiah.
- Mengembangkan model klasifikasi dengan akurasi minimal **80%**.
- Membandingkan performa antara model probabilistik sederhana (Naive Bayes), model linear margin (SVM), dan model sekuensial (LSTM).

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
- **Sumber:** https://archive.ics.uci.edu
- **Jumlah Data:** Jumlah baris (rows): 1138 & Jumlah kolom (columns/features): 3.
- **Tipe:** Text

**Fitur Utama:**
- `Abstract`: Berisi teks ringkasan jurnal ilmiah.
- `Label`: Kategori/Topik dari jurnal tersebut (Target Variable).
- **Nama File:** `classified_abstracts.json`
- **Sumber Data:** https://archive.ics.uci.edu
- **Jumlah baris (rows):** 1138
- **Jumlah kolom (columns/features):** 3.
- **Tipe data:** Text
- **Ukuran dataset:** 1.3122 MB
- **Format file:** JSON

---

# 4. 🔧 Data Preparation
1.  **Text Cleaning:**
    - *Lowercasing* (huruf kecil).
    - Menghapus tanda baca (regex) dan angka.
    - Menghapus *stopwords* (kata umum seperti "the", "and") menggunakan NLTK.
2.  **Encoding:** Mengubah label kategori menjadi angka (0, 1, ...).
3.  **Splitting:** Membagi data menjadi **80% Train** dan **20% Test** (Stratified).
4.  **Vectorization:**
    - **TF-IDF:** Digunakan untuk Naive Bayes dan SVM (Max features: 5000).
    - **Tokenization & Padding:** Digunakan untuk LSTM (Max sequence length: 200 kata). 

---

# 5. 🤖 Modeling
- **Model 1 – Baseline:** Multinomial Naive Bayes  
- **Model 2 – Advanced ML:** Support Vector Machine (SVM)
- **Model 3 – Deep Learning:** Long Short-Term Memory (LSTM)

---

# 6. 🧪 Evaluation
**Metrik:** Accuracy / F1 / MAE / MSE (pilih sesuai tugas)

### Hasil Singkat
| Model | Accuracy | F1-Score | Keterangan |
|-------|----------|----------|------------|
| Naive Bayes | 81% | 0.74 | *Underperform* pada kelas minoritas. |
| **SVM (Linear)** | **93%** | **0.93** | **Model Terbaik (Best Model)**. |
| LSTM (Deep Learning) | 87% | 0.87 | Performa baik, namun kalah efisien dari SVM. |

---

# 7. 🏁 Kesimpulan
- Model terbaik: SVM  
- Alasan: Karena kemampuannya menangani data teks berdimensi tinggi dengan jumlah sampel yang terbatas (*small dataset*).
- Insight penting:
    1.  **Kompleksitas ≠ Performa:** Model Deep Learning (LSTM) yang kompleks dan berat tidak selalu mengalahkan model Machine Learning klasik (SVM) jika datanya tidak cukup besar.
    2.  **Jebakan Akurasi:** Model Naive Bayes mengajarkan kita untuk berhati-hati dengan metrik akurasi pada *imbalanced data*. Meskipun akurasinya 81%, ia gagal mengenali kelas minoritas (Recall rendah).
    3.  **Efisiensi:** SVM memberikan keseimbangan terbaik (Sweet Spot) antara waktu training yang cepat (< 10 detik) dan akurasi yang tinggi, menjadikannya pilihan paling efisien untuk *deployment*.

---

# 8. 🔮 Future Work
- [x] Tambah data  
- [x] Tuning model  
- [x] Coba arsitektur DL lain  
- [ ] Deployment  

---

# 9. 🔁 Reproducibility
Gunakan environment: Google Colab

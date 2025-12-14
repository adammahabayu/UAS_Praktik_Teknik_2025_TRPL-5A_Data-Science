# Laporan Proyek Machine Learning

## INFORMASI PROYEK

**Judul Proyek:** Klasifikasi Otomatis Abstrak Jurnal Ilmiah Menggunakan LSTM dan Support Vector Machine

**Nama Mahasiswa:** Adam Mahabayu Muhibbulloh  
**NIM:** 234311002 
**Program Studi:** D4 Teknologi Rekayasa Perangkat Lunak 
**Mata Kuliah:** Data Science
**Dosen Pengampu:** Gus Nanang Syaifuddiin  
**Tahun Akademik:** 2024/2025  
**Link GitHub Repository:** [Isi Link Repository GitHub Anda Disini]  
**Link Video Pembahasan:** [Isi Link Video YouTube Anda Disini]  

---

## 1. LEARNING OUTCOMES
Pada proyek ini, mahasiswa diharapkan dapat:
1. Memahami konteks masalah klasifikasi teks tidak terstruktur.
2. Melakukan analisis dan eksplorasi data (EDA) secara komprehensif pada data teks.
3. Melakukan data preparation (Text Preprocessing) seperti cleaning, tokenization, dan vectorization.
4. Mengembangkan tiga model machine learning yang terdiri dari:
   - **Model Baseline:** Naive Bayes (MultinomialNB)
   - **Model Machine Learning:** Support Vector Machine (SVM)
   - **Model Deep Learning:** Long Short-Term Memory (LSTM)
5. Menggunakan metrik evaluasi yang relevan (Accuracy & Classification Report).
6. Melaporkan hasil eksperimen secara ilmiah dan sistematis.
7. Mengunggah seluruh kode proyek ke GitHub.
8. Menerapkan prinsip software engineering (penyimpanan model/deployment ready).

---

## 2. PROJECT OVERVIEW

### 2.1 Latar Belakang
Pertumbuhan publikasi ilmiah yang sangat pesat menyebabkan membludaknya jumlah artikel jurnal baru setiap harinya. Hal ini menyulitkan peneliti dan pengelola basis data untuk memilah artikel yang relevan ke dalam kategori topik tertentu secara manual. Pengelompokan manual memakan waktu lama dan rentan terhadap inkonsistensi (*human error*).

Oleh karena itu, diperlukan sistem otomatis berbasis *Machine Learning* dan *Deep Learning* yang dapat membaca abstrak jurnal dan memprediksi kategori topiknya. Proyek ini membandingkan kinerja metode tradisional (Naive Bayes & SVM) dengan metode jaringan saraf tiruan (LSTM) untuk menyelesaikan masalah klasifikasi teks tersebut.

---

## 3. BUSINESS UNDERSTANDING / PROBLEM UNDERSTANDING

### 3.1 Problem Statements
1. Bagaimana cara mengolah data teks abstrak mentah yang mengandung banyak *noise* (tanda baca, stopword) agar siap dimodelkan?
2. Apakah model *Deep Learning* (LSTM) mampu memberikan akurasi yang lebih baik dibandingkan model konvensional (Naive Bayes & SVM) pada dataset ini?
3. Bagaimana performa model dalam mengklasifikasikan berbagai topik jurnal yang berbeda?

### 3.2 Goals
1. Membangun pipa *preprocessing* teks yang efektif (cleaning, tokenizing, padding).
2. Melatih dan mengevaluasi tiga jenis model yang berbeda (Baseline, ML, DL).
3. Mencapai akurasi klasifikasi minimal 80% pada data uji (*test set*).
4. Menyimpan model yang telah dilatih agar dapat digunakan kembali (*reproducible*).

### 3.3 Solution Approach
- **Model 1 (Baseline):** Menggunakan **Multinomial Naive Bayes** dengan fitur TF-IDF. Dipilih karena kecepatan komputasinya dan kinerjanya yang menjadi standar dasar klasifikasi teks.
- **Model 2 (Advanced ML):** Menggunakan **Support Vector Machine (SVM)** dengan kernel Linear. Dipilih karena kemampuannya menangani ruang fitur berdimensi tinggi (hasil vektorisasi teks).
- **Model 3 (Deep Learning):** Menggunakan **LSTM (Long Short-Term Memory)**. Dipilih karena kemampuannya memahami urutan kata (*sequence*) dan konteks kalimat yang panjang dalam sebuah abstrak.

---

## 4. DATA UNDERSTANDING

### 4.1 Informasi Dataset
- **Nama File:** `classified_abstracts.json`
- **Sumber Data:** Dataset internal proyek / Uploaded File.
- **Tipe Data:** Teks Tidak Terstruktur (Unstructured Text).
- **Fitur Utama:**
  - `Abstract`: Berisi teks ringkasan jurnal ilmiah.
  - `Label`: Kategori/Topik dari jurnal tersebut (Target Variable).

### 4.2 Exploratory Data Analysis (EDA)
Berikut adalah hasil visualisasi minimal 3 aspek dari data:

#### **Visualisasi 1: Distribusi Label Kelas**
> *[Masukkan Screenshot Gambar Countplot/Bar Chart Distribusi Label dari Notebook Disini]*
<img width="705" height="402" alt="Visualisasi 1" src="https://github.com/user-attachments/assets/4c4ec48a-856c-4052-b8b5-cc80ec1b63fa" />


**Analisis:** Grafik ini menunjukkan jumlah dokumen untuk setiap kategori. Hal ini penting untuk mengetahui apakah dataset bersifat *imbalanced* (timpang) atau *balanced*. Jika timpang, kita perlu menggunakan metrik evaluasi selain akurasi (seperti F1-Score).

#### **Visualisasi 2: Distribusi Panjang Kata (Word Count)**
> *[Masukkan Screenshot Gambar Histogram Panjang Kata dari Notebook Disini]*
<img width="860" height="479" alt="Visualisasi 2" src="https://github.com/user-attachments/assets/ef8903ef-e7e3-4ab1-ac04-f1acbd51384e" />


**Analisis:** Histogram ini menunjukkan sebaran jumlah kata dalam abstrak. Informasi ini digunakan untuk menentukan parameter `MAX_SEQUENCE_LENGTH` pada model Deep Learning (LSTM), agar padding tidak terlalu panjang atau memotong informasi penting.

#### **Visualisasi 3: Word Cloud**
> *[Masukkan Screenshot Gambar WordCloud dari Notebook Disini]*
<img width="790" height="427" alt="Visualisasi 3" src="https://github.com/user-attachments/assets/3dd4e116-bd40-443b-81ed-fc1f58c1b72f" />


**Analisis:** Word Cloud menampilkan kata-kata yang paling sering muncul di seluruh korpus data. Kata yang berukuran besar mengindikasikan frekuensi kemunculan yang tinggi, memberikan gambaran umum mengenai topik dominan dalam dataset.

---

## 5. DATA PREPARATION

### 5.1 Data Cleaning
Proses pembersihan teks yang dilakukan meliputi:
1. **Lowercasing:** Mengubah seluruh teks menjadi huruf kecil.
2. **Regex Cleaning:** Menghapus karakter non-huruf (angka dan tanda baca).
3. **Stopwords Removal:** Menghapus kata hubung umum (seperti "the", "and", "is") menggunakan library NLTK.

### 5.2 Encoding & Splitting
- **Label Encoding:** Mengubah kolom `Label` menjadi format numerik (0, 1, 2, dst).
- **Train-Test Split:** Membagi data menjadi 80% Data Latih (*Training*) dan 20% Data Uji (*Testing*) dengan metode *stratified* untuk menjaga proporsi kelas.

### 5.3 Feature Engineering
- **TF-IDF (Untuk Model ML):** Mengubah teks menjadi vektor angka berdasarkan bobot kepentingannya dalam dokumen.
- **Tokenization & Padding (Untuk Model DL):** Mengubah kata menjadi urutan indeks angka dan menyamakan panjang input (padding) agar bisa diproses oleh Neural Network.

---

## 6. MODELING

### 6.1 Model 1: Baseline (Naive Bayes)
Menggunakan algoritma `MultinomialNB` dari Scikit-Learn. Model ini bekerja berdasarkan prinsip probabilitas Bayes dengan asumsi bahwa setiap kata muncul secara independen.
- **Parameter:** Default.
- **Input:** TF-IDF Vectors.

### 6.2 Model 2: Machine Learning (SVM)
Menggunakan algoritma `SVC` (Support Vector Classifier). Model ini mencari *hyperplane* terbaik yang memisahkan kelas data dengan margin terbesar.
- **Kernel:** `Linear` (Efektif untuk teks).
- **Probability:** `True`.
- **Input:** TF-IDF Vectors.

### 6.3 Model 3: Deep Learning (LSTM)
Menggunakan library TensorFlow/Keras dengan arsitektur Sequential.
- **Layer Embedding:** Mengubah indeks kata menjadi vektor padat (*dense vector*).
- **SpatialDropout1D:** Mengurangi overfitting.
- **Layer LSTM:** Menangkap konteks urutan kata (memori jangka panjang/pendek).
- **Dense Layer:** Output layer dengan aktivasi `Softmax` untuk klasifikasi multikelas.
- **Optimizer:** Adam.
- **Loss Function:** Sparse Categorical Crossentropy.

> *[Masukkan Screenshot Grafik Training Accuracy & Loss Model LSTM dari Notebook Disini]*
*(Grafik ini menunjukkan proses belajar model selama epoch berjalan)*

---

## 7. EVALUATION

Evaluasi dilakukan menggunakan metrik **Accuracy** dan **Classification Report** (Precision, Recall, F1-Score) pada data uji (20% data yang tidak dilihat saat training).

### 7.1 Hasil Evaluasi
Berikut adalah ringkasan hasil akurasi dari ketiga model:

| Model | Akurasi | Keterangan |
|-------|---------|------------|
| Naive Bayes | **[Isi Hasil NB]** | Baseline, training sangat cepat. |
| SVM (Linear) | **[Isi Hasil SVM]** | Akurasi tinggi, training lebih lambat. |
| LSTM (Deep Learning) | **[Isi Hasil LSTM]** | Mampu menangkap konteks kompleks. |

> *[Masukkan Screenshot Tabel Classification Report atau Grafik Perbandingan Model Disini]*

### 7.2 Analisis Hasil
Berdasarkan hasil eksperimen:
- **Model Terbaik:** Model **[Sebutkan Model Pemenang, misal: SVM atau LSTM]** memberikan performa terbaik dengan akurasi tertinggi.
- **Perbandingan:** Model Deep Learning (LSTM) dan SVM umumnya memberikan hasil yang lebih baik dibandingkan Baseline (Naive Bayes) karena kemampuannya menangkap hubungan antar kata yang lebih kompleks, meskipun memerlukan waktu komputasi yang lebih lama.

---

## 8. CONCLUSION

1. Proses *preprocessing* (pembersihan, tokenisasi) sangat krusial dalam meningkatkan kualitas data teks.
2. Proyek berhasil mengembangkan tiga model (Naive Bayes, SVM, LSTM) yang berfungsi dengan baik.
3. Model **[Sebutkan Model Terbaik]** terpilih sebagai model terbaik untuk dataset ini.
4. Model telah berhasil disimpan (*saved*) ke dalam format `.pkl` dan `.h5` untuk kebutuhan deployment di masa depan.

---

## 9. FUTURE WORK (Opsional)
- Menambah jumlah dataset untuk variasi yang lebih baik.
- Mencoba Pre-trained Model seperti BERT atau DistilBERT.
- Melakukan Hyperparameter Tuning (GridSearch) untuk optimalisasi model.

---

## 10. REPRODUCIBILITY (WAJIB)

### 10.1 GitHub Repository
Seluruh kode dan aset proyek dapat diakses melalui link berikut:
**[Isi Link Repository GitHub Anda]**

### 10.2 Environment & Dependencies
Proyek ini dikembangkan menggunakan Python dengan library utama:
- `pandas`, `numpy` (Data Manipulation)
- `matplotlib`, `seaborn`, `wordcloud` (Visualization)
- `scikit-learn` (ML Models & Metrics)
- `tensorflow` (Deep Learning)
- `nltk` (Text Processing)



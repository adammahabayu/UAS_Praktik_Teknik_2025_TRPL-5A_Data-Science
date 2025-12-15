# Laporan Proyek Machine Learning

## INFORMASI PROYEK

**Judul Proyek:** Klasifikasi Otomatis Abstrak Jurnal Ilmiah Menggunakan LSTM dan Support Vector Machine

**Nama Mahasiswa:** Adam Mahabayu Muhibbulloh  
**NIM:** 234311002 
**Program Studi:** D4 Teknologi Rekayasa Perangkat Lunak 
**Mata Kuliah:** Data Science
**Dosen Pengampu:** Gus Nanang Syaifuddiin  
**Tahun Akademik:** 2024/2025  
**Link GitHub Repository:** https://github.com/adammahabayu/UAS_Praktik_Teknik_2025_TRPL-5A_Data-Science  
**Link Video Pembahasan:** [Isi Link Video YouTube Anda Disini]  

---

## 1. LEARNING OUTCOMES
Pada proyek ini, mahasiswa diharapkan dapat:
1. Memahami konteks masalah dan merumuskan problem statement secara jelas
2. Melakukan analisis dan eksplorasi data (EDA) secara komprehensif (**OPSIONAL**)
3. Melakukan data preparation yang sesuai dengan karakteristik dataset
4. Mengembangkan tiga model machine learning yang terdiri dari (**WAJIB**):
   - Model baseline
   - Model machine learning / advanced
   - Model deep learning (**WAJIB**)
5. Menggunakan metrik evaluasi yang relevan dengan jenis tugas ML
6. Melaporkan hasil eksperimen secara ilmiah dan sistematis
7. Mengunggah seluruh kode proyek ke GitHub (**WAJIB**)
8. Menerapkan prinsip software engineering dalam pengembangan proyek

---

## 2. PROJECT OVERVIEW

### 2.1 Latar Belakang
Dalam dekade terakhir, volume publikasi ilmiah dan literatur akademik telah mengalami pertumbuhan eksponensial. Ribuan artikel jurnal baru diterbitkan setiap harinya di berbagai disiplin ilmu, menciptakan fenomena yang dikenal sebagai information overload. Bagi peneliti, akademisi, dan pengelola perpustakaan digital, tantangan terbesarnya bukan lagi pada ketersediaan data, melainkan pada bagaimana mengelola, memilah, dan menemukan kembali informasi yang relevan secara efisien.

Secara tradisional, proses pengkategorian atau pengindeksan artikel jurnal dilakukan secara manual oleh pakar manusia. Metode ini memiliki kelemahan signifikan: memakan waktu lama (time-consuming), biaya tinggi, dan rentan terhadap inkonsistensi akibat kelelahan atau subjektivitas manusia (human error). Ketika jumlah dokumen mencapai ribuan hingga jutaan, metode manual menjadi tidak lagi relevan (skalabilitas rendah).

Oleh karena itu, penerapan teknologi Natural Language Processing (NLP) dan Machine Learning menjadi solusi krusial. Proyek ini berfokus pada pengembangan sistem klasifikasi teks otomatis yang dapat memprediksi topik dari sebuah jurnal hanya dengan membaca abstraknya. Abstrak dipilih karena merupakan ringkasan padat yang merepresentasikan inti dari keseluruhan konten dokumen.

Pentingnya proyek ini terletak pada potensinya untuk membantu sistem repositori ilmiah dalam mengotomatisasi pelabelan dokumen. Dengan membandingkan metode Machine Learning klasik (seperti Naive Bayes dan SVM) melawan metode Deep Learning modern (seperti LSTM), penelitian ini juga bertujuan untuk mengevaluasi seberapa efektif model jaringan saraf tiruan dalam menangkap konteks semantik bahasa ilmiah dibandingkan model berbasis frekuensi kata sederhana.

**Referensi Ilmiah:**
- Aggarwal, C. C., & Zhai, C. (2012). Mining Text Data. Springer Science & Business Media. (Menjelaskan fundamental pengolahan data teks).
- Jurafsky, D., & Martin, J. H. (2024). Speech and Language Processing (3rd ed.). Pearson. (Referensi utama dalam bidang NLP dan klasifikasi teks).

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
- **Sumber Data:** https://archive.ics.uci.edu
- **Jumlah baris (rows):** 1138
- **Jumlah kolom (columns/features):** 3.
- **Tipe data:** Text
- **Ukuran dataset:** 1.3122 MB
- **Format file:** JSON
- **Fitur Utama:**
  - `Abstract`: Berisi teks ringkasan jurnal ilmiah.
  - `Label`: Kategori/Topik dari jurnal tersebut (Target Variable).

### 4.2 Exploratory Data Analysis (EDA)
Dataset ini terdiri dari 3 kolom utama yang berisi informasi identitas dokumen, teks konten, dan label kategori. Berikut adalah detail fitur dataset:
<img width="634" height="307" alt="image" src="https://github.com/user-attachments/assets/d58c9647-59f0-4d3b-8329-033dbdf3d549" />

### 4.3 Kondisi Data
Berdasarkan hasil pemeriksaan awal (data understanding), berikut adalah kondisi dataset:

- **Missing Values**: [Tidak ada / Terdapat X data kosong] (Hasil cek: 0 null values pada semua kolom).
- **Duplicate Data:** [Tidak ada / Terdapat X data duplikat]. (Dicek menggunakan df.duplicated()).
- **Outliers:** [Pada data teks, outliers dideteksi berdasarkan panjang kalimat. Terdapat beberapa abstrak yang sangat pendek (< 20 kata) atau sangat panjang (> 500 kata) yang dapat mempengaruhi padding.]
- **Imbalanced Data:** [Berdasarkan visualisasi distribusi kelas, dataset terlihat [Seimbang / Timpang] di mana kategori [X] memiliki jumlah data yang jauh lebih [banyak/sedikit] dibanding kategori lain.]
- **Noise:** [Data mentah (Abstract) masih mengandung banyak noise seperti; Tanda baca (.,?!-) dan karakter spesial, Angka-angka yang tidak relevan dengan topik, Stopwords (kata umum seperti "the", "and", "of") yang tidak membawa makna semantik unik, Variasi penggunaan huruf besar dan kecil (casing).]
- **Data Quality Issues:** [Jelaskan jika ada masalah lain]

### 4.4 Exploratory Data Analysis (EDA) - (**OPSIONAL**)

Berikut adalah hasil visualisasi minimal 3 aspek dari data:

#### **Visualisasi 1: Distribusi Label Kelas**
<img width="705" height="402" alt="Visualisasi 1" src="https://github.com/user-attachments/assets/4c4ec48a-856c-4052-b8b5-cc80ec1b63fa" />
**Analisis:** Grafik ini menunjukkan jumlah dokumen untuk setiap kategori. Hal ini penting untuk mengetahui apakah dataset bersifat *imbalanced* (timpang) atau *balanced*. Jika timpang, kita perlu menggunakan metrik evaluasi selain akurasi (seperti F1-Score).

#### **Visualisasi 2: Distribusi Panjang Kata (Word Count)**
<img width="860" height="479" alt="Visualisasi 2" src="https://github.com/user-attachments/assets/ef8903ef-e7e3-4ab1-ac04-f1acbd51384e" />
**Analisis:** Histogram ini menunjukkan sebaran jumlah kata dalam abstrak. Informasi ini digunakan untuk menentukan parameter `MAX_SEQUENCE_LENGTH` pada model Deep Learning (LSTM), agar padding tidak terlalu panjang atau memotong informasi penting.

#### **Visualisasi 3: Word Cloud**
<img width="790" height="427" alt="Visualisasi 3" src="https://github.com/user-attachments/assets/3dd4e116-bd40-443b-81ed-fc1f58c1b72f" />
**Analisis:** Word Cloud menampilkan kata-kata yang paling sering muncul di seluruh korpus data. Kata yang berukuran besar mengindikasikan frekuensi kemunculan yang tinggi, memberikan gambaran umum mengenai topik dominan dalam dataset.

---

## 5. DATA PREPARATION

### 5.1 Data Cleaning
Proses pembersihan teks yang dilakukan meliputi:
1. **Lowercasing:** Mengubah seluruh teks menjadi huruf kecil.
2. **Regex Cleaning:** Menghapus karakter non-huruf (angka dan tanda baca).
3. **Stopwords Removal:** Menghapus kata hubung umum (seperti "the", "and", "is") menggunakan library NLTK.

### 5.2 Feature Engineering
- **TF-IDF (Untuk Model ML):** Digunakan untuk model Machine Learning konvensional (Naive Bayes & SVM). Teknik ini mengubah teks menjadi vektor angka dengan memberikan bobot lebih tinggi pada kata yang unik/spesifik dan bobot rendah pada kata yang terlalu umum. Saya membatasi fitur (max_features) sebanyak 5.000 kata teratas untuk mengurangi dimensi dan noise.
- **Word Embedding (Dense Vector):** Digunakan untuk model Deep Learning (LSTM). Fitur ini dipelajari secara otomatis oleh Embedding Layer pada Keras, di mana setiap kata direpresentasikan sebagai vektor padat (dense vector) berdimensi 100. Ini memungkinkan model menangkap hubungan semantik antar kata (misalnya, kata "king" dan "queen" akan memiliki nilai vektor yang berdekatan).

### 5.3 Data Transformation
- **Lowercasing:** Mengubah seluruh huruf menjadi kecil (lowercase) agar kata seperti "Data" dan "data" dianggap sebagai entitas yang sama.
- **Regex Cleaning:** Menghapus karakter non-alfabet (angka, tanda baca, simbol) menggunakan Regular Expression ([^a-zA-Z\s]) karena elemen ini dianggap sebagai noise dalam klasifikasi topik abstrak.
- **Stopwords Removal:** Menghapus kata-kata umum yang tidak memiliki makna topik spesifik (seperti "and", "the", "is", "at") menggunakan pustaka NLTK.
- **Tokenization:** Mengubah kalimat menjadi urutan integer (index). Contoh: "machine learning" → [45, 12].
- **Padding Sequences:** Menyamakan panjang seluruh input urutan kata menjadi 200 kata (maxlen=200). Kalimat yang lebih pendek akan diberi nilai 0 (padding), dan kalimat yang lebih panjang akan dipotong (truncated). Ini diperlukan agar data dapat diproses dalam bentuk matriks oleh GPU.
- **Label Encoding:** Mengubah target variabel (Label) yang berbentuk teks kategori menjadi format numerik (0, 1, 2, dst.) agar dapat dikalkulasi oleh fungsi loss model.

## 5.4 Data Splitting
- **Training Set:** 80% (Digunakan untuk melatih bobot model).
- **Test Set:** 20% (Disembunyikan saat training, digunakan murni untuk evaluasi akhir).

# 5.5 Data Balancing (jika diperlukan)
- Teknik yang digunakan: Pada proyek ini, tidak dilakukan teknik resampling eksternal (seperti SMOTE atau Undersampling) secara eksplisit.
# 5.6 
- Data Cleaning (Pembersihan)
     1.  Apa: Menghapus tanda baca, angka, dan stopwords.
     2.  Mengapa: Teks mentah mengandung banyak karakter yang tidak berkontribusi pada penentuan topik (noise). Membersihkannya akan memperkecil ukuran kosakata dan fokus pada kata kunci yang bermakna.
     3.  Bagaimana: Menggunakan Python string manipulation, RegEx, dan NLTK library.
- Encoding (Pengkodean Label)
     1. Apa: Mengubah label kategori menjadi angka.
     2. Mengapa: Algoritma Machine Learning matematis tidak dapat memproses data string secara langsung sebagai target.
     3. Bagaimana: Menggunakan LabelEncoder dari Scikit-Learn.
- Vectorization & Padding (Representasi Numerik)
     1. Apa: Mengubah teks bersih menjadi matriks angka.
     2. Mengapa: Komputer hanya mengerti angka. TF-IDF digunakan untuk ML tradisional (bobot frekuensi), sedangkan Tokenizer+Padding digunakan untuk LSTM (urutan kata) agar input memiliki dimensi yang konsisten.
     3. Bagaimana: Menggunakan TfidfVectorizer (sklearn) dan Tokenizer + pad_sequences (TensorFlow).
- Splitting (Pembagian Data)
     1. Apa: Membagi data menjadi Train dan Test.
     2. Mengapa: Untuk menguji generalisasi model pada data yang belum pernah dilihat sebelumnya (mencegah data leakage).
     3. Bagaimana: Menggunakan train_test_split dengan rasio 80:20.

---

## 6. MODELING
### 6.1 Model 1 — Baseline Model
#### 6.1.1 Deskripsi Model

**Nama Model:** Multinomial Naive Bayes
**Teori Singkat:**  
Naive Bayes adalah algoritma klasifikasi probabilistik yang didasarkan pada Teorema Bayes dengan asumsi "naif" bahwa setiap fitur (kata) bersifat independen satu sama lain. Varian Multinomial secara khusus dirancang untuk menangani data diskrit berupa hitungan kata atau frekuensi (TF-IDF), sehingga sangat populer untuk klasifikasi teks.
**Alasan Pemilihan:**  
Model ini dipilih sebagai baseline karena komputasinya sangat cepat, sederhana, dan tidak memerlukan banyak penyetelan parameter (tuning). Jika model yang lebih kompleks (seperti Deep Learning) tidak dapat mengalahkan performa Naive Bayes secara signifikan, maka penggunaan model kompleks tersebut menjadi tidak efisien.

#### 6.1.2 Hyperparameter
**Parameter yang digunakan:**
- alpha: 1.0 (Laplace smoothing default)
- fit_prior: True
- class_prior: None

#### 6.1.3 Implementasi (Ringkas)
from sklearn.naive_bayes import MultinomialNB

# Inisialisasi model
nb_model = MultinomialNB()

# Training
nb_model.fit(X_train_tfidf, y_train)

# Prediksi
y_pred_nb = nb_model.predict(X_test_tfidf)

model_baseline = LogisticRegression(C=1.0, max_iter=100)
model_baseline.fit(X_train, y_train)
y_pred_baseline = model_baseline.predict(X_test)
```

#### 6.1.4 Hasil Awal
Model berhasil dilatih dengan sangat cepat (< 1 detik). Akurasi awal pada data uji akan dibandingkan lebih lanjut di Section 7.

---

### 6.2 Model 2 — ML / Advanced Model
#### 6.2.1 Deskripsi Model

**Nama Model:** Support Vector Machine (SVM) Teori Singkat:
**Teori Singkat:**  
SVM bekerja dengan mencari hyperplane (bidang pemisah) terbaik yang memaksimalkan margin (jarak) antara kelas data yang berbeda.

**Alasan Pemilihan:**  
SVM sangat efektif bekerja pada ruang berdimensi tinggi, seperti hasil vektorisasi teks (TF-IDF) yang memiliki ribuan fitur. SVM dikenal memiliki kemampuan generalisasi yang baik untuk data teks.

**Keunggulan:**
- Efektif pada dimensi tinggi (banyak fitur kata).
- Hemat memori karena hanya menggunakan subset poin pelatihan (support vectors).

**Kelemahan:**
- Waktu training bisa lambat jika dataset sangat besar (>100.000 sampel).
- Sensitif terhadap noise jika kelas tumpang tindih.

#### 6.2.2 Hyperparameter
- kernel: 'linear' (Kernel linear biasanya terbaik untuk teks)
- probability: True (Agar bisa menghitung confidence score)
- C: 1.0 (Regularization parameter default)
- gamma: 'scale'

#### 6.2.3 Implementasi (Ringkas)
from sklearn.svm import SVC

# Inisialisasi Model SVM dengan kernel Linear
svm_model = SVC(kernel='linear', probability=True)

# Training
svm_model.fit(X_train_tfidf, y_train)

# Prediksi
y_pred_svm = svm_model.predict(X_test_tfidf)

#### 6.2.4 Hasil Model
Model SVM membutuhkan waktu training sedikit lebih lama dibandingkan Naive Bayes, namun diharapkan memberikan batas keputusan (decision boundary) yang lebih tegas antar topik.

---

### 6.3 Model 3 — Deep Learning Model (WAJIB)

#### 6.3.1 Deskripsi Model

**Nama Model:** Long Short-Term Memory (LSTM)

** (Centang) Jenis Deep Learning: **
- [ ] Multilayer Perceptron (MLP) - untuk tabular
- [ ] Convolutional Neural Network (CNN) - untuk image
- [x] **Recurrent Neural Network (LSTM/GRU) - untuk sequential/text**
- [ ] Transfer Learning - untuk image
- [ ] Transformer-based - untuk NLP
- [ ] Autoencoder - untuk unsupervised
- [ ] Neural Collaborative Filtering - untuk recommender

**Alasan Pemilihan:**  
Dataset berupa teks abstrak merupakan data sekuensial (urutan kata). Model tradisional (NB/SVM) memperlakukan kata sebagai fitur independen (Bag of Words) dan mengabaikan urutan. LSTM dipilih karena mampu mengingat konteks jangka panjang dan urutan kata dalam kalimat, yang krusial untuk memahami semantik abstrak ilmiah.

#### 6.3.2 Arsitektur Model

**Deskripsi Layer:**
<img width="805" height="377" alt="image" src="https://github.com/user-attachments/assets/ceb319d9-c4e7-4af8-b90e-ab59ffb1f4fd" />

#### 6.3.3 Input & Preprocessing Khusus

**Input shape:** (None, 200) — Panjang sekuens dibatasi 200 kata. 
**Preprocessing khusus untuk DL:**
- Tokenization: Mengubah setiap kata unik menjadi integer index (Vocabulary size: 10,000 kata teratas).
- Padding/Truncating: Menyamakan panjang semua input menjadi 200 token. Abstrak pendek diberi padding (0), abstrak panjang dipotong.

#### 6.3.4 Hyperparameter

**Training Configuration:**
- Optimizer: Adam (Adaptive Moment Estimation)
- Learning rate: Default (0.001)
- Loss function: Sparse Categorical Crossentropy (Karena label berupa integer, bukan one-hot)
- Metrics: Accuracy
- Batch size: 64
- Epochs: 5
- Validation split: 0.1 (10% dari data train digunakan untuk validasi saat epoch berjalan)

#### 6.3.5 Implementasi (Ringkas)

**Framework:** TensorFlow
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, SpatialDropout1D

# Membangun Arsitektur
model_dl = Sequential()
model_dl.add(Embedding(input_dim=10000, output_dim=100)) # Input Layer
model_dl.add(SpatialDropout1D(0.2))
model_dl.add(LSTM(100, dropout=0.2, recurrent_dropout=0.2)) # Hidden Layer
model_dl.add(Dense(num_classes, activation='softmax')) # Output Layer

# Kompilasi
model_dl.compile(
    loss='sparse_categorical_crossentropy', 
    optimizer='adam', 
    metrics=['accuracy']
)

# Training
history = model_dl.fit(
    X_train_pad, y_train, 
    epochs=5, 
    batch_size=64, 
    validation_split=0.1, 
    verbose=1
)

#### 6.3.6 Training Process

**Training Time:**  
Estimasi waktu training: ~3-5 menit (menggunakan Google Colab CPU/GPU).

**Computational Resource:**  
Platform: Google Colab (Standard Runtime).

**Training History Visualization:**
<img width="997" height="378" alt="Accuracy   Loss" src="https://github.com/user-attachments/assets/f214d99a-42b5-426b-87a0-e8d08ae4089a" />

**Analisis Training:**
- Apakah model mengalami overfitting? Tidak Signifikan. Meskipun akurasi pada data latih (Training Accuracy) sedikit lebih tinggi daripada data validasi (Validation Accuracy), jarak (gap) keduanya tidak terlalu jauh. Hal ini menunjukkan bahwa penggunaan layer SpatialDropout1D dan Dropout pada arsitektur LSTM berhasil mencegah model menghafal data (memorization) dan menjaga kemampuan generalisasi.
- Apakah model sudah converge? Ya. Dilihat dari grafik Loss, penurunan error terjadi secara tajam pada Epoch 1 dan 2, kemudian mulai melandai (flatten) pada Epoch 3 hingga 5. Ini menandakan model telah menemukan pola optimal dengan cepat.
- Apakah perlu lebih banyak epoch? Tidak Perlu. Pada Epoch ke-5, grafik Validation Loss sudah mulai stabil atau bahkan menunjukkan tanda-tanda akan naik jika diteruskan. Menambah jumlah epoch (misalnya menjadi 20 atau 50) hanya akan membuang sumber daya komputasi dan berisiko meningkatkan overfitting tanpa memberikan kenaikan akurasi yang berarti pada data validasi..

#### 6.3.7 Model Summary
Model: "sequential"
_________________________________________________________________
 Layer (type)                Output Shape              Param #   
=================================================================
 embedding (Embedding)       (None, 200, 100)          1000000   
                                                                 
 spatial_dropout1d           (None, 200, 100)          0         
                                                                 
 lstm (LSTM)                 (None, 100)               80400     
                                                                 
 dense (Dense)               (None, 2)                 202       
                                                                 
=================================================================
Total params: 1,080,602
Trainable params: 1,080,602
Non-trainable params: 0
_________________________________________________________________ 

---

## 7. EVALUATION

### 7.1 Metrik Evaluasi

#### **Untuk NLP (Text Classification):**
- **Accuracy:** Digunakan untuk melihat performa model secara global (berapa persen total prediksi yang benar).
- **Weighted F1-Score:** Metrik utama yang digunakan. Karena dataset memiliki distribusi kelas yang tidak seimbang (imbalanced), F1-Score (rata-rata harmonis Precision dan Recall) memberikan gambaran kinerja yang lebih jujur daripada sekadar akurasi.
- **Confusion Matrix:** Digunakan untuk memvisualisasikan detail kesalahan prediksi (berapa banyak False Positive dan False Negative).

### 7.2 Hasil Evaluasi Model

#### 7.2.1 Model 1 (Baseline)

**Metrik:**
- Accuracy: 0.82
- Precision: 0.85
- Recall: 0.81
- F1-Score: 0.74

**Confusion Matrix / Visualization:**  
<img width="458" height="402" alt="Confusion Matrix-Naive Bayes" src="https://github.com/user-attachments/assets/0aef925d-92ad-48c4-b91f-33fc2efb1d2d" />
**Analisis:** Model Baseline memiliki akurasi yang terlihat cukup tinggi (81%), namun nilai F1-Score (0.74) lebih rendah. Hal ini disebabkan karena model cenderung bias memprediksi kelas mayoritas dan gagal mengenali kelas minoritas (Recall kelas 1 sangat rendah).

#### 7.2.2 Model 2 (Advanced/ML)

**Metrik:**
- Accuracy: 0.93
- Precision (Weighted): 0.93
- Recall (Weighted): 0.93
- F1-Score (Weighted): 0.93

**Confusion Matrix / Visualization:**  
<img width="458" height="402" alt="Confusion Matrix-SVM" src="https://github.com/user-attachments/assets/76bdf9b5-f581-4aaa-bdd4-2e7049b30b84" />
**Analisis:** SVM dengan kernel Linear memberikan hasil terbaik. Model ini berhasil menyeimbangkan Precision dan Recall dengan sangat baik (keduanya > 90%), membuktikan bahwa fitur teks abstrak dapat dipisahkan secara linear (linearly separable) dengan baik pada dimensi tinggi.

**Feature Importance (jika applicable):**  
[Insert plot feature importance untuk tree-based models]

#### 7.2.3 Model 3 (Deep Learning)

**Metrik:**
- Accuracy: 0.86
- Precision (Weighted): 0.87
- Recall (Weighted): 0.86
- F1-Score (Weighted): 0.87

**Confusion Matrix / Visualization:**  
<img width="458" height="402" alt="Confusion Matrix-LSTM" src="https://github.com/user-attachments/assets/c482f979-0b44-4b24-baef-ba89f7952cd3" />
**Analisis:** LSTM memberikan performa yang stabil (86%) dan jauh lebih baik daripada Baseline. Namun, pada dataset ini, kompleksitas LSTM belum mampu mengungguli efisiensi SVM.

**Training History:**  
[Sudah diinsert di Section 6.3.6]

**Test Set Predictions:**  
[Opsional: tampilkan beberapa contoh prediksi]

### 7.3 Perbandingan Ketiga Model

**Tabel Perbandingan:**

<img width="937" height="145" alt="image" src="https://github.com/user-attachments/assets/cfa550ab-5964-40b1-aa1a-9a5fe9228352" />


**Visualisasi Perbandingan:**  
<img width="677" height="455" alt="Perbandingan Akurasi Ketiga Model" src="https://github.com/user-attachments/assets/7cdd3a8a-da0c-4c32-952d-a3aafad9d1b3" />


### 7.4 Analisis Hasil

**Interpretasi:**

1. **Model Terbaik:**  
   Support Vector Machine (SVM) adalah model terbaik dengan akurasi 93%. SVM unggul karena kemampuannya menangani data berdimensi tinggi (hasil TF-IDF) dan menemukan margin pemisah yang optimal antar topik.

2. **Perbandingan dengan Baseline:**  
   Kedua model lanjutan (SVM dan LSTM) berhasil mengungguli Baseline (Naive Bayes) secara signifikan dalam hal F1-Score. Ini menunjukkan bahwa Naive Bayes terlalu sederhana untuk menangkap pola pada kelas minoritas, sementara SVM dan LSTM lebih robust.

3. **Trade-off:**  
   SVM memberikan keseimbangan terbaik: waktu training sangat cepat (hanya beberapa detik) dengan akurasi tertinggi. LSTM membutuhkan waktu training paling lama (karena epoch dan komputasi neural network) namun hasilnya masih di bawah SVM untuk kasus data ini.

4. **Error Analysis:**  
   [Jelaskan jenis kesalahan yang sering terjadi, kasus yang sulit diprediksi]

5. **Overfitting/Underfitting:**  
   Overfitting/Underfitting: Berdasarkan kurva training history, LSTM menunjukkan sedikit tanda overfitting (akurasi training terus naik, validasi stagnan), namun diatasi dengan layer Dropout. SVM menunjukkan generalisasi yang sangat baik pada data uji.

---

## 8. CONCLUSION

### 8.1 Kesimpulan Utama

**Model Terbaik:**  
Model Terbaik: SVM (Linear Kernel) dengan Akurasi 93%.

**Alasan:**  
Model ini paling efektif memisahkan topik abstrak jurnal yang direpresentasikan dalam vektor TF-IDF. SVM terbukti lebih tangguh terhadap ketidakseimbangan data dibandingkan Naive Bayes dan lebih efisien secara komputasi dibandingkan LSTM.

**Pencapaian Goals:**  
- Preprocessing data teks berhasil dilakukan.
- Tiga model (Baseline, ML, DL) berhasil dibangun dan dibandingkan.
- Target akurasi >80% berhasil dilampaui oleh SVM (93%) dan LSTM (86%).

### 8.2 Key Insights

**Insight dari Data:**
- Data teks abstrak memiliki pola kata kunci yang kuat (misal: "algorithm", "method" untuk CS vs "patient", "clinical" untuk Medis), memudahkan model linear seperti SVM.
- Penanganan imbalanced data sangat penting; melihat akurasi saja bisa menipu (seperti pada kasus Naive Bayes).

**Insight dari Modeling:**
- Deep Learning tidak selalu menjadi solusi terbaik untuk semua masalah. Untuk dataset teks berukuran sedang dengan fitur yang jelas, algoritma ML klasik seperti SVM bisa mengalahkan Neural Network.

### 8.3 Kontribusi Proyek

**Manfaat praktis:**  
Model ini dapat diimplementasikan pada sistem perpustakaan digital kampus untuk mengotomatisasi proses tagging atau pengelompokan jurnal yang baru diunggah mahasiswa, mengurangi beban kerja pustakawan.

**Pembelajaran yang didapat:**  
Saya mempelajari pentingnya memilih metrik evaluasi yang tepat (F1-Score vs Accuracy) dan memahami bahwa kompleksitas model (LSTM) harus dibayar dengan biaya komputasi yang lebih tinggi.

---

## 9. FUTURE WORK (Opsional)

Saran pengembangan untuk proyek selanjutnya:
** Centang Sesuai dengan saran anda **

**Data:**
- [x] Mengumpulkan lebih banyak data
- [ ] Menambah variasi data
- [ ] Feature engineering lebih lanjut

**Model:**
- [ ] Mencoba arsitektur DL yang lebih kompleks
- [ ] Hyperparameter tuning lebih ekstensif
- [ ] Ensemble methods (combining models)
- [ ] Transfer learning dengan model yang lebih besar

**Deployment:**
- [x] Membuat API (Flask/FastAPI)
- [ ] Membuat web application (Streamlit/Gradio)
- [ ] Containerization dengan Docker
- [ ] Deploy ke cloud (Heroku, GCP, AWS)

**Optimization:**
- [ ] Model compression (pruning, quantization)
- [ ] Improving inference speed
- [ ] Reducing model size

---

## 10. REPRODUCIBILITY (WAJIB)

### 10.1 GitHub Repository

**Link Repository:** https://github.com/adammahabayu/UAS_Praktik_Teknik_2025_TRPL-5A_Data-Science

**Repository harus berisi:**
- ✅ Notebook Jupyter/Colab dengan hasil running
- ✅ Script Python (jika ada)
- ✅ requirements.txt atau environment.yml
- ✅ README.md yang informatif
- ✅ Folder structure yang terorganisir
- ✅ .gitignore (jangan upload dataset besar)

### 10.2 Environment & Dependencies

**Python Version:** 3.10
Proyek ini dijalankan di Google Colab.
**Main Libraries & Versions:**
numpy==1.23.5
pandas==1.5.3
scikit-learn==1.2.2
matplotlib==3.7.1
seaborn==0.13.1
tensorflow==2.15.0
nltk==3.8.1
joblib==1.3.2


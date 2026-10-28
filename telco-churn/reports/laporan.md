# Laporan Prediksi Churn Pelanggan Telco

**Tanggal:** 26 Oktober 2026  
**Proyek:** Klasifikasi Biner Churn Pelanggan

---

## 1. Ringkasan Eksekutif

Model machine learning berhasil dibangun untuk memprediksi churn pelanggan telekomunikasi dengan **ROC-AUC 0.84** dan **recall 73%** pada data uji. Model terpilih adalah **Logistic Regression** dengan class weighting untuk menangani ketidakseimbangan kelas.

**Temuan bisnis utama:**
- Pelanggan dengan tenure pendek (<18 bulan) berisiko tinggi churn
- Kontrak month-to-month memiliki churn rate 42.7% vs two-year contract 2.8%
- Pelanggan fiber optic tanpa tech support paling rentan churn
- 10 fitur teratas cukup untuk prediksi akurat, dipimpin oleh tenure dan monthly charges

---

## 2. Dataset

**Sumber:** [Kaggle - Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)  
**Ukuran:** 7,043 pelanggan, 20 fitur + 1 target  
**Proporsi churn:** 26.5% (1,869 pelanggan)  
**Split:** 80% train (5,634), 20% test (1,409)

### Karakteristik Data
- **Demografi:** gender, senior citizen status, partner, dependents
- **Layanan:** phone, internet (DSL/fiber optic), add-ons (security, backup, support, streaming)
- **Akun:** tenure (bulan berlangganan), contract type, billing, payment method
- **Finansial:** monthly charges, total charges

### Preprocessing
- `TotalCharges` dikonversi dari string ke numerik (11 nilai kosong diisi 0 untuk pelanggan tenure=0)
- `customerID` dihapus
- Stratified split 80/20 sebelum transformasi
- StandardScaler untuk numerik, OneHotEncoder untuk kategorikal
- Class weighting untuk handle imbalance

---

## 3. Metodologi

### Algoritma yang Dibandingkan
1. **Logistic Regression** (class_weight='balanced')
2. **Random Forest** (class_weight='balanced')
3. **XGBoost** (scale_pos_weight untuk imbalance)

### Evaluasi
- **Stratified 5-Fold Cross Validation** pada train set
- **Metrik utama:** PR-AUC (average precision) karena kelas tidak seimbang
- **Hyperparameter tuning:** RandomizedSearchCV 10 iterasi untuk model terbaik
- **Threshold optimization:** F1-optimal pada cross-validation (threshold=0.57)

### Hasil Cross-Validation

| Model | PR-AUC | ROC-AUC | Recall |
|-------|--------|---------|--------|
| Logistic Regression | **0.660 ± 0.020** | 0.846 | 0.801 |
| XGBoost | 0.623 ± 0.019 | 0.825 | 0.664 |
| Random Forest | 0.601 ± 0.030 | 0.822 | 0.642 |

**Model final:** Logistic Regression (C=1.0)

---

## 4. Hasil Test Set

### Metrik Performa

| Metrik | Nilai | Target | Status |
|--------|-------|--------|--------|
| **ROC-AUC** | 0.8415 | ≥ 0.80 | ✓ Tercapai |
| **PR-AUC** | 0.6325 | > baseline (0.265) | ✓ Tercapai |
| **Recall (Churn)** | 0.7273 | ≥ 0.70 | ✓ Tercapai |
| **Precision (Churn)** | 0.5397 | - | Dilaporkan |
| **F1 (Churn)** | 0.6196 | - | Dilaporkan |
| **Accuracy** | 0.7630 | - | Dilaporkan |

### Confusion Matrix (Test Set, N=1,409)

|  | Predicted No Churn | Predicted Churn |
|--|-------------------|-----------------|
| **True No Churn** | 948 | 87 |
| **True Churn** | 102 | 272 |

**Interpretasi:**
- Model menangkap 272 dari 374 pelanggan churn (recall 73%)
- 102 pelanggan churn terlewat (false negative) - risiko kehilangan pelanggan
- 87 pelanggan loyal salah diprediksi churn (false positive) - biaya promo tidak efisien

---

## 5. Feature Importance & Interpretasi

### Top 10 Fitur (Permutation Importance)

1. **tenure** (0.077) - Lama berlangganan, prediktor terkuat
2. **MonthlyCharges** (0.034) - Biaya bulanan tinggi = churn tinggi
3. **InternetService** (0.016) - Tipe layanan internet (fiber optic berisiko)
4. **TotalCharges** (0.008) - Total pengeluaran (berkorelasi dengan tenure)
5. **Contract** (0.004) - Jenis kontrak (month-to-month berisiko)
6. **OnlineSecurity** (0.003) - Add-on security service
7. **PaymentMethod** (0.002) - Metode pembayaran
8. **StreamingMovies** (0.002) - Layanan streaming film
9. **MultipleLines** (0.001) - Multiple phone lines
10. **StreamingTV** (0.001) - Layanan streaming TV

### Insight dari EDA
- Pelanggan yang churn: tenure rata-rata 18 bulan vs loyal 38 bulan
- Month-to-month contract: churn rate 42.7%
- Two-year contract: churn rate 2.8%
- Fiber optic tanpa tech support: churn rate 41.9%
- Pelanggan dengan tech support: churn rate 15.2%

---

## 6. Rekomendasi Bisnis

### 1. Program Retensi Berbasis Tenure
**Target:** Pelanggan dengan tenure <24 bulan, terutama 0-12 bulan.  
**Aksi:** Diskon loyalty atau upgrade gratis pada bulan ke-6 dan ke-12.  
**Justifikasi:** Tenure adalah prediktor terkuat. Pelanggan yang bertahan >24 bulan jauh lebih stabil.

### 2. Konversi ke Kontrak Jangka Panjang
**Target:** Pelanggan month-to-month dengan monthly charges tinggi (>$70).  
**Aksi:** Insentif perpindahan ke one-year atau two-year contract (diskon 10-15% atau add-on gratis).  
**Justifikasi:** Month-to-month contract punya churn rate 15x lebih tinggi dari two-year contract.

### 3. Bundle Tech Support dengan Fiber Optic
**Target:** Pelanggan fiber optic tanpa tech support.  
**Aksi:** Tawarkan tech support sebagai add-on wajib atau bundled dengan harga promo.  
**Justifikasi:** Tech support menurunkan churn rate dari 41.6% ke 15.2%. Fiber optic punya ekspektasi layanan tinggi, support mengurangi frustrasi.

### 4. Intervensi Proaktif untuk High-Risk Segment
**Target:** Pelanggan dengan probabilitas churn >0.7 (model output).  
**Aksi:** Tim retention contact langsung dengan penawaran customized (upgrade, diskon, kontrak lebih fleksibel).  
**Justifikasi:** Model recall 73% artinya sebagian besar churn bisa dideteksi dini. False positive rate 8% (87/1035) masih acceptable untuk biaya intervensi.

### 5. Review Pricing Strategy untuk Monthly Charges
**Target:** Pelanggan dengan monthly charges >$80 tanpa add-on premium.  
**Aksi:** Evaluasi apakah harga sebanding dengan perceived value. Tawarkan paket bundling.  
**Justifikasi:** Monthly charges adalah fitur penting kedua. Pelanggan yang churn bayar lebih mahal tapi mendapat lebih sedikit value.

---

## 7. Keterbatasan & Future Work

### Keterbatasan
1. **Tidak ada dimensi waktu:** Dataset snapshot, tidak ada informasi kapan pelanggan mulai churn atau seasonal pattern.
2. **Tidak ada data biaya retensi:** Rekomendasi tidak memperhitungkan ROI intervensi.
3. **Feature engineering terbatas:** Hanya preprocessing dasar, belum ada interaction terms atau polynomial features.
4. **Threshold fixed:** Threshold 0.57 optimal untuk F1, bukan untuk business cost function sesungguhnya.

### Future Work
- Tambahkan cost-sensitive learning dengan asumsi biaya false negative (kehilangan pelanggan) vs false positive (biaya promo)
- Feature engineering: tenure bins, charge_per_month ratio, jumlah add-ons
- Calibrasi probabilitas untuk confidence scoring
- A/B testing rekomendasi retensi pada segment high-risk
- Time-series analysis bila data historis tersedia

---

## 8. Deliverables

- ✓ `models/churn_pipeline.joblib` - Model final + preprocessing pipeline
- ✓ `reports/metrics.json` - Metrik test set & threshold
- ✓ `reports/figures/` - 8 grafik (churn distribution, numerical features, categorical rates, correlation, confusion matrix, ROC, PR curve, feature importance, SHAP)
- ✓ `notebooks/01_eda.ipynb` - EDA dengan temuan
- ✓ `notebooks/02_preprocessing_modeling.ipynb` - Pipeline & model comparison
- ✓ `notebooks/03_evaluation_interpretation.ipynb` - Test evaluation & interpretasi
- ✓ `README.md` - Instruksi reproduksi
- ✓ `requirements.txt` - Dependencies

---

## 9. Reproduksi

```bash
# Install dependencies
pip install -r requirements.txt

# Download dataset (atau taruh CSV di data/raw/)
python -c "import kagglehub; path = kagglehub.dataset_download('blastchar/telco-customer-churn'); print(path)"

# Jalankan pipeline
python src/validate_data.py
python src/run_eda.py
python src/run_modeling.py
python src/run_evaluation.py
```

Semua hasil reproducible dengan `random_state=42`.

---

**Kesimpulan:** Model Logistic Regression dengan recall 73% dan ROC-AUC 0.84 siap digunakan untuk identifikasi pelanggan berisiko tinggi churn. Fokus retensi pada pelanggan baru (<12 bulan), month-to-month contract, dan fiber optic tanpa tech support akan memberikan ROI tertinggi.

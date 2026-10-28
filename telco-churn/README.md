# Telco Customer Churn Prediction

Proyek klasifikasi biner untuk memprediksi pelanggan telekomunikasi yang berisiko churn menggunakan dataset Kaggle Telco Customer Churn.

## Struktur Folder

```text
telco-churn/
├── data/
│   ├── raw/              # CSV asli (WA_Fn-UseC_-Telco-Customer-Churn.csv)
│   └── processed/        # data hasil split/preprocessing
├── notebooks/
│   ├── 01_eda.ipynb      # Validasi & EDA
│   ├── 02_preprocessing_modeling.ipynb  # Cross-validation & training
│   └── 03_evaluation_interpretation.ipynb # Test set evaluation & SHAP/feature importance
├── src/                  # Script pendukung
│   ├── validate_data.py
│   ├── run_eda.py
│   ├── run_modeling.py
│   └── run_evaluation.py
├── models/
│   └── churn_pipeline.joblib  # Model final + pipeline
├── reports/
│   ├── figures/          # Visualisasi grafik
│   ├── cv_evaluation.json# Hasil Cross Validation
│   ├── metrics.json      # Metrik test set akhir
│   └── laporan.md        # Laporan lengkap
├── requirements.txt
└── README.md
```

## Cara Menjalankan

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Unduh dataset:**
   Dataset sudah terunduh otomatis via KaggleHub di `data/raw/` saat script pertama dijalankan. Atau unduh manual dataset Telco Customer Churn dari Kaggle dan simpan sebagai `data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv`.

3. **Jalankan Pipeline End-to-End:**
   ```bash
   python src/validate_data.py
   python src/run_eda.py
   python src/run_modeling.py
   python src/run_evaluation.py
   ```

4. **Jalankan Notebooks (Opsional):**
   Gunakan Jupyter Lab / VS Code untuk membukanya secara interaktif:
   - `notebooks/01_eda.ipynb`
   - `notebooks/02_preprocessing_modeling.ipynb`
   - `notebooks/03_evaluation_interpretation.ipynb`

## Hasil Utama Test Set

- **Model Final:** Logistic Regression (C=1.0, class_weight='balanced')
- **ROC-AUC:** 0.8415
- **PR-AUC:** 0.6325
- **Recall (Churn):** 0.7273 ( threshold optimal = 0.57 )
- **Precision (Churn):** 0.5397
- **F1 (Churn):** 0.6196
- **Accuracy:** 0.7630

Detail laporan dan rekomendasi bisnis terdapat di [reports/laporan.md](reports/laporan.md).

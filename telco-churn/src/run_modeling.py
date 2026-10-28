import pandas as pd
import numpy as np
import json
import os
import joblib

from sklearn.model_selection import train_test_split, StratifiedKFold, RandomizedSearchCV, cross_validate
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (
    roc_auc_score, average_precision_score, recall_score, 
    precision_score, f1_score, accuracy_score, confusion_matrix, classification_report
)

print("Loading data...")
df = pd.read_csv("data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv")

# Preprocessing sesuai FR-3
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'].str.strip(), errors='coerce').fillna(0)
df = df.drop(columns=['customerID'])

# Map target
df['Churn'] = (df['Churn'] == 'Yes').astype(int)

# Identifikasi kolom numerik dan kategorikal
X = df.drop(columns=['Churn'])
y = df['Churn']

numeric_features = ['tenure', 'MonthlyCharges', 'TotalCharges']
categorical_features = [col for col in X.columns if col not in numeric_features]

print(f"Fitur numerik: {numeric_features}")
print(f"Fitur kategorikal: {categorical_features}")

# Split 80/20 stratify dengan random_state=42 (FR-3.4)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

print(f"Train shape: {X_train.shape}, Test shape: {X_test.shape}")

# Preprocessor transformer (FR-3.5)
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ]
)

# Model 1: Logistic Regression
lr_model = LogisticRegression(class_weight='balanced', random_state=42, max_iter=1000)
# Model 2: Random Forest
rf_model = RandomForestClassifier(class_weight='balanced', random_state=42, n_jobs=-1)
# Model 3: XGBoost (dengan scale_pos_weight untuk handle imbalanced)
scale_pos_weight = (len(y_train) - sum(y_train)) / sum(y_train)
xgb_model = XGBClassifier(scale_pos_weight=scale_pos_weight, random_state=42, eval_metric='logloss', n_jobs=-1)

models = {
    "Logistic Regression": lr_model,
    "Random Forest": rf_model,
    "XGBoost": xgb_model
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

print("\n--- Cross Validation Evaluation (FR-4.2) ---")
cv_results = {}
for name, model in models.items():
    pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('classifier', model)])
    
    scores = cross_validate(
        pipeline, X_train, y_train, 
        cv=cv, 
        scoring=['roc_auc', 'average_precision', 'recall', 'f1', 'accuracy'],
        n_jobs=-1
    )
    
    cv_results[name] = {
        "roc_auc_mean": float(scores['test_roc_auc'].mean()),
        "roc_auc_std": float(scores['test_roc_auc'].std()),
        "pr_auc_mean": float(scores['test_average_precision'].mean()),
        "pr_auc_std": float(scores['test_average_precision'].std()),
        "recall_mean": float(scores['test_recall'].mean()),
        "recall_std": float(scores['test_recall'].std()),
        "f1_mean": float(scores['test_f1'].mean()),
        "f1_std": float(scores['test_f1'].std()),
        "accuracy_mean": float(scores['test_accuracy'].mean()),
        "accuracy_std": float(scores['test_accuracy'].std()),
    }
    print(f"[{name}] PR-AUC: {cv_results[name]['pr_auc_mean']:.4f} (±{cv_results[name]['pr_auc_std']:.4f}) | ROC-AUC: {cv_results[name]['roc_auc_mean']:.4f} | Recall: {cv_results[name]['recall_mean']:.4f}")

# Pilih model terbaik berdasarkan PR-AUC mean
best_model_name = max(cv_results, key=lambda k: cv_results[k]['pr_auc_mean'])
print(f"\nModel terbaik terpilih: {best_model_name}")

# Hiperparameter tuning untuk 2 model terbaik atau model terbaik (FR-4.3)
print("\n--- Hyperparameter Tuning ---")
if best_model_name == "Random Forest":
    param_dist = {
        'classifier__n_estimators': [100, 200, 300],
        'classifier__max_depth': [10, 20, 30, None],
        'classifier__min_samples_split': [2, 5, 10],
        'classifier__min_samples_leaf': [1, 2, 4]
    }
elif best_model_name == "XGBoost":
    param_dist = {
        'classifier__n_estimators': [100, 200],
        'classifier__max_depth': [3, 5, 7],
        'classifier__learning_rate': [0.01, 0.05, 0.1],
        'classifier__subsample': [0.8, 1.0]
    }
else:
    param_dist = {
        'classifier__C': [0.01, 0.1, 1.0, 10.0]
    }

best_pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('classifier', models[best_model_name])])
search = RandomizedSearchCV(
    best_pipeline, param_distributions=param_dist, n_iter=10, 
    scoring='average_precision', cv=cv, random_state=42, n_jobs=-1
)
search.fit(X_train, y_train)

final_pipeline = search.best_estimator_
print(f"Best hyperparameters: {search.best_params_}")

# Tentukan threshold optimal pada data validasi (cross-validation out-of-fold or train set evaluation) (FR-4.5)
# Untuk kesederhanaan dan ketepatan tanpa kebocoran, kita gunakan data latih untuk mencari threshold F1-optimal, atau dari cross-validation.
# Kita gunakan teknik mencari threshold terbaik yang memaksimalkan F1 pada train set (atau stratified k-fold OOF).
from sklearn.model_selection import cross_val_predict
y_train_probs = cross_val_predict(final_pipeline, X_train, y_train, cv=cv, method='predict_proba')[:, 1]

thresholds = np.linspace(0.1, 0.9, 81)
best_threshold = 0.5
best_f1 = 0
for th in thresholds:
    preds = (y_train_probs >= th).astype(int)
    score = f1_score(y_train, preds)
    if score > best_f1:
        best_f1 = score
        best_threshold = th

print(f"Threshold optimal terkunci: {best_threshold:.4f} dengan F1 train: {best_f1:.4f}")

# Simpan model final (FR-5 / Deliverables)
os.makedirs("models", exist_ok=True)
joblib.dump(final_pipeline, "models/churn_pipeline.joblib")
print("Model tersimpan di models/churn_pipeline.joblib")

# Simpan metadata & hasil CV ke temporary json / evaluation
evaluation_data = {
    "cv_results": cv_results,
    "best_model": best_model_name,
    "best_params": search.best_params_,
    "optimal_threshold": float(best_threshold)
}
with open("reports/cv_evaluation.json", "w") as f:
    json.dump(evaluation_data, f, indent=4)

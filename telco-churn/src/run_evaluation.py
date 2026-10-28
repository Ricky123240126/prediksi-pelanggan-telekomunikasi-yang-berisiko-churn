import pandas as pd
import numpy as np
import json
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_auc_score, average_precision_score, recall_score, precision_score, 
    f1_score, accuracy_score, confusion_matrix, roc_curve, precision_recall_curve, auc
)
import shap

# Load data
df = pd.read_csv("data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv")
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'].str.strip(), errors='coerce').fillna(0)
df = df.drop(columns=['customerID'])
df['Churn'] = (df['Churn'] == 'Yes').astype(int)

X = df.drop(columns=['Churn'])
y = df['Churn']

# Recreate split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# Load model dan threshold
final_pipeline = joblib.load("models/churn_pipeline.joblib")
with open("reports/cv_evaluation.json", "r") as f:
    eval_data = json.load(f)
    optimal_threshold = eval_data["optimal_threshold"]

print(f"Optimal threshold: {optimal_threshold}")

# Prediksi pada test set
y_pred_proba = final_pipeline.predict_proba(X_test)[:, 1]
y_pred = (y_pred_proba >= optimal_threshold).astype(int)

# Evaluasi metrik (FR-5.1, FR-5.2)
roc_auc = roc_auc_score(y_test, y_pred_proba)
pr_auc = average_precision_score(y_test, y_pred_proba)
recall = recall_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
acc = accuracy_score(y_test, y_pred)

metrics = {
    "ROC_AUC": float(roc_auc),
    "PR_AUC": float(pr_auc),
    "Recall": float(recall),
    "Precision": float(precision),
    "F1": float(f1),
    "Accuracy": float(acc),
    "Threshold": float(optimal_threshold),
    "Test_Set_Size": len(y_test),
    "Churn_Cases": int(y_test.sum())
}

print("\n--- TEST SET METRICS ---")
print(f"ROC-AUC: {roc_auc:.4f}")
print(f"PR-AUC: {pr_auc:.4f}")
print(f"Recall (Churn): {recall:.4f}")
print(f"Precision (Churn): {precision:.4f}")
print(f"F1 (Churn): {f1:.4f}")
print(f"Accuracy: {acc:.4f}")

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['No Churn', 'Churn'], yticklabels=['No Churn', 'Churn'])
plt.title('Confusion Matrix (Test Set)')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.tight_layout()
plt.savefig("reports/figures/confusion_matrix.png", dpi=300)
plt.close()

# ROC Curve
fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
plt.figure(figsize=(7, 5))
plt.plot(fpr, tpr, label=f'ROC-AUC = {roc_auc:.4f}', linewidth=2)
plt.plot([0, 1], [0, 1], 'k--', label='Random')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve (Test Set)')
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("reports/figures/roc_curve.png", dpi=300)
plt.close()

# PR Curve
precision_vals, recall_vals, _ = precision_recall_curve(y_test, y_pred_proba)
plt.figure(figsize=(7, 5))
plt.plot(recall_vals, precision_vals, label=f'PR-AUC = {pr_auc:.4f}', linewidth=2)
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve (Test Set)')
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("reports/figures/pr_curve.png", dpi=300)
plt.close()

# Feature Importance (Permutation)
from sklearn.inspection import permutation_importance
perm_importance = permutation_importance(final_pipeline, X_test, y_test, n_repeats=10, random_state=42, n_jobs=-1)
feature_importance_df = pd.DataFrame({
    'feature': X.columns,
    'importance': perm_importance.importances_mean
}).sort_values('importance', ascending=False)

print("\n--- TOP 10 FEATURES (Permutation Importance) ---")
print(feature_importance_df.head(10))

top_features_perm = feature_importance_df.head(10)
plt.figure(figsize=(10, 6))
plt.barh(top_features_perm['feature'], top_features_perm['importance'])
plt.xlabel('Permutation Importance')
plt.title('Top 10 Most Important Features')
plt.tight_layout()
plt.savefig("reports/figures/feature_importance.png", dpi=300)
plt.close()

# SHAP Values
try:
    # Ambil preprocessor dan classifier
    preprocessor = final_pipeline.named_steps['preprocessor']
    classifier = final_pipeline.named_steps['classifier']
    
    # Transform test data
    X_test_transformed = preprocessor.transform(X_test)
    
    # Explainer SHAP untuk logistic regression
    explainer = shap.LinearExplainer(classifier, X_test_transformed)
    shap_values = explainer.shap_values(X_test_transformed)
    
    # SHAP summary plot
    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values, X_test_transformed, plot_type="bar", show=False)
    plt.tight_layout()
    plt.savefig("reports/figures/shap_summary.png", dpi=300)
    plt.close()
    print("\nSHAP summary plot saved.")
except Exception as e:
    print(f"SHAP calculation skipped: {e}")

# Simpan metrics ke json
metrics['Feature_Importance_Top_10'] = top_features_perm[['feature', 'importance']].to_dict('records')
with open("reports/metrics.json", "w") as f:
    json.dump(metrics, f, indent=4)

print("\nMetrics saved to reports/metrics.json")
print("Evaluasi selesai.")

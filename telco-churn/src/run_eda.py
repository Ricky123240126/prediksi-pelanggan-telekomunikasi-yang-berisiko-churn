import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

os.makedirs("reports/figures", exist_ok=True)

df = pd.read_csv("data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv")

# FR-3.1 Preprocess TotalCharges for EDA visualization
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'].str.strip(), errors='coerce').fillna(0)

# Set style
sns.set_theme(style="whitegrid")

# 1. Churn Distribution
plt.figure(figsize=(6, 4))
ax = sns.countplot(data=df, x='Churn', palette='Set2')
plt.title("Distribusi Churn Pelanggan")
plt.xlabel("Churn")
plt.ylabel("Jumlah Pelanggan")
for p in ax.patches:
    ax.annotate(f'{int(p.get_height())} ({p.get_height()/len(df):.1%})', 
                (p.get_x() + p.get_width() / 2., p.get_height()), 
                ha='center', va='center', xytext=(0, 5), textcoords='offset points')
plt.tight_layout()
plt.savefig("reports/figures/churn_distribution.png", dpi=300)
plt.close()

# 2. Numerical Features
num_cols = ['tenure', 'MonthlyCharges', 'TotalCharges']
fig, axes = plt.subplots(3, 2, figsize=(12, 10))
for i, col in enumerate(num_cols):
    sns.histplot(data=df, x=col, hue='Churn', kde=True, ax=axes[i, 0], palette='Set2')
    axes[i, 0].set_title(f"Histogram {col}")
    
    sns.boxplot(data=df, x='Churn', y=col, ax=axes[i, 1], palette='Set2')
    axes[i, 1].set_title(f"Boxplot {col} per Churn")

plt.tight_layout()
plt.savefig("reports/figures/numerical_features.png", dpi=300)
plt.close()

# 3. Categorical Churn Rate
cat_cols = ['Contract', 'InternetService', 'PaymentMethod', 'TechSupport']
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
axes = axes.flatten()

for i, col in enumerate(cat_cols):
    churn_rate = df.groupby(col)['Churn'].apply(lambda x: (x == 'Yes').mean()).reset_index()
    sns.barplot(data=churn_rate, x=col, y='Churn', ax=axes[i], palette='Set2')
    axes[i].set_title(f"Churn Rate berdasarkan {col}")
    axes[i].set_ylabel("Tingkat Churn")
    axes[i].set_ylim(0, 0.6)
    for p in axes[i].patches:
        axes[i].annotate(f'{p.get_height():.1%}', 
                         (p.get_x() + p.get_width() / 2., p.get_height()), 
                         ha='center', va='center', xytext=(0, 5), textcoords='offset points')

plt.tight_layout()
plt.savefig("reports/figures/categorical_churn_rate.png", dpi=300)
plt.close()

# 4. Correlation Heatmap
df_corr = df.copy()
df_corr['Churn_Num'] = (df_corr['Churn'] == 'Yes').astype(int)
corr = df_corr[['tenure', 'MonthlyCharges', 'TotalCharges', 'Churn_Num']].corr()

plt.figure(figsize=(7, 5))
sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f", vmin=-1, vmax=1)
plt.title("Heatmap Korelasi Fitur Numerik & Churn")
plt.tight_layout()
plt.savefig("reports/figures/correlation_heatmap.png", dpi=300)
plt.close()

print("EDA plots created successfully.")

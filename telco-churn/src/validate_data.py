import pandas as pd
import os

data_path = "data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv"
if not os.path.exists(data_path):
    raise FileNotFoundError(f"File {data_path} tidak ditemukan.")

df = pd.read_csv(data_path)
print("--- SHAPE ---")
print(df.shape)

print("\n--- INFO ---")
print(df.info())

print("\n--- MISSING VALUES ---")
print(df.isnull().sum())

print("\n--- DUPLICATES ---")
print(f"Jumlah duplikat: {df.duplicated().sum()}")

print("\n--- CUSTOMER ID UNIQUE ---")
print(f"Total baris: {len(df)}, Unique customerID: {df['customerID'].nunique()}")

print("\n--- CHURN DISTRIBUTION ---")
print(df['Churn'].value_counts(dropna=False))

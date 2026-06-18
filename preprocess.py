"""
preprocess.py
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import joblib
import os


def load_and_preprocess(filepath="credit_risk_dataset.csv"):

    print(f"📂 Loading dataset from: {filepath}")

    if filepath.endswith('.xlsx'):
        df = pd.read_excel(filepath)
    else:
        df = pd.read_csv(filepath)

    print(f"   Original shape: {df.shape}")

    # ── Strip whitespace from all string columns ──────────────
    # This fixes hidden spaces like " N" or " Y" in Excel files
    for col in df.select_dtypes(include='object').columns:
        df[col] = df[col].astype(str).str.strip().str.upper()

    # ── Show unique values before encoding (for debugging) ────
    print("\n   Unique values check:")
    for col in ['loan_status', 'cb_person_default_on_file',
                'person_home_ownership', 'loan_grade', 'loan_intent']:
        if col in df.columns:
            print(f"     {col}: {sorted(df[col].unique())}")

    # ── Fix loan_status ───────────────────────────────────────
    # It should be 0 or 1 (integer). If it came as string, fix it.
    if df['loan_status'].dtype == object:
        df['loan_status'] = df['loan_status'].map({'0': 0, '1': 1, 'NO': 0, 'YES': 1})

    df['loan_status'] = pd.to_numeric(df['loan_status'], errors='coerce')

    # ── Drop missing values ───────────────────────────────────
    df = df.dropna()
    print(f"\n   After dropna: {df.shape}")

    # ── Check class balance ───────────────────────────────────
    print(f"   loan_status distribution:\n{df['loan_status'].value_counts()}")

    if df['loan_status'].nunique() < 2:
        print("\n❌ CRITICAL: loan_status has only one class!")
        print("   This means the dataset loaded incorrectly.")
        print("   Try re-downloading from Kaggle as CSV directly.")
        exit(1)

    # ── Remove outliers ───────────────────────────────────────
    df = df[df['person_emp_length'] <= 60]
    df = df[df['person_age'] <= 100]
    print(f"   After outlier removal: {df.shape}")

    # ── Encode categorical columns ────────────────────────────
    categorical_cols = [
        'person_home_ownership',
        'loan_intent',
        'loan_grade',
        'cb_person_default_on_file'
    ]

    encoders = {}
    for col in categorical_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        encoders[col] = le
        print(f"   Encoded: {col} → classes: {list(le.classes_)}")

    # ── Save encoders ─────────────────────────────────────────
    os.makedirs("models", exist_ok=True)
    joblib.dump(encoders, "models/encoders.pkl")
    print("✅ Encoders saved to models/encoders.pkl")

    return df, encoders


def get_feature_columns():
    return [
        'person_age',
        'person_income',
        'person_home_ownership',
        'person_emp_length',
        'loan_intent',
        'loan_grade',
        'loan_amnt',
        'loan_int_rate',
        'loan_percent_income',
        'cb_person_default_on_file',
        'cb_person_cred_hist_length'
    ]

import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

from preprocess import load_and_preprocess, get_feature_columns
from model1_npa import train_model1
from model2_consortium import train_model2
import os

print("=" * 70)
print("   AI LOAN DEFAULT PREDICTION — MULTI-ALGORITHM TRAINING")
print("   with GridSearchCV Hyperparameter Tuning")
print("=" * 70)

# ── Check dataset exists ──────────────────────────────────────
if not os.path.exists("credit_risk_dataset.csv"):
    print("\n❌ ERROR: credit_risk_dataset.csv not found!")
    print("   Please download it from Kaggle and place it here:")
    print("   https://www.kaggle.com/datasets/laotse/credit-risk-dataset")
    exit(1)

if not os.path.exists("bank_dataset.csv"):
    print("\n❌ ERROR: bank_dataset.csv not found!")
    print("   Make sure bank_dataset.csv is in the same folder.")
    exit(1)

# ── Step 1: Load & preprocess data ───────────────────────────
print("\n📦 STEP 1: Loading and preprocessing dataset...")
df, encoders = load_and_preprocess("credit_risk_dataset.csv")

# ── Step 2: Get feature columns ───────────────────────────────
feature_cols = get_feature_columns()
print(f"\n📋 Features used ({len(feature_cols)}):")
for i, col in enumerate(feature_cols, 1):
    print(f"   {i:2d}. {col}")

# ── Step 3: Train Model 1 (3 algorithms + GridSearchCV) ───────
print("\n📦 STEP 2: Training Model 1 — NPA Prediction")
print("   Algorithms: Logistic Regression, Random Forest, XGBoost/Gradient Boosting")
model1, X_test, y_test = train_model1(df, feature_cols)

# ── Step 4: Train Model 2 (3 algorithms + GridSearchCV) ───────
print("\n📦 STEP 3: Training Model 2 — Consortium Decision")
print("   Algorithms: Decision Tree, KNN, Gradient Boosting")
model2 = train_model2(df, feature_cols, model1)

# ── Done ──────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("   ✅ ALL TRAINING COMPLETE!")
print("=" * 70)
print("\n📁 Saved files:")
print("   models/model1_best.pkl  — Best NPA prediction model")
print("   models/model2_best.pkl  — Best Consortium decision model")
print("   models/encoders.pkl     — Label encoders")
print("\n🚀 Now run the app with:")
print("   streamlit run app.py")
print("=" * 70)
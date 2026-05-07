"""
Run this ONCE locally to generate the runtime artifacts:
  model.pkl, encoders.pkl, dashboard_data.parquet

Then commit those three files and remove the two CSVs from the repo.

Usage:
    python prepare_artifacts.py
"""
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

CATEGORICAL_COLS = [
    "STUDENT_GENDER", "RACE_GRP", "STUDENT_ETHNICITY",
    "LANG_GRP", "STUDENT_CURRENT_GRADE_CODE", "SCHOOL_GRP",
]
NUMERIC_COLS = [
    "STUDENT_AGE", "STUDENT_SPECIAL_ED_INDICATOR", "STUDENT_HOMELESS_INDICATOR",
]
KEEP_COLS = [
    "STUDENT_KEY", "ENROLLMENT_HISTORY_STATUS",
    *CATEGORICAL_COLS, *NUMERIC_COLS, "target",
]
TARGET = "target"


def load_and_trim(path):
    df = pd.read_csv(path, usecols=KEEP_COLS)
    for col in CATEGORICAL_COLS:
        df[col] = df[col].astype("category")
    for col in NUMERIC_COLS:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype("int8")
    df[TARGET] = df[TARGET].astype("int8")
    return df


print("Loading CSVs…")
train = load_and_trim("train_with_target.csv")
test  = load_and_trim("test_with_target.csv")
print(f"  train: {len(train):,} rows   test: {len(test):,} rows")

# ── Encode & train ────────────────────────────────────────────────────────────
print("Encoding and training…")
encoders = {}
train_enc = train.copy()
for col in CATEGORICAL_COLS:
    le = LabelEncoder()
    train_enc[col] = le.fit_transform(train_enc[col].astype(str).fillna("Unknown"))
    encoders[col] = le

X_train = train_enc[CATEGORICAL_COLS + NUMERIC_COLS]
y_train = train_enc[TARGET].astype(int)

rf = RandomForestClassifier(
    n_estimators=100, max_depth=10, min_samples_leaf=50,
    n_jobs=-1, random_state=42, class_weight="balanced",
)
rf.fit(X_train, y_train)
print("  Training done.")

# ── Pre-compute test predictions ──────────────────────────────────────────────
test_enc = test.copy()
for col in CATEGORICAL_COLS:
    le = encoders[col]
    known = set(le.classes_)
    test_enc[col] = (
        test_enc[col].astype(str).fillna("Unknown")
        .apply(lambda v: v if v in known else le.classes_[0])
    )
    test_enc[col] = le.transform(test_enc[col])

X_test = test_enc[CATEGORICAL_COLS + NUMERIC_COLS]
risk_proba = rf.predict_proba(X_test)[:, 1].astype("float32")

# ── Build dashboard parquet (test rows + risk scores) ────────────────────────
dashboard = test.copy()
dashboard["risk_proba"] = risk_proba

# Re-attach original string values for display (they were cast to category above)
dashboard.to_parquet("dashboard_data.parquet", index=False)
print("  Saved dashboard_data.parquet")

# Also need train for Student Lookup dropdowns & population charts — save slim version
train_slim = train[CATEGORICAL_COLS + ["ENROLLMENT_HISTORY_STATUS", TARGET]].copy()
# Combine train+test for population-level charts
all_slim = pd.concat([train_slim, test[CATEGORICAL_COLS + ["ENROLLMENT_HISTORY_STATUS", TARGET]]], ignore_index=True)
all_slim["STUDENT_AGE"] = pd.concat([train["STUDENT_AGE"], test["STUDENT_AGE"]], ignore_index=True)
all_slim["STUDENT_SPECIAL_ED_INDICATOR"] = pd.concat(
    [train["STUDENT_SPECIAL_ED_INDICATOR"], test["STUDENT_SPECIAL_ED_INDICATOR"]], ignore_index=True
)
all_slim["STUDENT_HOMELESS_INDICATOR"] = pd.concat(
    [train["STUDENT_HOMELESS_INDICATOR"], test["STUDENT_HOMELESS_INDICATOR"]], ignore_index=True
)
all_slim["STUDENT_KEY"] = pd.concat([train["STUDENT_KEY"], test["STUDENT_KEY"]], ignore_index=True)
all_slim.to_parquet("population_data.parquet", index=False)
print("  Saved population_data.parquet")

# ── Save model & encoders ─────────────────────────────────────────────────────
joblib.dump(rf, "model.pkl")
joblib.dump(encoders, "encoders.pkl")
print("  Saved model.pkl and encoders.pkl")

print("\nDone! Commit: model.pkl, encoders.pkl, dashboard_data.parquet, population_data.parquet")
print("You can now remove train_with_target.csv and test_with_target.csv from the repo.")

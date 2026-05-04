"""
train_model.py
Trains multiple classifiers on heart.csv and saves them to models/.
Models: Random Forest, XGBoost (via scikit-learn HistGradientBoosting), Logistic Regression
"""

import os
import json
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
)

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "heart.csv")
MDL_DIR   = os.path.join(BASE_DIR, "models")
os.makedirs(MDL_DIR, exist_ok=True)

# ── Load & Preprocess ─────────────────────────────────────────────────────────
df = pd.read_csv(DATA_PATH)
df.columns = df.columns.str.strip().str.replace(" ", "_")

df = pd.get_dummies(df, drop_first=True)

X = df.drop("Heart_Disease", axis=1)
y = df["Heart_Disease"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Save columns so the app can align inputs
joblib.dump(list(X.columns), os.path.join(MDL_DIR, "model_columns.pkl"))
print(f"✅ Saved model_columns.pkl  ({len(X.columns)} features)")

# ── Model Definitions ─────────────────────────────────────────────────────────
models = {
    "Random Forest": RandomForestClassifier(
        n_estimators=200, max_depth=10, random_state=42, n_jobs=-1
    ),
    "XGBoost": HistGradientBoostingClassifier(
        max_iter=200, max_depth=6, random_state=42
    ),
    "Logistic Regression": Pipeline([
        ("scaler", StandardScaler()),
        ("clf",    LogisticRegression(max_iter=1000, random_state=42))
    ]),
}

# ── Train & Evaluate ──────────────────────────────────────────────────────────
results = {}
for name, clf in models.items():
    clf.fit(X_train, y_train)
    preds = clf.predict(X_test)
    probs = clf.predict_proba(X_test)[:, 1]

    metrics = {
        "Accuracy":  round(accuracy_score(y_test, preds),  4),
        "Precision": round(precision_score(y_test, preds, zero_division=0), 4),
        "Recall":    round(recall_score(y_test, preds,    zero_division=0), 4),
        "F1":        round(f1_score(y_test, preds,        zero_division=0), 4),
        "AUC-ROC":   round(roc_auc_score(y_test, probs),  4),
    }
    results[name] = metrics

    fname = name.lower().replace(" ", "_") + "_model.pkl"
    joblib.dump(clf, os.path.join(MDL_DIR, fname))
    print(f"✅ Saved {fname}  |  Accuracy={metrics['Accuracy']:.2%}  AUC={metrics['AUC-ROC']:.4f}")

# Save metrics JSON for the app
with open(os.path.join(MDL_DIR, "model_metrics.json"), "w") as f:
    json.dump(results, f, indent=2)

print("\n🎉 All models trained and saved to models/")
print(json.dumps(results, indent=2))
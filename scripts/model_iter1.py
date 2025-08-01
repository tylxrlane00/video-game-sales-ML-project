# model_iter1.py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    confusion_matrix, roc_curve
)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.dummy import DummyClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import joblib
from pathlib import Path
from datetime import datetime

# ================================
# Setup
# ================================
OUT_DIR = Path("outputs_iter1")
OUT_DIR.mkdir(exist_ok=True, parents=True)

DATA_PATH = "feature_set_1.csv"
TARGET = "Top_Grossing_Flag"
DROP_COLS = ["Game", "Publisher", "Era", "Units_Sold"]  # leakage / text fields

# ================================
# Load data
# ================================
df = pd.read_csv(DATA_PATH)

num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
X_cols = [c for c in num_cols if c not in [TARGET] + DROP_COLS]
X = df[X_cols]
y = df[TARGET].astype(int)

# ================================
# Class balance check
# ================================
class_counts = y.value_counts().rename_axis("class").reset_index(name="count")
print("Class balance:\n", class_counts)
class_counts.to_csv(OUT_DIR / "class_balance.csv", index=False)

# ================================
# Train/test split
# ================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

def evaluate(model_name, y_true, y_pred, y_proba=None):
    return {
        "model": model_name,
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_true, y_proba) if y_proba is not None else 0.5
    }

# ================================
# Dummy Classifier
# ================================
dummy = DummyClassifier(strategy="most_frequent", random_state=42)
dummy.fit(X_train, y_train)
y_pred_dummy = dummy.predict(X_test)
metrics_dummy = evaluate("Dummy (most_frequent)", y_test, y_pred_dummy)

# ================================
# Logistic Regression (balanced)
# ================================
logreg = Pipeline([
    ("scaler", StandardScaler(with_mean=False)),
    ("clf", LogisticRegression(max_iter=2000, n_jobs=-1, class_weight="balanced"))
])
logreg.fit(X_train, y_train)
y_pred_lr = logreg.predict(X_test)
y_proba_lr = logreg.predict_proba(X_test)[:, 1]
metrics_lr = evaluate("Logistic Regression (balanced)", y_test, y_pred_lr, y_proba_lr)

# ================================
# Random Forest (default + tuned)
# ================================
rf_default = RandomForestClassifier(
    n_estimators=300,
    random_state=42,
    n_jobs=-1,
    class_weight="balanced_subsample"
)
rf_default.fit(X_train, y_train)
y_pred_rf = rf_default.predict(X_test)
y_proba_rf = rf_default.predict_proba(X_test)[:, 1]
metrics_rf_default = evaluate("Random Forest (default-ish)", y_test, y_pred_rf, y_proba_rf)

rf_tuned = RandomForestClassifier(
    n_estimators=400,
    max_depth=20,
    min_samples_split=5,
    random_state=42,
    n_jobs=-1,
    class_weight="balanced_subsample"
)
rf_tuned.fit(X_train, y_train)
y_pred_rft = rf_tuned.predict(X_test)
y_proba_rft = rf_tuned.predict_proba(X_test)[:, 1]
metrics_rf_tuned = evaluate("Random Forest (light-tuned)", y_test, y_pred_rft, y_proba_rft)

# ================================
# 5-fold Cross-Validation for tuned RF
# ================================
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(rf_tuned, X, y, scoring="f1", cv=cv, n_jobs=-1)
pd.Series(cv_scores, name="f1").to_csv(OUT_DIR / "rf_tuned_cv_f1_scores.csv", index=False)
print("\n5-fold CV F1 scores (RF tuned):", cv_scores, "\nMean:", cv_scores.mean())

# ================================
# Compile metrics
# ================================
metrics_df = pd.DataFrame([
    metrics_dummy, metrics_lr, metrics_rf_default, metrics_rf_tuned
]).sort_values(["f1", "roc_auc"], ascending=False)
print("\nMetrics:\n", metrics_df)
timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
metrics_df.to_csv(OUT_DIR / f"iter1_metrics_{timestamp}.csv", index=False)

best_name = metrics_df.iloc[0]["model"]
best_model = {
    "Dummy (most_frequent)": dummy,
    "Logistic Regression (balanced)": logreg,
    "Random Forest (default-ish)": rf_default,
    "Random Forest (light-tuned)": rf_tuned
}[best_name]
joblib.dump(best_model, OUT_DIR / "best_iter1_model.pkl")

# ================================
# Confusion Matrices
# ================================
def plot_cm(cm, title, path):
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.imshow(cm, interpolation="nearest")
    ax.set_title(title)
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["Pred 0", "Pred 1"])
    ax.set_yticklabels(["True 0", "True 1"])
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, cm[i, j], ha="center", va="center")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    plt.tight_layout()
    plt.savefig(path)
    plt.close()

plot_cm(confusion_matrix(y_test, y_pred_lr), "Confusion Matrix - Logistic Regression", OUT_DIR / "cm_lr.png")
plot_cm(confusion_matrix(y_test, y_pred_rf), "Confusion Matrix - RF Default", OUT_DIR / "cm_rf_default.png")
plot_cm(confusion_matrix(y_test, y_pred_rft), "Confusion Matrix - RF Tuned", OUT_DIR / "cm_rf_tuned.png")

# ================================
# ROC Curves
# ================================
fig, ax = plt.subplots(figsize=(6, 5))
fpr_lr, tpr_lr, _ = roc_curve(y_test, y_proba_lr)
fpr_rf, tpr_rf, _ = roc_curve(y_test, y_proba_rf)
fpr_rft, tpr_rft, _ = roc_curve(y_test, y_proba_rft)
ax.plot(fpr_lr, tpr_lr, label=f"LogReg (AUC={metrics_lr['roc_auc']:.3f})")
ax.plot(fpr_rf, tpr_rf, label=f"RF Default (AUC={metrics_rf_default['roc_auc']:.3f})")
ax.plot(fpr_rft, tpr_rft, label=f"RF Tuned (AUC={metrics_rf_tuned['roc_auc']:.3f})")
ax.plot([0, 1], [0, 1], linestyle="--")
ax.set_xlabel("False Positive Rate")
ax.set_ylabel("True Positive Rate")
ax.set_title("ROC Curves")
ax.legend()
plt.tight_layout()
plt.savefig(OUT_DIR / "roc_curves.png")
plt.close()

# ================================
# Feature Importance (RF tuned)
# ================================
fi = pd.Series(rf_tuned.feature_importances_, index=X_cols).sort_values(ascending=False).head(20)
fi.to_csv(OUT_DIR / "rf_tuned_top20_feature_importances.csv", header=["importance"])
plt.figure(figsize=(8, 5))
fi.sort_values().plot(kind="barh")
plt.title("Top 20 RF (light-tuned) Feature Importances")
plt.tight_layout()
plt.savefig(OUT_DIR / "rf_tuned_top20_feature_importances.png")
plt.close()

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, roc_curve
)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.dummy import DummyClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
import joblib

# ---------- Load ----------
PATH = "feature_set_1.csv"   # adjust if needed
df = pd.read_csv(PATH)

target_col = "Top_Grossing_Flag"
drop_cols = ["Game", "Publisher", "Era", "Units_Sold"]  # text + leakage

num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
X_cols = [c for c in num_cols if c not in [target_col] and c not in drop_cols]

X = df[X_cols]
y = df[target_col].astype(int)

# ---------- Split ----------
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

# ---------- Dummy ----------
dummy = DummyClassifier(strategy="most_frequent", random_state=42)
dummy.fit(X_train, y_train)
y_pred_dummy = dummy.predict(X_test)
metrics_dummy = evaluate("Dummy (most_frequent)", y_test, y_pred_dummy)

# ---------- Logistic Regression ----------
logreg = Pipeline([
    ("scaler", StandardScaler(with_mean=False)),
    ("clf", LogisticRegression(max_iter=2000, n_jobs=-1))
])
logreg.fit(X_train, y_train)
y_pred_lr = logreg.predict(X_test)
y_proba_lr = logreg.predict_proba(X_test)[:, 1]
metrics_lr = evaluate("Logistic Regression", y_test, y_pred_lr, y_proba_lr)

# ---------- Random Forest ----------
rf = RandomForestClassifier(
    n_estimators=300,
    random_state=42,
    n_jobs=-1,
    class_weight="balanced_subsample"
)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)
y_proba_rf = rf.predict_proba(X_test)[:, 1]
metrics_rf = evaluate("Random Forest", y_test, y_pred_rf, y_proba_rf)

# ---------- Compare ----------
metrics_df = pd.DataFrame([metrics_dummy, metrics_lr, metrics_rf]).sort_values("f1", ascending=False)
print(metrics_df)

# ---------- Save best ----------
best_name = metrics_df.iloc[0]["model"]
best_model = rf if best_name == "Random Forest" else logreg
joblib.dump(best_model, "best_baseline_model.pkl")
metrics_df.to_csv("baseline_metrics.csv", index=False)
print(f"\nBest model: {best_name} → saved to best_baseline_model.pkl")

# ---------- Confusion matrices ----------
def plot_cm(cm, title, path):
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.imshow(cm, interpolation="nearest")
    ax.set_title(title)
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
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

plot_cm(confusion_matrix(y_test, y_pred_lr), "Confusion Matrix - Logistic Regression", "cm_lr.png")
plot_cm(confusion_matrix(y_test, y_pred_rf), "Confusion Matrix - Random Forest", "cm_rf.png")

# ---------- ROC curves ----------
fig, ax = plt.subplots(figsize=(6, 5))
fpr_lr, tpr_lr, _ = roc_curve(y_test, y_proba_lr)
fpr_rf, tpr_rf, _ = roc_curve(y_test, y_proba_rf)
ax.plot(fpr_lr, tpr_lr, label=f"LogReg (AUC={metrics_lr['roc_auc']:.3f})")
ax.plot(fpr_rf, tpr_rf, label=f"RF (AUC={metrics_rf['roc_auc']:.3f})")
ax.plot([0, 1], [0, 1], linestyle="--")
ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
ax.set_title("ROC Curves")
ax.legend()
plt.tight_layout()
plt.savefig("roc_curves.png")
plt.close()

# ---------- RF Feature importance ----------
fi = pd.Series(rf.feature_importances_, index=X_cols).sort_values(ascending=False).head(20)
print("\nTop 20 RF feature importances:")
print(fi)
fi.to_csv("rf_top20_feature_importances.csv", header=["importance"])

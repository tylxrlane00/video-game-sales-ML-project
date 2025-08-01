# final_model.py
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
)

# Create output folder
os.makedirs("app/model", exist_ok=True)

# Load dataset
df = pd.read_csv("./data/feature_set_1.csv")

# Drop rows with missing values
df = df.dropna()
df = df[df["Top_Grossing_Flag"].isin([0, 1])]

# Define features and target
target = "Top_Grossing_Flag"
X = df.drop(columns=[target])
y = df[target]

X = X.drop(columns=["Game", "Publisher", "Rank", "Year"], errors="ignore")
X = X.select_dtypes(include=["number"])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

clf = RandomForestClassifier(random_state=42)
param_grid = {
    "n_estimators": [100],
    "min_samples_split": [2, 5],
    "max_depth": [None, 10, 20],
}
grid = GridSearchCV(clf, param_grid, scoring="f1", cv=3, n_jobs=-1)
grid.fit(X_train, y_train)
best_model = grid.best_estimator_

# Save model with joblib
joblib.dump(best_model, "app/model/best_final_model.pkl")

# Evaluation outputs
y_pred = best_model.predict(X_test)
y_proba = best_model.predict_proba(X_test)[:, 1]

report_df = pd.DataFrame(classification_report(y_test, y_pred, output_dict=True)).T
report_df.to_csv("final_output/final_metrics.csv")

conf_matrix = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 5))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="viridis")
plt.title("Final Model Confusion Matrix")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.tight_layout()
plt.savefig("final_output/final_confusion_matrix.png")
plt.close()

try:
    auc = roc_auc_score(y_test, y_proba)
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    plt.figure()
    plt.plot(fpr, tpr, label=f"ROC-AUC: {auc:.2f}")
    plt.plot([0, 1], [0, 1], linestyle="--", color="orange")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Final Model ROC Curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig("final_output/final_roc_curve.png")
    plt.close()
except Exception as e:
    print("ROC Curve could not be generated:", e)

importances = pd.Series(best_model.feature_importances_, index=X.columns)
top_features = importances.sort_values(ascending=False).head(20)
top_features.to_csv("final_output/final_feature_importances.csv")

plt.figure(figsize=(10, 5))
sns.barplot(x=top_features.values, y=top_features.index)
plt.title("Top 20 Feature Importances")
plt.xlabel("Importance")
plt.ylabel("Feature")
plt.tight_layout()
plt.savefig("final_output/final_feature_importances.png")
plt.close()

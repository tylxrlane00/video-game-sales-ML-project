#!/usr/bin/env python
"""
predict.py

Run predictions with a trained model and produce readable outputs:
- predictions.csv and predictions.xlsx
- probability_distribution.png
- top_n_predictions.png
- confusion_matrix.png

It also aligns incoming data columns to the training schema so you can score
datasets that follow the same feature engineering pipeline.
"""

import argparse
import joblib
import pandas as pd
import numpy as np
from pathlib import Path

# Use a headless backend for matplotlib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)

TARGET_COL = "Top_Grossing_Flag"
DROP_COLS = ["Game", "Publisher", "Era", "Units_Sold"]  # matches training scripts


def parse_args():
    p = argparse.ArgumentParser(description="Score data with a trained model and generate readable outputs & charts.")
    p.add_argument("--model", type=str, default="outputs_iter1/best_iter1_model.pkl",
                   help="Path to the trained .pkl model.")
    p.add_argument("--data", type=str, default="feature_set_1.csv",
                   help="CSV to score (must already be feature-engineered like feature_set_1.csv).")
    p.add_argument("--train-feature-set", type=str, default="feature_set_1.csv",
                   help="CSV used to derive the original training schema (columns).")
    p.add_argument("--output-dir", type=str, default="predictions_out",
                   help="Directory to write predictions and charts.")
    p.add_argument("--output-base", type=str, default="predictions",
                   help="Base filename for outputs (without extension).")
    p.add_argument("--threshold", type=float, default=0.5,
                   help="Decision threshold for the positive class.")
    p.add_argument("--top-n", type=int, default=20,
                   help="How many top predictions to plot in the bar chart.")
    return p.parse_args()


def build_feature_list(train_df: pd.DataFrame) -> list:
    """Recreate the same X_cols used during training."""
    num_cols = train_df.select_dtypes(include=[np.number]).columns.tolist()
    X_cols = [c for c in num_cols if c not in [TARGET_COL] + DROP_COLS]
    return X_cols


def align_to_schema(df: pd.DataFrame, X_cols: list) -> pd.DataFrame:
    """Reorder/fill missing columns to match training schema."""
    return df.reindex(columns=X_cols, fill_value=0)


def evaluate_if_possible(y_true, y_prob, y_pred):
    if y_true is None:
        return None

    roc_auc = None
    if y_prob is not None:
        try:
            roc_auc = roc_auc_score(y_true, y_prob)
        except Exception:
            roc_auc = None

    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "roc_auc": roc_auc
    }


def plot_probability_distribution(y_prob, out_path):
    plt.figure(figsize=(7, 4))
    plt.hist(y_prob, bins=30)
    plt.title("Predicted Probability Distribution")
    plt.xlabel("Predicted probability of Top-Grossing (class = 1)")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def plot_top_n(y_prob, games, top_n, out_path):
    # If we don't have probabilities, skip
    if y_prob is None:
        return
    probs = pd.Series(y_prob, index=games if games is not None else range(len(y_prob)))
    top = probs.sort_values(ascending=False).head(top_n)

    plt.figure(figsize=(8, 6))
    top.sort_values().plot(kind="barh")
    plt.title(f"Top {len(top)} Games by Predicted Probability")
    plt.xlabel("Predicted probability of Top-Grossing")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def plot_confusion_matrix(y_true, y_pred, out_path):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.imshow(cm, interpolation="nearest")
    ax.set_title("Confusion Matrix")
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
    plt.savefig(out_path)
    plt.close()


def main():
    args = parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load model
    print(f"Loading model from: {args.model}")
    model = joblib.load(args.model)

    # Load scoring data
    print(f"Loading data to score from: {args.data}")
    df_score = pd.read_csv(args.data)

    # Load training feature set to get schema
    print(f"Loading training schema from: {args.train_feature_set}")
    df_train = pd.read_csv(args.train_feature_set)
    X_cols = build_feature_list(df_train)

    # Extract y_true if available
    y_true = df_score[TARGET_COL].astype(int) if TARGET_COL in df_score.columns else None

    # Align features
    X_score = align_to_schema(df_score, X_cols)

    # Predict
    has_proba = hasattr(model, "predict_proba")
    y_prob = model.predict_proba(X_score)[:, 1] if has_proba else None
    if y_prob is not None:
        y_pred = (y_prob >= args.threshold).astype(int)
    else:
        y_pred = model.predict(X_score)

    # Evaluate (if labels exist)
    metrics = evaluate_if_possible(y_true, y_prob, y_pred)
    if metrics is not None:
        print("\n=== Evaluation on provided data ===")
        for k, v in metrics.items():
            if v is not None:
                print(f"{k:>10}: {v:.4f}")
        print("\nClassification report:")
        print(classification_report(y_true, y_pred, zero_division=0))

    # Build readable output
    keep_cols = ["Game", "Publisher", "Year", "Global_Sales"]
    cols_to_take = [c for c in keep_cols if c in df_score.columns]
    out_df = df_score[cols_to_take].copy() if cols_to_take else pd.DataFrame(index=df_score.index)

    out_df["pred_class"] = y_pred
    if y_prob is not None:
        out_df["pred_prob"] = y_prob
    if y_true is not None:
        out_df["true_label"] = y_true.values

    # Human-friendly label
    out_df["Prediction_Label"] = out_df["pred_class"].map({1: "Top-Grossing", 0: "Not Top-Grossing"})

    # Sort by probability if available
    if "pred_prob" in out_df.columns:
        out_df = out_df.sort_values(by="pred_prob", ascending=False)

    # Save CSV & Excel
    csv_path = out_dir / f"{args.output_base}.csv"
    xlsx_path = out_dir / f"{args.output_base}.xlsx"
    out_df.to_csv(csv_path, index=False)
    try:
        out_df.to_excel(xlsx_path, index=False)
    except Exception as e:
        print(f"Could not write Excel file (missing openpyxl?): {e}")

    print(f"\nPredictions saved to:\n- {csv_path}\n- {xlsx_path if xlsx_path.exists() else '(Excel not written)'}")

    # Charts
    if y_prob is not None:
        prob_dist_path = out_dir / "probability_distribution.png"
        plot_probability_distribution(y_prob, prob_dist_path)
        print(f"Saved: {prob_dist_path}")

        games = df_score["Game"] if "Game" in df_score.columns else None
        top_n_path = out_dir / f"top_{args.top_n}_predictions.png"
        plot_top_n(y_prob, games, args.top_n, top_n_path)
        print(f"Saved: {top_n_path}")

    if y_true is not None:
        cm_path = out_dir / "confusion_matrix.png"
        plot_confusion_matrix(y_true, y_pred, cm_path)
        print(f"Saved: {cm_path}")


if __name__ == "__main__":
    main()

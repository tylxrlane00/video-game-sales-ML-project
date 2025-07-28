#!/usr/bin/env python
"""
predict.py

Use a previously trained model (e.g., best_iter1_model.pkl) to:
 - Predict on the original feature_set_1.csv (to sanity check)
 - Predict on NEW data that already has the same engineered columns
 - Optionally evaluate if the target column is present

It aligns columns to the training schema so you don't have to worry about
column ordering or missing dummy columns (they will be filled with 0s).
"""

import argparse
import joblib
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)

TARGET_COL = "Top_Grossing_Flag"
DROP_COLS = ["Game", "Publisher", "Era", "Units_Sold"]  # same as training

def parse_args():
    p = argparse.ArgumentParser(description="Score data with a trained model.")
    p.add_argument("--model", type=str, default="outputs_iter1/best_iter1_model.pkl",
                   help="Path to the trained .pkl model.")
    p.add_argument("--data", type=str, default="feature_set_1.csv",
                   help="CSV with rows to score (can be your training feature set or new data).")
    p.add_argument("--train-feature-set", type=str, default="feature_set_1.csv",
                   help="CSV used to derive the original training schema (columns). "
                        "Use the same file you used in training (feature_set_1.csv).")
    p.add_argument("--output", type=str, default="predictions.csv",
                   help="Where to save predictions.")
    p.add_argument("--threshold", type=float, default=0.5,
                   help="Decision threshold for positive class (only used if model outputs probabilities).")
    return p.parse_args()

def build_feature_list(train_df: pd.DataFrame) -> list:
    """Recreate the same X_cols used during training."""
    num_cols = train_df.select_dtypes(include=[np.number]).columns.tolist()
    X_cols = [c for c in num_cols if c not in [TARGET_COL] + DROP_COLS]
    return X_cols

def align_to_schema(df: pd.DataFrame, X_cols: list) -> pd.DataFrame:
    """
    Reorder and fill any missing columns to match the training schema.
    Extra columns in df are dropped.
    """
    # Keep only columns in schema
    df_aligned = df.reindex(columns=X_cols, fill_value=0)
    return df_aligned

def evaluate_if_possible(y_true, y_prob, y_pred):
    """Return a dict of metrics (if y_true provided)."""
    if y_true is None:
        return None

    # Some models may not expose predict_proba (e.g., SVM w/o probability)
    roc_auc = None
    if y_prob is not None:
        try:
            roc_auc = roc_auc_score(y_true, y_prob)
        except Exception:
            roc_auc = None

    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "roc_auc": roc_auc
    }
    return metrics

def main():
    args = parse_args()
    model_path = Path(args.model)
    data_path = Path(args.data)
    train_fs_path = Path(args.train_feature_set)
    out_path = Path(args.output)

    # --- Load model ---
    print(f"Loading model from: {model_path}")
    model = joblib.load(model_path)

    # --- Load data to score ---
    print(f"Loading data to score from: {data_path}")
    df_score = pd.read_csv(data_path)

    # --- Load training feature set to recreate schema ---
    print(f"Loading training feature set schema from: {train_fs_path}")
    df_train = pd.read_csv(train_fs_path)
    X_cols = build_feature_list(df_train)

    # --- Separate target if present (for evaluation only) ---
    y_true = None
    if TARGET_COL in df_score.columns:
        y_true = df_score[TARGET_COL].astype(int)
        print(f"Found target column '{TARGET_COL}' in scoring data. Will evaluate.")

    # --- Align columns ---
    X_score = align_to_schema(df_score, X_cols)

    # --- Predict ---
    has_proba = hasattr(model, "predict_proba")
    y_prob = model.predict_proba(X_score)[:, 1] if has_proba else None
    # If you want threshold tuning, use args.threshold here:
    if y_prob is not None:
        y_pred = (y_prob >= args.threshold).astype(int)
    else:
        y_pred = model.predict(X_score)

    # --- Evaluate if possible ---
    metrics = evaluate_if_possible(y_true, y_prob, y_pred)
    if metrics is not None:
        print("\n=== Evaluation on provided data ===")
        for k, v in metrics.items():
            if v is not None:
                print(f"{k:>10}: {v:.4f}")
        print("\nConfusion matrix:")
        print(confusion_matrix(y_true, y_pred))
        print("\nClassification report:")
        print(classification_report(y_true, y_pred, zero_division=0))

    # --- Save predictions ---
    out_df = pd.DataFrame({
        "pred_class": y_pred
    })
    if y_prob is not None:
        out_df["pred_prob"] = y_prob
    if y_true is not None:
        out_df["true_label"] = y_true.values

    # Add an identifier if available (e.g., Game)
    if "Game" in df_score.columns:
        out_df["Game"] = df_score["Game"]

    out_df.to_csv(out_path, index=False)
    print(f"\nPredictions saved to: {out_path.resolve()}")

if __name__ == "__main__":
    main()

# app/app.py

import os
import pandas as pd
from flask import Flask, request, jsonify, render_template, send_file
from joblib import load

app = Flask(__name__)

# Load model with joblib
model_path = os.path.join(os.path.dirname(__file__), 'model', 'best_final_model.pkl')
model = load(model_path)

@app.route("/")
def home():
    return render_template('index.html')

@app.route("/predict", methods=["POST"])
def predict():
    try:
        if 'file' not in request.files:
            return "No file part", 400

        file = request.files['file']
        if file.filename == "":
            return "No selected file", 400

        df = pd.read_csv(file)

        # Match training-time preprocessing
        df = df.drop(columns=["Game", "Publisher", "Rank", "Year"], errors="ignore")
        df = df.select_dtypes(include=["number"])

        # Make sure columns match exactly (same order, no extras or missing)
        expected_features = model.feature_names_in_
        df = df[expected_features]

        preds = model.predict(df)
        probas = model.predict_proba(df)[:, 1]

        df_results = df.copy()
        df_results["Prediction"] = preds
        df_results["Probability_Top_Grossing"] = probas.round(4)

        output_path = "final_output/predictions.csv"
        df_results.to_csv(output_path, index=False)

        return send_file(output_path, as_attachment=True)

    except Exception as e:
        return f"An error occurred: {str(e)}", 500

    
@app.route("/upload", methods=["POST"])
def upload():
    try:
        if "file" not in request.files:
            return "No file part", 400

        file = request.files["file"]

        if file.filename == "":
            return "No selected file", 400

        # Read CSV into a DataFrame
        df = pd.read_csv(file)

        # Make predictions
        predictions = model.predict(df)
        probabilities = model.predict_proba(df)[:, 1]

        # Add predictions to DataFrame
        df["Prediction"] = predictions
        df["Probability_Top_Grossing"] = probabilities

        # Show result as HTML table
        return render_template("results.html", tables=[df.to_html(classes="table table-bordered", index=False)], titles=df.columns.values)

    except Exception as e:
        return f"An error occurred: {e}", 500


if __name__ == "__main__":
    app.run(debug=True)

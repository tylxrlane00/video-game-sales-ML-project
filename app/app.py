# app/app.py

import os
import pandas as pd
from flask import Flask, request, render_template, send_file
from joblib import load

app = Flask(__name__)

# Load model
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

        # Remove non-numeric and identifier columns
        df = df.select_dtypes(include=["number"])
        expected_features = model.feature_names_in_

        # Keep only expected features
        df = df[[col for col in expected_features if col in df.columns]]

        # Check for missing columns
        missing_cols = set(expected_features) - set(df.columns)
        if missing_cols:
            return f"Missing required features: {', '.join(missing_cols)}", 400

        # Ensure correct column order
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

if __name__ == "__main__":
    app.run(debug=True)

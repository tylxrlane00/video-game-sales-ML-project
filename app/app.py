# app.py

import os
import uuid
import pandas as pd
import matplotlib.pyplot as plt
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

        # Preprocess
        df = df.select_dtypes(include=["number"])
        expected_features = model.feature_names_in_

        df = df[[col for col in expected_features if col in df.columns]]
        missing_cols = set(expected_features) - set(df.columns)
        if missing_cols:
            return f"Missing required features: {', '.join(missing_cols)}", 400
        df = df[expected_features]

        preds = model.predict(df)
        probas = model.predict_proba(df)[:, 1]

        df_results = df.copy()
        df_results["Prediction"] = preds
        df_results["Probability_Top_Grossing"] = probas.round(4)

        # Save predictions
        os.makedirs("final_output", exist_ok=True)
        output_filename = f"final_output/predictions_{uuid.uuid4().hex}.csv"
        df_results.to_csv(output_filename, index=False)

        # Chart
        prediction_counts = df_results["Prediction"].value_counts().sort_index()
        labels = ["Not Top Grossing", "Top Grossing"]
        counts = [prediction_counts.get(0, 0), prediction_counts.get(1, 0)]

        plt.figure(figsize=(6, 4))
        plt.bar(labels, counts, color=["gray", "green"])
        plt.title("Prediction Breakdown")
        plt.ylabel("Number of Games")
        plt.tight_layout()

        os.makedirs("static", exist_ok=True)
        chart_path = f"static/chart_{uuid.uuid4().hex}.png"
        plt.savefig(chart_path)
        plt.close()

        return render_template("results.html",
                               tables=[df_results.head().to_html(classes="table table-bordered", index=False)],
                               titles=df_results.columns.values,
                               chart_url=chart_path,
                               download_link=output_filename)

    except Exception as e:
        return f"An error occurred: {str(e)}", 500

if __name__ == "__main__":
    app.run(debug=True)

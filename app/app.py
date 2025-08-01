# app/app.py

import os
import pandas as pd
from flask import Flask, request, jsonify, render_template
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
        data = request.get_json()

        if not data:
            return jsonify({"error": "No input data provided"}), 400

        df = pd.DataFrame([data])  # wrap in list to handle single record
        prediction = model.predict(df)[0]
        prob = model.predict_proba(df)[0][1]  # probability of class 1 (top grossing)

        return jsonify({
            "prediction": int(prediction),
            "probability_top_grossing": round(prob, 4)
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500
    
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

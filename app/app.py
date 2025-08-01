# app/app.py

import os
import pickle
import pandas as pd
from flask import Flask, request, jsonify
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)

# Load model
model_path = os.path.join(os.path.dirname(__file__), 'model', 'best_final_model.pkl')
with open(model_path, 'rb') as f:
    model = pickle.load(f)

@app.route("/")
def home():
    return "Top Grossing Game Predictor API is running!"

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

if __name__ == "__main__":
    app.run(debug=True)

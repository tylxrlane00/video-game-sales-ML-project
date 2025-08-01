# 🎮 Video Game Sales ML Project

This project analyzes video game sales data to predict whether a game will become a top-grossing title. It covers the full machine learning pipeline — from raw data preprocessing and feature engineering to model training, evaluation, and deployment via a Flask API on Heroku.

---

## 📁 Project Structure

```
VIDEO-GAME-SALES-ML/
├── app/                            # Flask API app
│   ├── app.py                      # Flask backend script
│   └── model/
│       └── best_final_model.pkl    # Trained final model
│
├── data/                           # Raw and engineered datasets
│   ├── vgsales.csv
│   ├── feature_set_1.csv
│   ├── feature_set_2.csv
│   └── merged_features.csv
│
├── scripts/                        # Data prep and model training
│   ├── dataPrep_vgsales.py
│   ├── dataPrep_topgrossing.py
│   ├── dataset2_features.py
│   ├── merged_features.py
│   ├── baseline.py
│   ├── model_iter1.py
│   ├── final_model.py
│   └── predict.py
│
├── outputs/                        # Model outputs and visualizations
│   ├── final_output/
│   │   ├── final_roc_curve.png
│   │   ├── final_confusion_matrix.png
│   │   ├── final_feature_importances.png
│   │   ├── final_metrics.csv
│   │   └── final_feature_importances.csv
│   ├── outputs_iter1/
│   └── baseline-model-results/
│
├── predictions_out/               # Model prediction files
│
├── requirements.txt
├── Procfile
├── runtime.txt
├── readme.md
```

---

## ⚙️ Key Scripts

| Script | Purpose |
|--------|---------|
| `dataPrep_vgsales.py` | Cleans the VGChartz sales dataset. |
| `dataPrep_topgrossing.py` | Cleans the top-grossing games dataset. |
| `merged_features.py` | Merges cleaned datasets into a training-ready format. |
| `dataset2_features.py` | Optional feature engineering script for secondary dataset. |
| `baseline.py` | Trains initial baseline models (e.g., Logistic Regression, Random Forest). |
| `model_iter1.py` | Improves the baseline model using class balancing and hyperparameter tuning. |
| `final_model.py` | Final training pipeline with best feature set and export of evaluation metrics and visuals. |
| `predict.py` | Loads any trained model to make predictions on new data. |

---

## 🧪 How to Run the Project

### 1. Install Python Dependencies

```bash
pip install -r requirements.txt
```

### 2. Run Data Preparation Scripts

```bash
python scripts/dataPrep_vgsales.py
python scripts/dataPrep_topgrossing.py
python scripts/merged_features.py
```

### 3. Train Final Model

```bash
python scripts/final_model.py
```

### 4. Generate Predictions (Optional)

```bash
python scripts/predict.py --model app/model/best_final_model.pkl --data data/feature_set_1.csv
```

---

## 🚀 Deploy the MVP Flask API

To serve predictions using a Flask API and deploy to Heroku:

1. Ensure the `app/` folder contains:
   - `app.py`
   - `model/best_final_model.pkl`

2. From the project root:

```bash
heroku login
heroku create video-game-sales-mvp
git add .
git commit -m "Deploy MVP to Heroku"
git push heroku main
```

Heroku will use `requirements.txt`, `Procfile`, and `runtime.txt` to build and run the Flask app.

---

## 📊 Outputs

Evaluation metrics and visualizations for the final model are saved to:
```
outputs/final_output/
├── final_roc_curve.png
├── final_confusion_matrix.png
├── final_feature_importances.png
├── final_metrics.csv
├── final_feature_importances.csv
```

---

## 📌 Data Sources

All base data was retrieved from [VGChartz](https://www.vgchartz.com/) and cleaned for academic research purposes. Reference datasets were provided by:  
https://github.com/mfun06/ACS5513Data


---

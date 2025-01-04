# Car Resale Price Prediction Project

## Project Description
This project predicts car resale prices using Granger causality for feature selection and advanced ML techniques.

## Project Structure
- `data/`: Contains raw data files.
- `scripts/`: Contains scripts for data preprocessing, feature engineering, model training, and evaluation.
- `outputs/`: Stores generated outputs like preprocessed data, model performance reports, and plots.
- `app/`: Flask application for model deployment.
- `README.md`: Project documentation.
- `requirements.txt`: Python dependencies.

## Steps to Run the Project
1. Install dependencies: `pip install -r requirements.txt`
2. Preprocess the data: `python scripts/preprocess_data.py`
3. Engineer features: `python scripts/feature_engineering.py`
4. Train models:
   - Baseline: `python scripts/train_baseline_model.py`
   - LSTM: `python scripts/train_lstm_model.py`
5. Optimize and explain: `python scripts/optimize_and_explain.py`
6. Evaluate models: `python scripts/evaluate_model.py`
7. Deploy the model: `python app/app.py`

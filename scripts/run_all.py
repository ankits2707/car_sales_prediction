import os

os.system("python preprocess_data.py")
os.system("python feature_engineering.py")
os.system("python train_baseline_model.py")
os.system("python train_lstm_model.py")
os.system("python optimize_and_explain.py")
os.system("python evaluate_model.py")
os.system("python ../app/app.py")

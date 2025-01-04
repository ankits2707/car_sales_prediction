import pandas as pd
import joblib  # For loading the model
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np  # For square root calculation

# Load the saved model
best_model = joblib.load('../outputs/best_model.pkl')

# Load data
data = pd.read_csv('../outputs/feature_engineered_data.csv')
X = data[['mileage_lag1', 'brand', 'engine_size']]
y = data['price']

# Predictions
y_pred = best_model.predict(X)

# Evaluate
print("Final Model Performance:")
print("MAE:", mean_absolute_error(y, y_pred))

# Calculate RMSE
mse = mean_squared_error(y, y_pred)
rmse = np.sqrt(mse)
print("RMSE:", rmse)

print("R^2:", r2_score(y, y_pred))

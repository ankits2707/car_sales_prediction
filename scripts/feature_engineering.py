import pandas as pd
from statsmodels.tsa.stattools import grangercausalitytests

# Load preprocessed data
data = pd.read_csv('../outputs/preprocessed_data.csv')

# Feature Engineering: Add lagged variables
data['mileage_lag1'] = data['mileage'].shift(1)
data['price_lag1'] = data['price'].shift(1)

# Drop NaN rows caused by lagging
data.dropna(inplace=True)

# Check the shape of the dataset to determine the maximum allowable lag
print(f"Data Shape: {data.shape}")

# Granger Causality Analysis
print("Performing Granger Causality Tests:")
max_lag = 1  # Adjust this based on your data size
grangercausalitytests(data[['price', 'mileage_lag1']], maxlag=max_lag)

# Save feature-engineered dataset
data.to_csv('../outputs/feature_engineered_data.csv', index=False)
print("Feature engineering complete. Data saved to 'outputs/feature_engineered_data.csv'")

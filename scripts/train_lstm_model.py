import pandas as pd
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.model_selection import train_test_split

# Load data
data = pd.read_csv('../outputs/feature_engineered_data.csv')
X = data[['mileage_lag1', 'brand', 'engine_size']].values
y = data['price'].values

# Reshape for LSTM
X = X.reshape(X.shape[0], 1, X.shape[1])

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define LSTM model
model = Sequential([
    LSTM(50, activation='relu', input_shape=(1, X.shape[2])),
    Dense(1)
])
model.compile(optimizer='adam', loss='mean_squared_error')

# Train model
model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test))

# Save model
model.save('../outputs/lstm_model.h5')
print("LSTM model training complete and saved to 'outputs/lstm_model.h5'")

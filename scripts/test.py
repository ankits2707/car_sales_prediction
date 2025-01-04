import joblib
import numpy as np

# Load the model, encoder, and scaler
model = joblib.load('../outputs/best_model.pkl')
brand_encoder = joblib.load('../outputs/encoder.pkl')
scaler = joblib.load('../outputs/scaler.pkl')

# Example input
mileage = 50000
engine_size = 2.0
year = 2018
brand = 'Honda'

# Encode brand
brand_encoded = brand_encoder.transform([brand])[0]

# Normalize numerical features
input_data = np.array([mileage, engine_size, year]).reshape(1, -1)
input_data = scaler.transform(input_data)

# Final input for prediction
final_input = np.concatenate([input_data, np.array([[brand_encoded]])], axis=1)

# Make prediction
prediction = model.predict(final_input)
print(f"Predicted Price: {prediction[0]}")

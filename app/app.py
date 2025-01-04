# app.py

from flask import Flask, request, jsonify
import numpy as np
import joblib
from sklearn.preprocessing import LabelEncoder

app = Flask(__name__)

# Load your pre-trained model (assuming it's saved in a 'model.pkl' file)
model = joblib.load('../outputs/best_model.pkl')

# Initialize a label encoder for the 'brand' feature
brand_encoder = LabelEncoder()

# You need to fit the encoder on the list of possible brands during training
# Example: brand_encoder.fit(['Toyota', 'Ford', 'Honda']) - fit it on your training data
brand_encoder.fit(['Toyota', 'Ford', 'Honda'])  # Example fitting

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json

    # Ensure that mileage and engine_size are numeric
    mileage = float(data['mileage'])
    engine_size = float(data['engine_size'])

    # Handle brand encoding (convert string to numeric)
    brand = brand_encoder.transform([data['brand']])[0]  # Convert brand to numeric

    # Construct input data array
    input_data = np.array([mileage, engine_size, brand]).reshape(1, -1)

    # Make prediction
    prediction = model.predict(input_data)

    # Return the prediction as a JSON response
    return jsonify({'predicted_price': prediction[0]})

if __name__ == '__main__':
    app.run(debug=True)

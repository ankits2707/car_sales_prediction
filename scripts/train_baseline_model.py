import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import joblib

# Load the dataset
data = pd.read_csv('../data/car_resale_data.csv')

# Train the encoder for 'brand' column
brand_encoder = LabelEncoder()
brand_encoder.fit(data['brand'])

# Save the encoder
joblib.dump(brand_encoder, '../outputs/encoder.pkl')

# Prepare the features and target
X = data[['mileage', 'engine_size', 'year', 'brand']]
y = data['price']

# Encode the brand column
X['brand'] = brand_encoder.transform(X['brand'])

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the numerical features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train[['mileage', 'engine_size', 'year']])
X_test_scaled = scaler.transform(X_test[['mileage', 'engine_size', 'year']])

# Include the encoded brand in the final input
X_train_final = np.concatenate([X_train_scaled, X_train[['brand']].values], axis=1)
X_test_final = np.concatenate([X_test_scaled, X_test[['brand']].values], axis=1)

# Train the model
model = LinearRegression()
model.fit(X_train_final, y_train)

# Save the trained model and scaler
joblib.dump(model, '../outputs/baseline_model.pkl')
joblib.dump(scaler, '../outputs/scaler.pkl')

# Print model accuracy
print("Model training complete. Accuracy on test set:", model.score(X_test_final, y_test))

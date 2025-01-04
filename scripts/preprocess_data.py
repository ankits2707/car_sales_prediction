import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer

# Load dataset
data = pd.read_csv('../data/car_resale_data.csv')

# Handle missing values
imputer = SimpleImputer(strategy='mean')
data.loc[:, data.select_dtypes(include=[np.number]).columns] = imputer.fit_transform(data.select_dtypes(include=[np.number]))


# Encode categorical variables
encoder = LabelEncoder()
data['brand'] = encoder.fit_transform(data['brand'])

# Normalize numerical variables
scaler = StandardScaler()
numeric_cols = ['mileage', 'engine_size', 'year']  
data[numeric_cols] = scaler.fit_transform(data[numeric_cols])

# Save preprocessed data
data.to_csv('../outputs/preprocessed_data.csv', index=False)
print("Preprocessing complete. Data saved to 'outputs/preprocessed_data.csv'")

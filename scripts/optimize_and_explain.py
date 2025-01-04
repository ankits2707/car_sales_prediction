import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import GridSearchCV
import shap
import joblib  # For saving the model

# Load data
data = pd.read_csv('../outputs/feature_engineered_data.csv')
X = data[['mileage_lag1', 'brand', 'engine_size']]
y = data['price']

# Hyperparameter tuning
param_grid = {'n_estimators': [50, 100], 'learning_rate': [0.01, 0.1], 'max_depth': [3, 5]}
grid_search = GridSearchCV(GradientBoostingRegressor(), param_grid, cv=3)
grid_search.fit(X, y)
best_model = grid_search.best_estimator_

# Save the model
joblib.dump(best_model, '../outputs/best_model.pkl')

# SHAP explainability
explainer = shap.TreeExplainer(best_model)
shap_values = explainer.shap_values(X)

# Save SHAP plot
shap.summary_plot(shap_values, X, show=False)
import matplotlib.pyplot as plt
plt.savefig('../outputs/shap_summary_plot.png')
print("Hyperparameter tuning and SHAP explainability complete. Plot saved to 'outputs/shap_summary_plot.png'")

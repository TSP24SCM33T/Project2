import os
import sys

import numpy as np
from sklearn.ensemble import GradientBoostingRegressor as SklearnGBR
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt

# Get the directory of the current file
current_file_directory = os.path.dirname(os.path.abspath(__file__))
# Add a relative path to sys.path
relative_path = os.path.join(current_file_directory, '..')

print("relative_path", relative_path)
sys.path.append(relative_path)

from model.gradient_boosting import GradientBoosting

print("Test case has started running .....")
print("For some of test cases it can take a while .....")

# Generate a small synthetic dataset
np.random.seed(42)
X_small = np.random.rand(10, 2)  # 10 samples, 2 features
y_small = np.sin(X_small[:, 0]) + np.cos(X_small[:, 1]) + np.random.randn(10) * 0.1  # Target with noise

print(X_small.shape)
print(y_small.shape)

# Split the data into training and testing sets
X_train, X_test = X_small[:8], X_small[8:]
y_train, y_test = y_small[:8], y_small[8:]

# Initialize and fit the custom model
gb = GradientBoosting(
    n_estimators=10,
    learning_rate=0.1,
    max_depth=2,
    min_samples_leaf=1,
    subsample=1.0,
    max_features=None,
    handle_missing='none'
)
gb.fit(X_train, y_train)

# Make predictions with the custom model
y_pred = gb.predict(X_test)
# Initialize and fit scikit-learn's model
sklearn_gb = SklearnGBR(
    n_estimators=10,
    learning_rate=0.1,
    max_depth=2,
    min_samples_leaf=1,
    subsample=1.0,
    max_features=None
)
sklearn_gb.fit(X_train, y_train)

# Make predictions with scikit-learn's model
y_pred_sklearn = sklearn_gb.predict(X_test)


# Evaluate the custom model
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Evaluate scikit-learn's model
mse_sklearn = mean_squared_error(y_test, y_pred_sklearn)
mae_sklearn = mean_absolute_error(y_test, y_pred_sklearn)
r2_sklearn = r2_score(y_test, y_pred_sklearn)

print("Results for Small Dataset:")
print("Custom Gradient Boosting Regressor Metrics:")
print(f"  MSE: {mse:.4f}")
print(f"  MAE: {mae:.4f}")
print(f"  R^2 Score: {r2:.4f}")
print("Scikit-Learn Gradient Boosting Regressor Metrics:")
print(f"  MSE: {mse_sklearn:.4f}")
print(f"  MAE: {mae_sklearn:.4f}")
print(f"  R^2 Score: {r2_sklearn:.4f}")

# Plot predictions vs. true values
plt.figure(figsize=(6, 4))
plt.scatter(range(len(y_test)), y_test, label='True Values')
plt.scatter(range(len(y_pred)), y_pred, label='Custom Model Predictions')
plt.scatter(range(len(y_pred_sklearn)), y_pred_sklearn, label='Scikit-Learn Predictions')
plt.title('Predictions vs. True Values on Small Dataset')
plt.xlabel('Sample Index')
plt.ylabel('Target Value')
plt.legend()
plt.show()


gb.plot_learning_curve("small dataset")

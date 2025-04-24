import os
import sys
import numpy as np
import time
from sklearn.datasets import make_classification
from sklearn.impute import SimpleImputer
from sklearn.ensemble import GradientBoostingClassifier as SklearnGBC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Get the directory of the current file
current_file_directory = os.path.dirname(os.path.abspath(__file__))
# Add a relative path to sys.path
relative_path = os.path.join(current_file_directory, '..')
sys.path.append(relative_path)

from model.gradient_boosting import GradientBoosting

print("Test case for classification with missing values is starting...")
print("This may take a moment...")

# Generate dataset with missing values
np.random.seed(42)
n_samples = 300
n_features = 6
X, y = make_classification(n_samples=n_samples, n_features=n_features, n_informative=4,
                           n_redundant=2, random_state=42, flip_y=0.1)

# Introduce random missing values (10% missing)
missing_mask = np.random.rand(n_samples, n_features) < 0.1
X[missing_mask] = np.nan

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- Custom Gradient Boosting Classifier ---
start_time = time.time()
gb = GradientBoosting(
    loss='logistic',
    n_estimators=50,
    learning_rate=0.1,
    max_depth=3,
    min_samples_leaf=1,
    subsample=1.0,
    max_features=None,
    handle_missing='mean'  # Try 'median' or 'none' too
)
gb.fit(X_train, y_train)
custom_time = time.time() - start_time

y_pred = gb.predict(X_test)
y_proba = gb.predict_proba(X_test)[:, 1]

# --- Scikit-learn Baseline ---
imputer = SimpleImputer(strategy='mean')
X_train_imputed = imputer.fit_transform(X_train)
X_test_imputed = imputer.transform(X_test)

sklearn_gb = SklearnGBC(
    n_estimators=50,
    learning_rate=0.1,
    max_depth=3,
    min_samples_leaf=1,
    subsample=1.0,
    max_features=None
)
sklearn_gb.fit(X_train_imputed, y_train)
y_pred_sklearn = sklearn_gb.predict(X_test_imputed)
y_proba_sklearn = sklearn_gb.predict_proba(X_test_imputed)[:, 1]

# --- Evaluation ---
print("Results for Dataset with Missing Values (Classification):")
print("Custom Gradient Boosting Classifier Metrics:")
print(f"  Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print(f"  Training Time: {custom_time:.2f} seconds")
print("  Classification Report:\n", classification_report(y_test, y_pred))

print("Scikit-Learn Gradient Boosting Classifier Metrics:")
print(f"  Accuracy: {accuracy_score(y_test, y_pred_sklearn):.4f}")
print("  Classification Report:\n", classification_report(y_test, y_pred_sklearn))

# --- Confusion Matrix ---
fig, axs = plt.subplots(1, 2, figsize=(12, 5))
ConfusionMatrixDisplay(confusion_matrix(y_test, y_pred)).plot(ax=axs[0])
axs[0].set_title("Custom GB Confusion Matrix")

ConfusionMatrixDisplay(confusion_matrix(y_test, y_pred_sklearn)).plot(ax=axs[1])
axs[1].set_title("Sklearn GB Confusion Matrix")
plt.tight_layout()
plt.show()

# --- Learning Curve ---
gb.plot_learning_curve("Classification with Missing Values")

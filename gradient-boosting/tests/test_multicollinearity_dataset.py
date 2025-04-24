import os
import sys
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier as SklearnGBC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# Get the directory of the current file
current_file_directory = os.path.dirname(os.path.abspath(__file__))
# Add a relative path to sys.path
relative_path = os.path.join(current_file_directory, '..')
print("relative_path", relative_path)
sys.path.append(relative_path)

from model.gradient_boosting import GradientBoosting

print("Test case has started running .....")
print("Testing multicollinearity with classification .....")

# Generate a dataset with multicollinearity
np.random.seed(42)
n_samples = 300
X_base = np.random.rand(n_samples, 1)
noise = np.random.randn(n_samples, 1) * 0.01

X_multi = np.hstack([
    X_base,
    X_base + noise,      # highly correlated
    X_base - noise,      # highly correlated
    np.random.rand(n_samples, 1)  # uncorrelated
])

# Generate binary target based on base variable
# Use sigmoid activation on a linear function for probabilistic labeling
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

p = sigmoid(5 * X_base.squeeze() - 2)
y_multi = (np.random.rand(n_samples) < p).astype(int)

# Split into train and test
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_multi, y_multi, test_size=0.2, random_state=42)

# --- Custom Gradient Boosting Classifier ---
gb = GradientBoosting(
    loss='logistic',
    n_estimators=50,
    learning_rate=0.1,
    max_depth=3,
    min_samples_leaf=1,
    subsample=1.0,
    max_features=None,
    handle_missing='none'
)
gb.fit(X_train, y_train)
y_pred = gb.predict(X_test)
y_proba = gb.predict_proba(X_test)[:, 1]

# --- Sklearn Baseline ---
sklearn_gb = SklearnGBC(
    n_estimators=50,
    learning_rate=0.1,
    max_depth=3,
    min_samples_leaf=1,
    subsample=1.0,
    max_features=None
)
sklearn_gb.fit(X_train, y_train)
y_pred_sklearn = sklearn_gb.predict(X_test)
y_proba_sklearn = sklearn_gb.predict_proba(X_test)[:, 1]

# --- Evaluation ---
print("Results for Dataset with Multicollinearity (Classification):")
print("Custom Gradient Boosting Classifier Metrics:")
print(f"  Accuracy: {accuracy_score(y_test, y_pred):.4f}")
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
gb.plot_learning_curve("Multicollinearity - Classification")

import os
import sys
import time
import numpy as np

from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.ensemble import GradientBoostingClassifier as SklearnGBC
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay

# Add model path
current_file_directory = os.path.dirname(os.path.abspath(__file__))
relative_path = os.path.join(current_file_directory, '..')
sys.path.append(relative_path)

from model.gradient_boosting import GradientBoosting

print("Test case for classification has started running .....")
print("For some test cases it may take a while .....")

# Generate a high-dimensional binary classification dataset
np.random.seed(42)
n_samples = 400
n_features = 8
X, y = make_classification(n_samples=n_samples, n_features=n_features, n_informative=6,
                           n_redundant=2, random_state=42, flip_y=0.05)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- Custom Gradient Boosting Classifier ---
start_time = time.time()
gb = GradientBoosting(
    loss='logistic',
    n_estimators=100,
    learning_rate=0.1,
    max_depth=5,
    min_samples_leaf=5,
    subsample=0.8,
    max_features='sqrt',
    handle_missing='none'
)
gb.fit(X_train, y_train)
custom_time = time.time() - start_time

y_pred = gb.predict(X_test)
y_proba = gb.predict_proba(X_test)[:, 1]

# --- Sklearn Gradient Boosting Classifier ---
start_time = time.time()
sklearn_gb = SklearnGBC(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=5,
    min_samples_leaf=5,
    subsample=0.8,
    max_features='sqrt'
)
sklearn_gb.fit(X_train, y_train)
sklearn_time = time.time() - start_time

y_pred_sklearn = sklearn_gb.predict(X_test)
y_proba_sklearn = sklearn_gb.predict_proba(X_test)[:, 1]

# --- Evaluation ---
print("\nResults for Binary Classification Task:")
print("Custom Gradient Boosting Classifier:")
print(f"  Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print(f"  Training Time: {custom_time:.2f} seconds")
print("  Classification Report:\n", classification_report(y_test, y_pred))

print("\nScikit-Learn Gradient Boosting Classifier:")
print(f"  Accuracy: {accuracy_score(y_test, y_pred_sklearn):.4f}")
print(f"  Training Time: {sklearn_time:.2f} seconds")
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
gb.plot_learning_curve("Binary Classification (Custom GB)")

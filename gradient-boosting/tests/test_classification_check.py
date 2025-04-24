import os
import sys

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier as SklearnGBC
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay

# Get the directory of the current file
current_file_directory = os.path.dirname(os.path.abspath(__file__))
# Add a relative path to sys.path
relative_path = os.path.join(current_file_directory, '..')

print("relative_path", relative_path)
sys.path.append(relative_path)

from model.gradient_boosting import GradientBoosting

# Generate synthetic binary classification data
X, y = make_classification(n_samples=500, n_features=2, n_informative=2,
                           n_redundant=0, n_clusters_per_class=1, random_state=42)

# Split into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train custom Gradient Boosting
my_gb = GradientBoosting(loss='logistic', n_estimators=100, learning_rate=0.1, max_depth=3)
my_gb.fit(X_train, y_train)
my_preds = my_gb.predict(X_test)
my_proba = my_gb.predict_proba(X_test)[:, 1]

# Train sklearn Gradient Boosting
sk_gb = SklearnGBC(n_estimators=100, learning_rate=0.1, max_depth=3)
sk_gb.fit(X_train, y_train)
sk_preds = sk_gb.predict(X_test)
sk_proba = sk_gb.predict_proba(X_test)[:, 1]

# Accuracy Comparison
print("Custom GB Accuracy:", accuracy_score(y_test, my_preds))
print("Sklearn GB Accuracy:", accuracy_score(y_test, sk_preds))

# Confusion Matrix
fig, axs = plt.subplots(1, 2, figsize=(12, 5))
ConfusionMatrixDisplay(confusion_matrix(y_test, my_preds)).plot(ax=axs[0])
axs[0].set_title("Custom GB Confusion Matrix")

ConfusionMatrixDisplay(confusion_matrix(y_test, sk_preds)).plot(ax=axs[1])
axs[1].set_title("Sklearn GB Confusion Matrix")
plt.tight_layout()
plt.show()

# Decision Boundary
def plot_decision_boundary(model, title, ax):
    h = .02  # step size in the mesh
    x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
    y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    ax.contourf(xx, yy, Z, alpha=0.4)
    ax.scatter(X[:, 0], X[:, 1], c=y, edgecolors='k')
    ax.set_title(title)

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
plot_decision_boundary(my_gb, "Custom Gradient Boosting", axes[0])
plot_decision_boundary(sk_gb, "Sklearn Gradient Boosting", axes[1])
plt.tight_layout()
plt.show()

# Learning Curve (Custom Only)
my_gb.plot_learning_curve("Custom Gradient Boosting")

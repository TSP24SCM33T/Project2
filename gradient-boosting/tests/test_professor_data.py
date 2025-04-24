import os
import sys
import csv
import numpy as np

# Get the directory of the current file
current_file_directory = os.path.dirname(os.path.abspath(__file__))
# Add a relative path to sys.path
relative_path = os.path.join(current_file_directory, '..')
print("relative_path", relative_path)
sys.path.append(relative_path)

from model.gradient_boosting import GradientBoosting

print("Test case has started running .....")
print("This may take a while for larger inputs .....")

def test_predict():
    # Initialize model
    model = GradientBoosting(
        loss='squared_error',  # or 'logistic' depending on your y values
        n_estimators=10,
        learning_rate=0.5,
        max_depth=2,
        min_samples_leaf=1,
        subsample=1.0,
        max_features=None,
        handle_missing='none'
    )

    # Load dataset from CSV
    data = []
    with open("small_test.csv", "r") as file:
        reader = csv.DictReader(file)
        for row in reader:
            data.append(row)

    # Extract features and labels
    X = np.array([[float(row[k]) for k in row if k.startswith("x")] for row in data])
    y = np.array([float(row["y"]) for row in data])

    print("Feature shape:", X.shape)
    print("Label shape:", y.shape)

    # Fit model
    model.fit(X, y)

    # Predict and visualize
    preds = model.predict(X)
    print("Predictions:", preds[:10], "...")
    model.plot_learning_curve("Professor Given Data")
    model.plot_predictions_vs_actual(preds, y)

    # Optional: check model output in expected range
    # assert np.all(preds >= 0) and np.all(preds <= 1), "Predictions out of range"

test_predict()

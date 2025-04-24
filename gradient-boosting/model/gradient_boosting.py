import numpy as np
import matplotlib.pyplot as plt


class DecisionTreeNode:
    def __init__(self, feature_index=None, threshold=None, left=None, right=None, *, value=None):
        self.feature_index = feature_index  # Index of the feature to split on
        self.threshold = threshold  # Threshold value to split at
        self.left = left  # Left child node
        self.right = right  # Right child node
        self.value = value  # Value at leaf node


class DecisionTreeRegressor:
    def __init__(self, max_depth=5, min_samples_split=2, min_samples_leaf=1, max_features=None):
        # Parameter validation
        if not isinstance(max_depth, int) or max_depth <= 0:
            raise ValueError("max_depth must be a positive integer")
        if not isinstance(min_samples_split, int) or min_samples_split <= 1:
            raise ValueError("min_samples_split must be an integer greater than 1")
        if not isinstance(min_samples_leaf, int) or min_samples_leaf <= 0:
            raise ValueError("min_samples_leaf must be a positive integer")
        if max_features is not None:
            if isinstance(max_features, int):
                if max_features <= 0:
                    raise ValueError("max_features must be positive")
            elif isinstance(max_features, float):
                if not (0.0 < max_features <= 1.0):
                    raise ValueError("max_features must be in (0.0, 1.0]")
            elif max_features not in ["sqrt", "log2"]:
                raise ValueError("max_features must be an int, float, 'sqrt', 'log2', or None")

        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.root = None
        self.n_features = None

    def fit(self, X, y):
        # Input validation
        X, y = self._validate_data(X, y)

        self.n_features = X.shape[1]
        self.root = self._build_tree(X, y, depth=0)

    def predict(self, X):
        # Input validation
        X = self._validate_X(X)

        return np.array([self._predict(inputs, self.root) for inputs in X])

    def _build_tree(self, X, y, depth):
        num_samples, num_features = X.shape
        if (depth >= self.max_depth) or (num_samples < self.min_samples_split) or (np.unique(y).shape[0] == 1):
            leaf_value = self._calculate_leaf_value(y)
            return DecisionTreeNode(value=leaf_value)

        # Find the best split
        best_split = self._get_best_split(X, y)
        if not best_split or best_split['gain'] == 0:
            leaf_value = self._calculate_leaf_value(y)
            return DecisionTreeNode(value=leaf_value)

        left_subtree = self._build_tree(best_split['X_left'], best_split['y_left'], depth + 1)
        right_subtree = self._build_tree(best_split['X_right'], best_split['y_right'], depth + 1)
        return DecisionTreeNode(feature_index=best_split['feature_index'], threshold=best_split['threshold'],
                                left=left_subtree, right=right_subtree)

    def _get_best_split(self, X, y):
        num_samples, num_features = X.shape
        if self.max_features is None:
            features = np.arange(num_features)
        elif isinstance(self.max_features, int):
            features = np.random.choice(num_features, self.max_features, replace=False)
        elif isinstance(self.max_features, float):
            max_features = int(self.max_features * num_features)
            features = np.random.choice(num_features, max_features, replace=False)
        elif self.max_features == "sqrt":
            max_features = int(np.sqrt(num_features))
            features = np.random.choice(num_features, max_features, replace=False)
        elif self.max_features == "log2":
            max_features = int(np.log2(num_features))
            features = np.random.choice(num_features, max_features, replace=False)
        else:
            features = np.arange(num_features)

        best_split = {}
        min_mse = float('inf')

        for feature_index in features:
            feature_values = X[:, feature_index]
            possible_thresholds = np.unique(feature_values[~np.isnan(feature_values)])
            for threshold in possible_thresholds:
                X_left, y_left, X_right, y_right = self._split(X, y, feature_index, threshold)
                # Enforce min_samples_leaf
                if len(y_left) >= self.min_samples_leaf and len(y_right) >= self.min_samples_leaf:
                    if len(y_left) > 0 and len(y_right) > 0:
                        current_mse = self._calculate_mse(y_left, y_right)
                        if current_mse < min_mse:
                            min_mse = current_mse
                            best_split = {
                                'feature_index': feature_index,
                                'threshold': threshold,
                                'X_left': X_left,
                                'y_left': y_left,
                                'X_right': X_right,
                                'y_right': y_right,
                                'gain': self._variance(y) - current_mse
                            }
        return best_split if best_split else None

    def _split(self, X, y, feature_index, threshold):
        feature_values = X[:, feature_index]
        # Handle missing values by considering NaNs as a separate group
        left_indices = np.where((feature_values <= threshold) & ~np.isnan(feature_values))[0]
        right_indices = np.where((feature_values > threshold) & ~np.isnan(feature_values))[0]
        nan_indices = np.where(np.isnan(feature_values))[0]

        X_left = X[left_indices]
        y_left = y[left_indices]
        X_right = X[right_indices]
        y_right = y[right_indices]

        # Assign missing values to the side with more samples
        if len(y_left) > len(y_right):
            X_left = np.vstack((X_left, X[nan_indices]))
            y_left = np.concatenate((y_left, y[nan_indices]))
        else:
            X_right = np.vstack((X_right, X[nan_indices]))
            y_right = np.concatenate((y_right, y[nan_indices]))

        return X_left, y_left, X_right, y_right

    def _calculate_leaf_value(self, y):
        return np.mean(y)

    def _predict(self, x, tree):
        if tree.value is not None:
            return tree.value
        feature_value = x[tree.feature_index]
        # Handle missing values during prediction
        if np.isnan(feature_value):
            # Choose the child with the higher number of samples
            if tree.left.value is not None and tree.right.value is not None:
                if len(tree.left.value) > len(tree.right.value):
                    return self._predict(x, tree.left)
                else:
                    return self._predict(x, tree.right)
            elif tree.left is not None:
                return self._predict(x, tree.left)
            else:
                return self._predict(x, tree.right)
        elif feature_value <= tree.threshold:
            return self._predict(x, tree.left)
        else:
            return self._predict(x, tree.right)

    def _calculate_mse(self, y_left, y_right):
        total_samples = len(y_left) + len(y_right)
        mse_left = np.var(y_left) * len(y_left) if len(y_left) > 0 else 0
        mse_right = np.var(y_right) * len(y_right) if len(y_right) > 0 else 0
        total_mse = (mse_left + mse_right) / total_samples
        return total_mse

    def _variance(self, y):
        return np.var(y)

    def _validate_data(self, X, y):
        # Check if X and y are NumPy arrays
        if not isinstance(X, np.ndarray):
            raise ValueError("X must be a NumPy array")
        if not isinstance(y, np.ndarray):
            raise ValueError("y must be a NumPy array")

        # Check that X and y have compatible shapes
        if X.shape[0] != y.shape[0]:
            raise ValueError("X and y have incompatible shapes")

        return X, y

    def _validate_X(self, X):
        # Check if X is a NumPy array
        if not isinstance(X, np.ndarray):
            raise ValueError("X must be a NumPy array")

        return X


class GradientBoosting:
    def __init__(self, loss='squared_error', n_estimators=100, learning_rate=0.1, max_depth=3,
                 min_samples_split=2, min_samples_leaf=1, subsample=1.0, max_features=None,
                 handle_missing='none'):
        # Parameter validation
        if loss not in ['squared_error', 'logistic']:
            raise ValueError("loss must be 'squared_error' or 'logistic'")
        if not isinstance(n_estimators, int) or n_estimators <= 0:
            raise ValueError("n_estimators must be a positive integer")
        if not 0 < learning_rate <= 1:
            raise ValueError("learning_rate must be in (0, 1]")
        if not isinstance(max_depth, int) or max_depth <= 0:
            raise ValueError("max_depth must be a positive integer")
        if not isinstance(min_samples_split, int) or min_samples_split <= 1:
            raise ValueError("min_samples_split must be an integer greater than 1")
        if not isinstance(min_samples_leaf, int) or min_samples_leaf <= 0:
            raise ValueError("min_samples_leaf must be a positive integer")
        if not 0 < subsample <= 1:
            raise ValueError("subsample must be in (0, 1]")
        if handle_missing not in ['none', 'mean', 'median']:
            raise ValueError("handle_missing must be 'none', 'mean', or 'median'")

        self.loss = loss
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.subsample = subsample
        self.max_features = max_features
        self.handle_missing = handle_missing
        self.trees = []
        self.initial_prediction = None
        self.imputer_values = None  # For storing mean or median values
        self.classes_ = None
        self.is_classification = self.loss == 'logistic'
        self.train_loss = []

    def fit(self, X, y):
        # Input validation
        X, y = self._validate_data(X, y)
        # Ensure target is 1D (flatten if necessary)
        if y.ndim != 1:
            y = y.ravel()
        # Handle missing values
        if self.handle_missing == 'mean':
            self.imputer_values = np.nanmean(X, axis=0)
            inds = np.where(np.isnan(X))
            X = X.copy()
            X[inds] = np.take(self.imputer_values, inds[1])
        elif self.handle_missing == 'median':
            self.imputer_values = np.nanmedian(X, axis=0)
            inds = np.where(np.isnan(X))
            X = X.copy()
            X[inds] = np.take(self.imputer_values, inds[1])
        # If handle_missing is 'none', we handle NaNs in the tree methods

        if self.is_classification:
            # Classification setup
            self.classes_ = np.unique(y)
            if len(self.classes_) != 2:
                raise ValueError("This implementation supports binary classification only.")
            y_encoded = np.where(y == self.classes_[0], -1, 1)

            # Compute initial prediction (log-odds)
            p = np.mean(y_encoded == 1)
            p = np.clip(p, 1e-15, 1 - 1e-15)  # Avoid division by zero
            self.initial_prediction = 0.5 * np.log(p / (1 - p))
            y_pred = np.full(y_encoded.shape, self.initial_prediction)
        else:
            # Regression setup
            y_encoded = y
            self.initial_prediction = np.mean(y)
            y_pred = np.full(y.shape, self.initial_prediction)

        n_samples = X.shape[0]

        for i in range(self.n_estimators):
            if self.is_classification:
                # Compute probabilities
                proba = 1 / (1 + np.exp(-2 * y_pred))
                # Compute negative gradients (pseudo-residuals)
                residuals = 2 * y_encoded / (1 + np.exp(2 * y_encoded * y_pred))
                # Compute training loss
                loss = np.mean(np.log(1 + np.exp(-2 * y_encoded * y_pred)))
                self.train_loss.append(loss)
            else:
                # Compute residuals for regression
                residuals = y_encoded - y_pred
                # Compute training loss
                mse = np.mean(residuals ** 2)
                self.train_loss.append(mse)

            # Implement subsampling
            if self.subsample < 1.0:
                indices = np.random.choice(n_samples, int(n_samples * self.subsample), replace=False)
                X_sub = X[indices]
                residuals_sub = residuals[indices]
            else:
                X_sub = X
                residuals_sub = residuals

            tree = DecisionTreeRegressor(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                min_samples_leaf=self.min_samples_leaf,
                max_features=self.max_features
            )
            tree.fit(X_sub, residuals_sub)
            prediction = tree.predict(X)
            y_pred += self.learning_rate * prediction
            self.trees.append(tree)

    def predict(self, X):
        # Input validation
        X = self._validate_X(X)
        # if X.ndim != 1:
        #     X = X.ravel()

        # Handle missing values in prediction
        if self.handle_missing in ['mean', 'median'] and self.imputer_values is not None:
            inds = np.where(np.isnan(X))
            X = X.copy()
            X[inds] = np.take(self.imputer_values, inds[1])

        y_pred = np.full((X.shape[0],), self.initial_prediction)
        for tree in self.trees:
            y_pred += self.learning_rate * tree.predict(X)

        if self.is_classification:
            proba = 1 / (1 + np.exp(-2 * y_pred))
            class_predictions = np.where(proba >= 0.5, self.classes_[1], self.classes_[0])
            return class_predictions
        else:
            return y_pred

    def predict_proba(self, X):
        if not self.is_classification:
            raise AttributeError("predict_proba is only available for classification problems.")
        # Input validation
        X = self._validate_X(X)

        # Handle missing values in prediction
        if self.handle_missing in ['mean', 'median'] and self.imputer_values is not None:
            inds = np.where(np.isnan(X))
            X = X.copy()
            X[inds] = np.take(self.imputer_values, inds[1])

        y_pred = np.full((X.shape[0],), self.initial_prediction)
        for tree in self.trees:
            y_pred += self.learning_rate * tree.predict(X)

        proba = 1 / (1 + np.exp(-2 * y_pred))
        return np.vstack([1 - proba, proba]).T

    def _validate_data(self, X, y):
        # Convert inputs to NumPy arrays
        feature = np.asarray(X)
        target = np.asarray(y)

        # Convert to float if data is not already numeric
        if not np.issubdtype(feature.dtype, np.number):
            try:
                feature = feature.astype(float)
            except ValueError:
                raise ValueError("Feature array contains non-numeric values that cannot be converted to float.")

        if not np.issubdtype(target.dtype, np.number):
            try:
                target = target.astype(float)
            except ValueError:
                raise ValueError("Target array contains non-numeric values that cannot be converted to float.")

        # Same validation as in DecisionTreeRegressor
        if not isinstance(feature, np.ndarray):
            raise ValueError("X must be a NumPy array")
        if not isinstance(target, np.ndarray):
            raise ValueError("y must be a NumPy array")
        if feature.shape[0] != target.shape[0]:
            raise ValueError("X and y have incompatible shapes")
        return feature, target

    def _validate_X(self, X):
        # Convert feature to a NumPy array if it isn't already

        feature = np.asarray(X)

        # Convert to float if data is not already numeric
        if not np.issubdtype(feature.dtype, np.number):
            try:
                feature = feature.astype(float)
            except ValueError:
                raise ValueError("Feature array contains non-numeric values that cannot be converted to float.")

        if not isinstance(X, np.ndarray):
            raise ValueError("X must be a NumPy array")
        return feature

    def plot_learning_curve(self, curve_name):
        plt.figure(figsize=(10, 6))
        plt.plot(range(1, len(self.train_loss) + 1), self.train_loss, marker='o')
        plt.title(f'Learning Curve for {curve_name}')
        plt.xlabel('Number of Trees')
        plt.ylabel('Mean Squared Error')
        plt.grid(True)
        plt.show()

    def plot_predictions_vs_actual(self, predictions, y):

        y = np.asarray(y, dtype=float)
        if y.ndim != 1:
            y = y.ravel()

        plt.figure(figsize=(10, 6))
        plt.scatter(y, predictions, alpha=0.7, color='dodgerblue')
        plt.plot([y.min(), y.max()], [y.min(), y.max()], '--', color='red')
        plt.title('Predictions vs Actual')
        plt.xlabel('Actual')
        plt.ylabel('Predicted')
        plt.grid(True)
        plt.show()

    def feature_importances_(self):
        # Compute feature importances as the average gain across all splits
        feature_importances = np.zeros(self.trees[0].n_features)
        for tree in self.trees:
            self._accumulate_importances(tree.root, feature_importances)
        # Normalize the importances
        feature_importances /= np.sum(feature_importances)
        return feature_importances

    def _accumulate_importances(self, node, importances):
        if node.value is None:
            # Internal node
            importances[node.feature_index] += self._variance(node.left.value) + self._variance(node.right.value)
            self._accumulate_importances(node.left, importances)
            self._accumulate_importances(node.right, importances)

    def _variance(self, y):
        return np.var(y) if y is not None else 0

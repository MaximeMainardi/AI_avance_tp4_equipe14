import numpy as np

class DecisionTree:
    def __init__(self, max_depth=None, min_samples_split=2):
        """
        Initialize the decision tree with optional constraints.
        - max_depth: Maximum depth of the tree.
        - min_samples_split: Minimum number of samples required to split.
        """
        self.tree = None
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split

    def _entropy(self, y):
        """
        Calculate entropy for a class vector.
        """
        classes, counts = np.unique(y, return_counts=True)
        probabilities = counts / len(y)
        return -np.sum(probabilities * np.log2(probabilities))

    def _information_gain(self, X_column, y, threshold):
        """
        Calculate information gain for a given split.
        """
        left_indices = X_column <= threshold
        right_indices = X_column > threshold

        if len(y[left_indices]) == 0 or len(y[right_indices]) == 0:
            return 0

        parent_entropy = self._entropy(y)
        left_entropy = self._entropy(y[left_indices])
        right_entropy = self._entropy(y[right_indices])

        n = len(y)
        n_left = len(y[left_indices])
        n_right = len(y[right_indices])

        child_entropy = (n_left / n) * left_entropy + (n_right / n) * right_entropy

        return parent_entropy - child_entropy

    def _best_split(self, X, y):
        """
        Find the best attribute and threshold for splitting the data.
        """
        best_gain = -1
        best_column = None
        best_threshold = None

        for column_index in range(X.shape[1]):
            X_column = X[:, column_index]
            thresholds = np.unique(X_column)

            for threshold in thresholds:
                gain = self._information_gain(X_column, y, threshold)

                if gain > best_gain:
                    best_gain = gain
                    best_column = column_index
                    best_threshold = threshold

        return best_column, best_threshold

    def _build_tree(self, X, y, depth=0):
        """
        Recursively build the decision tree.
        """
        # Stopping conditions
        if len(np.unique(y)) == 1:  # Pure leaf
            return np.unique(y)[0]
        if X.shape[0] < self.min_samples_split:  # Minimum samples split
            return np.bincount(y).argmax()
        if self.max_depth is not None and depth >= self.max_depth:  # Max depth
            return np.bincount(y).argmax()

        # Best split
        column, threshold = self._best_split(X, y)
        if column is None:  # No split possible
            return np.bincount(y).argmax()

        # Split data
        left_indices = X[:, column] <= threshold
        right_indices = X[:, column] > threshold

        # Recursively build left and right branches
        left_tree = self._build_tree(X[left_indices], y[left_indices], depth + 1)
        right_tree = self._build_tree(X[right_indices], y[right_indices], depth + 1)

        return {"column": column, "threshold": threshold, "left": left_tree, "right": right_tree}

    def train(self, train, train_labels):
        """
        Train the decision tree on a training set.
        """
        self.tree = self._build_tree(train, train_labels)

    def _predict_single(self, x, tree):
        """
        Predict a single instance using the decision tree.
        """
        if not isinstance(tree, dict):
            return tree

        column = tree["column"]
        threshold = tree["threshold"]

        if x[column] <= threshold:
            return self._predict_single(x, tree["left"])
        else:
            return self._predict_single(x, tree["right"])

    def predict(self, X):
        """
        Predict classes for multiple instances.
        """
        if X.ndim == 1:  # If a single example is provided, reshape it
            X = X.reshape(1, -1)
        return np.array([self._predict_single(x, self.tree) for x in X])

    def evaluate(self, X, y):
        """
        Evaluate the decision tree on a test set.
        """
        predictions = self.predict(X)
        accuracy = np.mean(predictions == y)
        return accuracy


class DecisionTree_Pruning(DecisionTree):
    def __init__(self, max_depth=None, min_samples_split=2, min_samples_leaf=1):
        """
        Initialize the decision tree with pruning options.
        - max_depth: Maximum depth of the tree.
        - min_samples_split: Minimum number of samples required to split.
        - min_samples_leaf: Minimum number of samples per leaf node.
        """
        super().__init__(max_depth, min_samples_split)
        self.min_samples_leaf = min_samples_leaf

    def _prune_tree(self, tree, X, y):
        """
        Prune the tree to avoid overfitting.
        """
        if not isinstance(tree, dict):
            return tree

        left_indices = X[:, tree["column"]] <= tree["threshold"]
        right_indices = X[:, tree["column"]] > tree["threshold"]

        # Check if leaf pruning is required
        if (
            np.sum(left_indices) <= self.min_samples_leaf
            or np.sum(right_indices) <= self.min_samples_leaf
        ):
            return np.bincount(y).argmax()

        # Recursively prune left and right subtrees
        tree["left"] = self._prune_tree(tree["left"], X[left_indices], y[left_indices])
        tree["right"] = self._prune_tree(tree["right"], X[right_indices], y[right_indices])

        return tree

    def train(self, train, train_labels):
        """
        Train the decision tree with pruning on a training set.
        """
        self.tree = self._build_tree(train, train_labels)
        self.tree = self._prune_tree(self.tree, train, train_labels)
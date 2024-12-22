"""
Vous allez definir une classe pour chaque algorithme que vous allez développer,
votre classe doit contenit au moins les 3 méthodes definies ici bas, 
	* train 	: pour entrainer le modèle sur l'ensemble d'entrainement.
	* predict 	: pour prédire la classe d'un exemple donné.
	* evaluate 		: pour evaluer le classifieur avec les métriques demandées. 
vous pouvez rajouter d'autres méthodes qui peuvent vous etre utiles, mais la correction
se fera en utilisant les méthodes train, predict et evaluate de votre code.
"""

import numpy as np


# le nom de votre classe
# BayesNaif pour le modèle bayesien naif
# Knn pour le modèle des k plus proches voisins


class DecisionTree:

    def __init__(self):
        """
        Initialisation de l'arbre de décision. L'arbre est représenté comme une structure récursive.
        """
        self.tree = None

    def _entropy(self, y):
        """
        Calcul de l'entropie pour un vecteur de classes.
        """
        classes, counts = np.unique(y, return_counts=True)
        probabilities = counts / len(y)
        return -np.sum(probabilities * np.log2(probabilities))

    def _information_gain(self, X_column, y, threshold):
        """
        Calcul du gain d'information pour une division donnée par un seuil.
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
        Trouve le meilleur attribut et le meilleur seuil pour diviser les données.
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

    def _build_tree(self, X, y):
        """
        Construction récursive de l'arbre de décision.
        """
        if len(np.unique(y)) == 1:
            return np.unique(y)[0]

        if X.shape[1] == 0:
            return np.bincount(y).argmax()

        column, threshold = self._best_split(X, y)

        if column is None:
            return np.bincount(y).argmax()

        left_indices = X[:, column] <= threshold
        right_indices = X[:, column] > threshold

        left_tree = self._build_tree(X[left_indices], y[left_indices])
        right_tree = self._build_tree(X[right_indices], y[right_indices])

        return {"column": column, "threshold": threshold, "left": left_tree, "right": right_tree}

    def _prune_tree(self, tree, X, y, min_samples):
        """
        Élague l'arbre pour éviter le surapprentissage.
        """
        if not isinstance(tree, dict):
            return tree

        left_indices = X[:, tree["column"]] <= tree["threshold"]
        right_indices = X[:, tree["column"]] > tree["threshold"]

        if np.sum(left_indices) < min_samples or np.sum(right_indices) < min_samples:
            return np.bincount(y).argmax()

        tree["left"] = self._prune_tree(tree["left"], X[left_indices], y[left_indices], min_samples)
        tree["right"] = self._prune_tree(tree["right"], X[right_indices], y[right_indices], min_samples)

        return tree

    def train(self, train, train_labels, min_samples=1):
        """
        Entraîne l'arbre de décision sur un ensemble d'entraînement avec élague.
        """
        self.tree = self._build_tree(train, train_labels)
        self.tree = self._prune_tree(self.tree, train, train_labels, min_samples)

    def _predict_single(self, x, tree):
        """
        Prédiction pour un exemple unique à partir de l'arbre.
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
        Prédire les classes pour plusieurs exemples.
        """
        return np.array([self._predict_single(x, self.tree) for x in X])

    def evaluate(self, X, y):
        """
        Évaluer l'arbre de décision sur un ensemble de test.
        """
        predictions = self.predict(X)
        accuracy = np.mean(predictions == y)
        return accuracy
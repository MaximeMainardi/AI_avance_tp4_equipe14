import numpy as np
from classifieur import Classifier


class Knn(Classifier):
    def __init__(self, k=5, distance_metric="euclidean"):
        self.k = k
        self.distance_metric = distance_metric
        self.train_data = None
        self.train_labels = None

    def train(
        self, train, train_labels
    ):  # vous pouvez rajouter d'autres attributs au besoin
        self.train_data = train
        self.train_labels = train_labels

    def predict(self, x: int) -> int:
        """
        Prédire la classe d'un exemple x donné en entrée
        exemple est de taille 1xm
        """
        distances = self._compute_distances(x)
        nearest_neighbors = self.train_labels[np.argsort(distances)[: self.k]]
        unique, counts = np.unique(nearest_neighbors, return_counts=True)
        return unique[np.argmax(counts)]

    def evaluate(self, X, y):
        predictions = np.array([self.predict(x) for x in X])
        accuracy = np.mean(predictions == y)
        return accuracy

    def _compute_distances(self, x):
        """
        Calcule les distances entre les données d'entraînement et un point donné selon la métrique de distance spécifiée.

        Paramètres:
        -----------
        x : numpy.ndarray
            Le point pour lequel les distances doivent être calculées.

        Retourne:
        ---------
        numpy.ndarray
            Un tableau des distances entre le point donné et chaque point des données d'entraînement.

        Lève:
        -----
        ValueError
            Si la métrique de distance spécifiée n'est pas supportée.

        Métriques de distance supportées:
        ---------------------------------
        - 'euclidean' : Distance euclidienne
        - 'manhattan' : Distance de Manhattan
        - 'chebyshev' : Distance de Chebyshev
        - 'minkowski' : Distance de Minkowski (avec p = 3 par défaut, modifiable)
        """
        if self.distance_metric == "euclidean":
            return np.sqrt(np.sum((self.train_data - x) ** 2, axis=1))
        elif self.distance_metric == "manhattan":
            return np.sum(np.abs(self.train_data - x), axis=1)
        elif self.distance_metric == "chebyshev":
            return np.max(np.abs(self.train_data - x), axis=1)
        elif self.distance_metric == "minkowski":
            p = 3  # You can change the value of p as needed
            return np.sum(np.abs(self.train_data - x) ** p, axis=1) ** (1 / p)
        else:
            raise ValueError(f"Unsupported distance metric: {self.distance_metric}")

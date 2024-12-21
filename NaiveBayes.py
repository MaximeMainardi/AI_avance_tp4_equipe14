import numpy as np


class NaiveBayes:
    def __init__(self):
        """
        Initializer. 
        Pour ce modele, nous allons stocker les statistiques necessaires pour le calcul de la probabilite conditionnelle, telles que la moyenne et l'ecart-type de chaque caracteristique par classe.
        """
        self.classes = None
        self.mean = {}
        self.var = {}
        self.priors = {}

    def train(self, train: np.ndarray, train_labels: np.ndarray) -> None:
        """
        Entrainer le modele Naive Bayes sur l'ensemble d'entrainement.

        Args:
            train : matrice numpy de taille nxm, avec n exemples d'entrainement et m caracteristiques.
            train_labels : vecteur numpy de taille nx1 contenant les etiquettes des exemples d'entrainement.
        """
        self.classes = np.unique(train_labels)
        for c in self.classes:
            X_c = train[train_labels == c]
            self.mean[c] = X_c.mean(axis=0)
            self.var[c] = X_c.var(axis=0)
            self.priors[c] = X_c.shape[0] / train.shape[0]

    def predict(self, x: int) -> int:
        """
        Predire la classe d'un exemple donne en entree.

        Args:
            x : vecteur numpy de taille 1xm representant un exemple.

        Returns:
            La classe predite pour l'exemple x.
        """
        posteriors = []

        for c in self.classes:
            prior = np.log(self.priors[c])
            likelihood = -0.5 * np.sum(np.log(2. * np.pi * self.var[c])) - 0.5 * np.sum(
                ((x - self.mean[c]) ** 2) / (self.var[c]))
            posterior = prior + likelihood
            posteriors.append(posterior)

        resultat = self.classes[np.argmax(posteriors)]
        return resultat

    def evaluate(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Evaluer le classifieur sur un ensemble de test.

        Args:
            X : matrice numpy de taille nxm, avec n exemples de test et m caracteristiques.
            y : vecteur numpy de taille nx1 contenant les etiquettes des exemples de test.

        Returns:
            Un dictionnaire contenant les metriques demandees : accuracy, precision, recall, F1-score, confusion matrix.
        """
        # Predict for each example
        y_pred = np.array([self.predict(x) for x in X])

        # Calculate accuracy
        accuracy = np.mean(y_pred == y)
        return accuracy

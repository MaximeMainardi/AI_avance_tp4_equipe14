## Travail pratique 4

#### Classes et modules utilisés

- `Classifier`
La classe `Classifier` est une classe de base pour les classifieurs. Elle définit une interface commune pour les méthodes d'entraînement, de prédiction et d'évaluation que les autres classifieurs doivent implémenter.

- `DecisionTree`
La classe `DecisionTree` implémente l'algorithme de l'arbre de décision. Elle inclut des méthodes pour entraîner le modèle, faire des prédictions et évaluer les performances du modèle.

- `NeuralNet`
La classe `NeuralNet` implemente l'algorithme d'un réseau de neurones. Elle inclut des méthodes pour entraîner le modèle, faire des prédictions et évaluer les performances du modèle.

- `load_datasets`
Ce module contient des fonctions pour charger différents ensembles de données, tels que les datasets Iris, Wine et Abalone. Les fonctions de ce module divisent également les données en ensembles d'entraînement et de test.

- `entrainer_tester`
Ce fichier contient des fonctions pour entraîner et tester les modèles de classification. Il inclut également des fonctions pour calculer des métriques de performance telles que la précision, le rappel, le F1-score et la matrice de confusion.

#### Fonctions utilisées

- `load_iris_dataset`
La fonction `load_iris_dataset` charge le dataset Iris, le divise en ensembles d'entraînement et de test, et retourne ces ensembles.

- `load_wine_dataset`
La fonction `load_wine_dataset` charge le dataset Wine, le divise en ensembles d'entraînement et de test, et retourne ces ensembles.

- `load_abalone_dataset`
La fonction `load_abalone_dataset` charge le dataset Abalone, le divise en ensembles d'entraînement et de test, et retourne ces ensembles.

- `precision`
La fonction `precision` calcule la précision d'un modèle sur un ensemble de données donné.

- `recall`
La fonction `recall` calcule le rappel d'un modèle sur un ensemble de données donné.

- `f1_score`
La fonction `f1_score` calcule le F1-score d'un modèle sur un ensemble de données donné.

- `confusion_matrix`
La fonction `confusion_matrix` calcule la matrice de confusion d'un modèle sur un ensemble de données donné.

- `kfold_cross_validation`
La fonction `kfold_cross_validation` effectue une validation croisée pour sélectionner les meilleurs hyperparamètres pour le modèle NeuralNet.


### Répartition des tâches

 Comme pour le dernier tp, chacun des membres s'est occupé d'implémenter et tester un modèle. 

### Difficultés rencontrées

L'implémentation du réseau de neurones s'est avérée difficile, en particulier pour les couches cachées. Nous avons eu plusieurs versions avant d'arriver à des résultats satisfaisants, dans le cas où ils sont valides.
La validation croisé est effectué sur le modèle mlp de sklearn et selon les hyperparamêtres qu'on obtient notre réseau de neurones a parfois de la difficulté à bien performer.  
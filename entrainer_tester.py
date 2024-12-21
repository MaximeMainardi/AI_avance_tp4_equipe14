import numpy as np
import sys
import load_datasets

# import DecisionTree  # importer la classe de l'arbre de décision
import NeuralNet  # importer la classe du Knn

# importer d'autres fichiers et classes si vous en avez développés
import time

from sklearn.neural_network import MLPClassifier
from Knn import Knn
from NaiveBayes import NaiveBayes

"""
C'est le fichier main duquel nous allons tout lancer
Vous allez dire en commentaire c'est quoi les paramètres que vous avez utilisés
En gros, vous allez :
1- Initialiser votre classifieur avec ses paramètres
2- Charger les datasets
3- Entraîner votre classifieur
4- Le tester
"""


def accuracy(predictions: list, values: list):
    return np.sum(predictions == values) / len(values)


def precision(predictions: list, values: list):
    unique_classes = np.unique(values)
    precisions = []

    for cls in unique_classes:
        true_positives = np.sum((predictions == cls) & (values == cls))
        false_positives = np.sum((predictions == cls) & (values != cls))
        if (true_positives + false_positives) != 0:
            precisions.append(true_positives / (true_positives + false_positives))
        else:
            precisions.append(0)

    return np.mean(precisions)


def recall(predictions: list, values: list):
    unique_classes = np.unique(values)
    recalls = []

    for cls in unique_classes:
        true_positives = np.sum((predictions == cls) & (values == cls))
        false_negatives = np.sum((predictions != cls) & (values == cls))
        if (true_positives + false_negatives) != 0:
            recalls.append(true_positives / (true_positives + false_negatives))
        else:
            recalls.append(0)

    return np.mean(recalls)


def f1_score(predictions: list, values: list):
    p = precision(predictions, values)
    r = recall(predictions, values)
    return 2 * p * r / (p + r) if (p + r) != 0 else 0


def confusion_matrix(predictions: list, values: list):
    unique_values = np.unique(values)
    matrix = np.zeros((len(unique_values), len(unique_values)))
    for true_label, pred_label in zip(values, predictions):
        matrix[true_label, pred_label] += 1
    return matrix


def print_prediction_summary(predictions: list, values: list):
    print(f"\tConfusion matrix:\n{confusion_matrix(predictions, values)}")

    for i in range(np.unique(values).shape[0]):
        print(f"\n\tClass {i}")
        print(f"\t\tAccuracy: {accuracy(predictions == i, values == i)}")
        print(f"\t\tPrecision: {precision(predictions == i, values == i)}")
        print(f"\t\tRecall: {recall(predictions == i, values == i)}")
        print(f"\t\tF1-score: {f1_score(predictions == i, values == i)}")


def kfold_cross_validation(X, y, model_class, model_params, k=5):
    fold_size = len(X) // k
    accuracies = []

    for i in range(k):
        start, end = i * fold_size, (i + 1) * fold_size
        X_val, y_val = X[start:end], y[start:end]
        X_train = np.concatenate([X[:start], X[end:]], axis=0)
        y_train = np.concatenate([y[:start], y[end:]], axis=0)

        model = model_class(**model_params)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_val)
        accuracy = np.mean(y_pred == y_val)
        accuracies.append(accuracy)

    return np.mean(accuracies)


# load datasets
train_ratio = 0.7
train_iris, train_labels_iris, test_iris, test_labels_iris = (
    load_datasets.load_iris_dataset(train_ratio)
)
train_wine, train_labels_wine, test_wine, test_labels_wine = (
    load_datasets.load_wine_dataset(train_ratio)
)
train_abalone, train_labels_abalone, test_abalone, test_labels_abalone = (
    load_datasets.load_abalone_dataset(train_ratio)
)

# Decision Tree

# decision_tree_iris = DecisionTree.DecisionTree()
# decision_tree_wine = DecisionTree.DecisionTree()
# decision_tree_abalones = DecisionTree.DecisionTree()

print("\n\u001b[31;1mTrain Decision Tree:\u001b[0m")
print("\u001b[32;1mIris:\u001b[0m")
time_decision_tree_train_iris = time.time()
# decision_tree_iris.train(train_iris, train_labels_iris)
time_decision_tree_train_iris = time.time() - time_decision_tree_train_iris
# predictions = np.array([decision_tree_iris.predict(x) for x in train_iris])
# print_prediction_summary(predictions, train_labels_iris)

print("\u001b[32;1mWine:\u001b[0m")
time_decision_tree_train_wine = time.time()
# decision_tree_wine.train(train_wine, train_labels_wine)
time_decision_tree_train_wine = time.time() - time_decision_tree_train_wine
# predictions = np.array([decision_tree_wine.predict(x) for x in train_wine])
# print_prediction_summary(predictions, train_labels_wine)

print("\u001b[32;1mAbalones:\u001b[0m")
time_decision_tree_train_abalones = time.time()
# decision_tree_abalones.train(train_abalone, train_labels_abalone)
time_decision_tree_train_abalones = time.time() - time_decision_tree_train_abalones
# predictions = np.array([decision_tree_abalones.predict(x) for x in train_abalone])
# print_prediction_summary(predictions, train_labels_abalone)


# Neural Network
hidden_layers_sizes_to_test = [
    (10,),
    (30,),
    (50,),
    (100,),
    (150,),
    (200,),
    (250,),
    (300,),
]

best_size_iris = None
best_size_wine = None
best_size_abalones = None
best_depth_iris = None
best_depth_wine = None
best_depth_abalones = None

# optimisation du nombre de neurones pour le dataset iris
print("\nOptimizing Neural Network:")
print("\nNeural Network layer size for iris:")
for hidden_layers in hidden_layers_sizes_to_test:
    score = kfold_cross_validation(
        train_iris,
        train_labels_iris,
        MLPClassifier,
        {
            "hidden_layer_sizes": hidden_layers,
            "max_iter": 1000,
            "activation": "relu",
            "learning_rate": "constant",
            "learning_rate_init": 0.01,
        },
    )
    if best_size_iris is None or score > best_size_iris[1]:
        best_size_iris = (hidden_layers, score)
    print(f"Hidden layers: {hidden_layers}, Accuracy: {score}")

# optimisation du nombre de neurones pour le dataset wine
print("\nNeural Network layer size for wine:")
for hidden_layers in hidden_layers_sizes_to_test:
    score = kfold_cross_validation(
        train_wine,
        train_labels_wine,
        MLPClassifier,
        {
            "hidden_layer_sizes": hidden_layers,
            "max_iter": 1000,
            "activation": "relu",
            "learning_rate": "constant",
            "learning_rate_init": 0.01,
        },
    )
    if best_size_wine is None or score > best_size_wine[1]:
        best_size_wine = (hidden_layers, score)
    print(f"Hidden layers: {hidden_layers}, Accuracy: {score}")

# optimisation du nombre de neurones pour le dataset abalones
print("\nNeural Network layer size for abalones:")
for hidden_layers in hidden_layers_sizes_to_test:
    score = kfold_cross_validation(
        train_abalone,
        train_labels_abalone,
        MLPClassifier,
        {
            "hidden_layer_sizes": hidden_layers,
            "max_iter": 1000,
            "activation": "relu",
            "learning_rate": "constant",
            "learning_rate_init": 0.01,
        },
    )
    if best_size_abalones is None or score > best_size_abalones[1]:
        best_size_abalones = (hidden_layers, score)
    print(f"Hidden layers: {hidden_layers}, Accuracy: {score}")

print(
    f"\nBest hidden layers for iris: {best_size_iris[0][0]}, Accuracy: {best_size_iris[1]}"
)
print(
    f"Best hidden layers for wine: {best_size_wine[0][0]}, Accuracy: {best_size_wine[1]}"
)
print(
    f"Best hidden layers for abalones: {best_size_abalones[0][0]}, Accuracy: {best_size_abalones[1]}"
)

bsi = best_size_iris[0][0]
bsw = best_size_wine[0][0]
bsa = best_size_abalones[0][0]

hidden_layers_to_test_iris = [
    (bsi,),
    (bsi, bsi),
    (bsi, bsi, bsi),
    (bsi, bsi, bsi, bsi),
    (bsi, bsi, bsi, bsi, bsi),
]
hidden_layers_to_test_wine = [
    (bsw,),
    (bsw, bsw),
    (bsw, bsw, bsw),
    (bsw, bsw, bsw, bsw),
    (bsw, bsw, bsw, bsw, bsw),
]
hidden_layers_to_test_abalones = [
    (bsa,),
    (bsa, bsa),
    (bsa, bsa, bsa),
    (bsa, bsa, bsa, bsa),
    (bsa, bsa, bsa, bsa, bsa),
]

# optimisation de la profondeur pour le dataset iris
print("\nNeural Network depth for iris:")
for hidden_layers in hidden_layers_to_test_iris:
    score = kfold_cross_validation(
        train_iris,
        train_labels_iris,
        MLPClassifier,
        {
            "hidden_layer_sizes": hidden_layers,
            "max_iter": 1000,
            "activation": "relu",
            "learning_rate": "constant",
            "learning_rate_init": 0.01,
        },
    )
    if best_depth_iris is None or score > best_depth_iris[1]:
        best_depth_iris = (hidden_layers, score)
    print(f"Hidden layers: {hidden_layers}, Accuracy: {score}")

# optimisation de la profondeur pour le dataset wine
print("\nNeural Network depth for wine:")
for hidden_layers in hidden_layers_to_test_wine:
    score = kfold_cross_validation(
        train_wine,
        train_labels_wine,
        MLPClassifier,
        {
            "hidden_layer_sizes": hidden_layers,
            "max_iter": 1000,
            "activation": "relu",
            "learning_rate": "constant",
            "learning_rate_init": 0.01,
        },
    )
    if best_depth_wine is None or score > best_depth_wine[1]:
        best_depth_wine = (hidden_layers, score)
    print(f"Hidden layers: {hidden_layers}, Accuracy: {score}")

# optimisation de la profondeur pour le dataset abalones
print("\nNeural Network depth for abalones:")
for hidden_layers in hidden_layers_to_test_abalones:
    score = kfold_cross_validation(
        train_abalone,
        train_labels_abalone,
        MLPClassifier,
        {
            "hidden_layer_sizes": hidden_layers,
            "max_iter": 1000,
            "activation": "relu",
            "learning_rate": "constant",
            "learning_rate_init": 0.01,
        },
    )
    if best_depth_abalones is None or score > best_depth_abalones[1]:
        best_depth_abalones = (hidden_layers, score)
    print(f"Hidden layers: {hidden_layers}, Accuracy: {score}")

print(
    f"\nBest hidden layers for iris: {best_depth_iris[0]}, Accuracy: {best_depth_iris[1]}"
)
print(
    f"Best hidden layers for wine: {best_depth_wine[0]}, Accuracy: {best_depth_wine[1]}"
)
print(
    f"Best hidden layers for abalones: {best_depth_abalones[0]}, Accuracy: {best_depth_abalones[1]}"
)

nn_iris = NeuralNet.NeuralNet(best_depth_iris[0])
nn_wine = NeuralNet.NeuralNet(best_depth_wine[0])
nn_abalones = NeuralNet.NeuralNet(best_depth_abalones[0])


print("\n\u001b[31;1mTrain Neural Network:\u001b[0m")
print("\u001b[32;1mIris:\u001b[0m")
time_nn_train_iris = time.time()
nn_iris.train(train_iris, train_labels_iris)
time_nn_train_iris = time.time() - time_nn_train_iris
predictions = np.array([nn_iris.predict(x) for x in train_iris])
print_prediction_summary(predictions, train_labels_iris)

print("\n\u001b[32;1mWine:\u001b[0m")
time_nn_train_wine = time.time()
nn_wine.train(train_wine, train_labels_wine)
time_nn_train_wine = time.time() - time_nn_train_wine
predictions = np.array([nn_wine.predict(x) for x in train_wine])
print_prediction_summary(predictions, train_labels_wine)

print("\n\u001b[32;1mAbalones:\u001b[0m")
time_nn_train_abalones = time.time()
nn_abalones.train(train_abalone, train_labels_abalone)
time_nn_train_abalones = time.time() - time_nn_train_abalones
predictions = np.array([nn_abalones.predict(x) for x in train_abalone])
print_prediction_summary(predictions, train_labels_abalone)

print("\n\u001b[31;1mTest Neural Network:\u001b[0m")
print("\u001b[32;1mIris:\u001b[0m")
predictions = np.array([nn_iris.predict(x) for x in test_iris])
print_prediction_summary(predictions, test_labels_iris)

print("\n\u001b[32;1mWine:\u001b[0m")
predictions = np.array([nn_wine.predict(x) for x in test_wine])
print_prediction_summary(predictions, test_labels_wine)

print("\n\u001b[32;1mAbalones:\u001b[0m")
predictions = np.array([nn_abalones.predict(x) for x in test_abalone])
print_prediction_summary(predictions, test_labels_abalone)


# comparaison des 4 modèles

knn_iris = Knn(k=3, distance_metric="chebyshev")
knn_wine = Knn(k=2, distance_metric="manhattan")
knn_abalones = Knn(k=2, distance_metric="manhattan")

naive_bayes_iris = NaiveBayes()
naive_bayes_wine = NaiveBayes()
naive_bayes_abalones = NaiveBayes()

"""training time comparison"""
# time train knn iris
time_knn_train_iris = time.time()
knn_iris.train(train_iris, train_labels_iris)
time_knn_train_iris = time.time() - time_knn_train_iris

# time train knn wine
time_knn_train_wine = time.time()
knn_wine.train(train_wine, train_labels_wine)
time_knn_train_wine = time.time() - time_knn_train_wine

# time train knn abalones
time_knn_train_abalones = time.time()
knn_abalones.train(train_abalone, train_labels_abalone)
time_knn_train_abalones = time.time() - time_knn_train_abalones

# time train naive bayes iris
time_naive_bayes_train_iris = time.time()
naive_bayes_iris.train(train_iris, train_labels_iris)
time_naive_bayes_train_iris = time.time() - time_naive_bayes_train_iris

# time train naive bayes wine
time_naive_bayes_train_wine = time.time()
naive_bayes_wine.train(train_wine, train_labels_wine)
time_naive_bayes_train_wine = time.time() - time_naive_bayes_train_wine

# time train naive bayes abalones
time_naive_bayes_train_abalones = time.time()
naive_bayes_abalones.train(train_abalone, train_labels_abalone)
time_naive_bayes_train_abalones = time.time() - time_naive_bayes_train_abalones


"""prediction time comparison"""
# time predict knn iris
time_knn_predict_iris = time.time()
knn_iris_pred = knn_iris.predict(test_iris[5])
time_knn_predict_iris = time.time() - time_knn_predict_iris
accuracy_knn_iris = knn_iris.evaluate(test_iris, test_labels_iris)

# time predict knn wine
time_knn_predict_wine = time.time()
knn_wine_pred = knn_wine.predict(test_wine[5])
time_knn_predict_wine = time.time() - time_knn_predict_wine
accuracy_knn_wine = knn_wine.evaluate(test_wine, test_labels_wine)

# time predict knn abalones
time_knn_predict_abalones = time.time()
knn_abalones_pred = knn_abalones.predict(test_abalone[5])
time_knn_predict_abalones = time.time() - time_knn_predict_abalones
accuracy_knn_abalones = knn_abalones.evaluate(test_abalone, test_labels_abalone)

# time predict naive bayes iris
time_naive_bayes_predict_iris = time.time()
naive_bayes_iris_pred = naive_bayes_iris.predict(test_iris[5])
time_naive_bayes_predict_iris = time.time() - time_naive_bayes_predict_iris
accuracy_naive_bayes_iris = naive_bayes_iris.evaluate(test_iris, test_labels_iris)

# time predict naive bayes wine
time_naive_bayes_predict_wine = time.time()
naive_bayes_wine_pred = naive_bayes_wine.predict(test_wine[5])
time_naive_bayes_predict_wine = time.time() - time_naive_bayes_predict_wine
accuracy_naive_bayes_wine = naive_bayes_wine.evaluate(test_wine, test_labels_wine)

# time predict naive bayes abalones
time_naive_bayes_predict_abalones = time.time()
naive_bayes_abalones_pred = naive_bayes_abalones.predict(test_abalone[5])
time_naive_bayes_predict_abalones = time.time() - time_naive_bayes_predict_abalones
accuracy_naive_bayes_abalones = naive_bayes_abalones.evaluate(
    test_abalone, test_labels_abalone
)

# time predict decision tree iris
time_decision_tree_predict_iris = time.time()
# decision_tree_iris_pred = decision_tree_iris.predict(test_iris[5])
time_decision_tree_predict_iris = time.time() - time_decision_tree_predict_iris
# accuracy_decision_tree_iris = decision_tree_iris.evaluate(test_iris, test_labels_iris)

# time predict decision tree wine
time_decision_tree_predict_wine = time.time()
# decision_tree_wine_pred = decision_tree_wine.predict(test_wine[5])
time_decision_tree_predict_wine = time.time() - time_decision_tree_predict_wine
# accuracy_decision_tree_wine = decision_tree_wine.evaluate(test_wine, test_labels_wine)

# time predict decision tree abalones
time_decision_tree_predict_abalones = time.time()
# decision_tree_abalones_pred = decision_tree_abalones.predict(test_abalone[5])
time_decision_tree_predict_abalones = time.time() - time_decision_tree_predict_abalones
# accuracy_decision_tree_abalones = decision_tree_abalones.evaluate(test_abalone, test_labels_abalone)

# time predict neural network iris
time_nn_predict_iris = time.time()
nn_iris_pred = nn_iris.predict(test_iris[5])
time_nn_predict_iris = time.time() - time_nn_predict_iris
accuracy_nn_iris = nn_iris.evaluate(test_iris, test_labels_iris)

# time predict neural network wine
time_nn_predict_wine = time.time()
nn_wine_pred = nn_wine.predict(test_wine[5])
time_nn_predict_wine = time.time() - time_nn_predict_wine
accuracy_nn_wine = nn_wine.evaluate(test_wine, test_labels_wine)

# time predict neural network abalones
time_nn_predict_abalones = time.time()
nn_abalones_pred = nn_abalones.predict(test_abalone[5])
time_nn_predict_abalones = time.time() - time_nn_predict_abalones
accuracy_nn_abalones = nn_abalones.evaluate(test_abalone, test_labels_abalone)

import numpy as np
import sys
import load_datasets
import matplotlib.pyplot as plt
import time
import os

from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier

from NeuralNet import NeuralNet  # importer le reseau de neurone
from DecisionTree import DecisionTree, DecisionTree_Pruning  # importer la classe de l'arbre de décision
from NaiveBayes import NaiveBayes
from Knn import Knn


from sklearn.model_selection import learning_curve
from sklearn.metrics import accuracy_score
"""
C'est le fichier main duquel nous allons tout lancer
Vous allez dire en commentaire c'est quoi les paramètres que vous avez utilisés
En gros, vous allez :
1- Initialiser votre classifieur avec ses paramètres
2- Charger les datasets
3- Entraîner votre classifieur
4- Le tester
"""
output_dir=r"C:\Users\User\Documents\AI_avance_tp4_equipe14"

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


# fonction pour obtenir un graph de l'accuracy en fonction du nombre de neurones
def get_accuracy_vs_neurons(X, y, model_class, model_params, neurons_range):
    accuracies = []
    for neurons in neurons_range:
        model = model_class(hidden_layer_sizes=(neurons,), **model_params)
        accuracy = kfold_cross_validation(X, y, model_class, model_params)
        accuracies.append(accuracy)
    return accuracies


def plot_error_vs_neurons(neurons_range, losses, dataset_name):
    plt.figure(figsize=(10, 6))
    plt.plot(neurons_range, losses, marker="o")
    plt.title(f"Error vs number of neurones for {dataset_name}")
    plt.xlabel("Number of Neurons")
    plt.ylabel("Error(1-Accuracy)")
    plt.grid(True)
    title = dataset_name + " " + str(neurons_range)
    # Save the plot
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    plot_path = os.path.join(output_dir, f"{title.replace(' ', '_').lower()}.png")
    plt.savefig(plot_path)
    plt.close()  # Close the plot to free memory
    print(f"Plot saved at: {plot_path}")


def plot_error_vs_depth(depth_range, losses, dataset_name, best_size):
    plt.figure(figsize=(10, 6))
    plt.plot(depth_range, losses, marker="o")
    plt.title(f"Error vs depth for {dataset_name} (neurons per layer: {best_size})")
    plt.xlabel("Depth")
    plt.ylabel("Error(1-Accuracy)")
    plt.grid(True)
    plt.xticks(depth_range)
    title = dataset_name + " " + str(depth_range)
    # Save the plot
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    plot_path = os.path.join(output_dir, f"{title.replace(' ', '_').lower()}.png")
    plt.savefig(plot_path)
    plt.close()  # Close the plot to free memory
    print(f"Plot saved at: {plot_path}")

def custom_learning_curve(estimator, X, y, title="Learning Curve", cv=5):
    """
    Custom learning curve function for non-Scikit-learn estimators.
    """
    n_samples = X.shape[0]
    train_sizes = np.linspace(0.1, 1.0, 20, endpoint=True).astype(float) * n_samples
    train_sizes = train_sizes.astype(int)

    train_errors = []
    test_errors = []

    # Split data into k folds for cross-validation
    indices = np.arange(n_samples)
    np.random.shuffle(indices)
    fold_size = n_samples // cv
    folds = [indices[i * fold_size:(i + 1) * fold_size] for i in range(cv)]

    for train_size in train_sizes:
        fold_train_errors = []
        fold_test_errors = []

        for fold_idx in range(cv):
            # Create train-test split for this fold
            test_indices = folds[fold_idx]
            train_indices = np.setdiff1d(indices, test_indices)[:train_size]

            X_train, y_train = X[train_indices], y[train_indices]
            X_test, y_test = X[test_indices], y[test_indices]

            # Train the model
            estimator.train(X_train, y_train)

            # Compute errors
            train_predictions = estimator.predict(X_train)
            test_predictions = estimator.predict(X_test)

            train_error = 1 - np.mean(train_predictions == y_train)
            test_error = 1 - np.mean(test_predictions == y_test)

            fold_train_errors.append(train_error)
            fold_test_errors.append(test_error)

        # Average error across folds
        train_errors.append(np.mean(fold_train_errors))
        test_errors.append(np.mean(fold_test_errors))

    # Plot the learning curve (error rates)
    plt.figure(figsize=(10, 6))
    plt.plot(train_sizes, train_errors, label="Training Error", color="red")
    plt.plot(train_sizes, test_errors, label="Validation Error", color="blue")
    plt.title(title)
    plt.xlabel("Training Set Size")
    plt.ylabel("Error Rate")
    plt.legend(loc="best")
    plt.grid()

    # Save the plot
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    plot_path = os.path.join(output_dir, f"{title.replace(' ', '_').lower()}.png")
    plt.savefig(plot_path)
    plt.close()  # Close the plot to free memory
    print(f"Plot saved at: {plot_path}")

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

decision_tree_iris = DecisionTree(max_depth=3,min_samples_split=2)
decision_tree_wine = DecisionTree(max_depth=5,min_samples_split=5)
decision_tree_abalones = DecisionTree(max_depth=10,min_samples_split=20)

print("\n\u001b[31;1mTrain Decision Tree:\u001b[0m")
print("\u001b[32;1mIris:\u001b[0m")
time_decision_tree_train_iris = time.time()
decision_tree_iris.train(train_iris, train_labels_iris)
time_decision_tree_train_iris = time.time() - time_decision_tree_train_iris
predictions = np.array([decision_tree_iris.predict(x) for x in train_iris])
print_prediction_summary(predictions, train_labels_iris)
custom_learning_curve(decision_tree_iris, train_iris, train_labels_iris, title="Learning Curve Decision Tree (Iris)")

print("\u001b[32;1mWine:\u001b[0m")
time_decision_tree_train_wine = time.time()
decision_tree_wine.train(train_wine, train_labels_wine)
time_decision_tree_train_wine = time.time() - time_decision_tree_train_wine
predictions = np.array([decision_tree_wine.predict(x) for x in train_wine])
print_prediction_summary(predictions, train_labels_wine)
custom_learning_curve(decision_tree_wine,train_wine, train_labels_wine, title="Learning Curve Decision Tree (Wine)")

print("\u001b[32;1mAbalones:\u001b[0m")
time_decision_tree_train_abalones = time.time()
decision_tree_abalones.train(train_abalone, train_labels_abalone)
time_decision_tree_train_abalones = time.time() - time_decision_tree_train_abalones
predictions = np.array([decision_tree_abalones.predict(x) for x in train_abalone])
print_prediction_summary(predictions, train_labels_abalone)
custom_learning_curve(decision_tree_abalones,train_abalone, train_labels_abalone, title="Learning Curve Decision Tree (Abalone)")

decision_tree_iris_sklearn = DecisionTreeClassifier(
    criterion="gini", max_depth=3,min_samples_split=2, random_state=15
)
decision_tree_wine_sklearn = DecisionTreeClassifier(
    criterion="gini", max_depth=5,min_samples_split=5, random_state=15
)
decision_tree_abalones_sklearn = DecisionTreeClassifier(
    criterion="gini", max_depth=10,min_samples_split=20, random_state=15
)

print("\n\u001b[31;1mTrain Scikit-learn Decision Tree:\u001b[0m")
print("\u001b[32;1mIris:\u001b[0m")
decision_tree_iris_sklearn.fit(train_iris, train_labels_iris)
# predictions = np.array([decision_tree_iris_sklearn.predict(x) for x in train_iris])
predictions = decision_tree_iris_sklearn.predict(train_iris)
print_prediction_summary(predictions, train_labels_iris)

print("\u001b[32;1mWine:\u001b[0m")
decision_tree_wine_sklearn.fit(train_wine, train_labels_wine)
# predictions = np.array([decision_tree_wine_sklearn.predict(x) for x in train_wine])
predictions = decision_tree_wine_sklearn.predict(train_wine)
print_prediction_summary(predictions, train_labels_wine)

print("\u001b[32;1mAbalones:\u001b[0m")
decision_tree_abalones_sklearn.fit(train_abalone, train_labels_abalone)
# predictions = np.array(
#     [decision_tree_abalones_sklearn.predict(x) for x in train_abalone]
# )
predictions = decision_tree_abalones_sklearn.predict(train_abalone)
print_prediction_summary(predictions, train_labels_abalone)

decision_tree_iris_pruning = DecisionTree_Pruning(max_depth=3,min_samples_split=2)
decision_tree_wine_pruning = DecisionTree_Pruning(max_depth=5,min_samples_split=5)
decision_tree_abalones_pruning = DecisionTree_Pruning(max_depth=10,min_samples_split=20)

print("\n\u001b[31;1mTrain Decision Tree with Pruning:\u001b[0m")
print("\u001b[32;1mIris:\u001b[0m")
time_decision_tree_train_iris = time.time()
decision_tree_iris_pruning.train(train_iris, train_labels_iris)
time_decision_tree_train_iris = time.time() - time_decision_tree_train_iris
predictions = np.array([decision_tree_iris_pruning.predict(x) for x in train_iris])
print_prediction_summary(predictions, train_labels_iris)
custom_learning_curve(decision_tree_iris_pruning,train_iris, train_labels_iris, title="Learning Curve Decision Tree Pruning (Iris)")

print("\u001b[32;1mWine:\u001b[0m")
time_decision_tree_train_wine = time.time()
decision_tree_wine_pruning.train(train_wine, train_labels_wine)
time_decision_tree_train_wine = time.time() - time_decision_tree_train_wine
predictions = np.array([decision_tree_wine_pruning.predict(x) for x in train_wine])
print_prediction_summary(predictions, train_labels_wine)
custom_learning_curve(decision_tree_wine_pruning,train_wine, train_labels_wine, title="Learning Curve Decision Tree Pruning (Wine)")

print("\u001b[32;1mAbalones:\u001b[0m")
time_decision_tree_train_abalones = time.time()
decision_tree_abalones_pruning.train(train_abalone, train_labels_abalone)
time_decision_tree_train_abalones = time.time() - time_decision_tree_train_abalones
predictions = np.array([decision_tree_abalones_pruning.predict(x) for x in train_abalone])
print_prediction_summary(predictions, train_labels_abalone)
custom_learning_curve(decision_tree_abalones_pruning,train_abalone, train_labels_abalone, title="Learning Curve Decision Tree Pruning (Abalone)")

print("\n\u001b[31;1mTest Decision Tree:\u001b[0m")
print("\u001b[32;1mIris:\u001b[0m")
# predictions = np.array([nn_iris.predict(x) for x in test_iris])
predictions = decision_tree_iris.predict(test_iris)
print_prediction_summary(predictions, test_labels_iris)

print("\n\u001b[32;1mWine:\u001b[0m")
# predictions = np.array([nn_wine.predict(x) for x in test_wine])
predictions = decision_tree_wine.predict(test_wine)
print_prediction_summary(predictions, test_labels_wine)

print("\n\u001b[32;1mAbalones:\u001b[0m")
# predictions = np.array([nn_abalones.predict(x) for x in test_abalone])
predictions = decision_tree_abalones.predict(test_abalone)
print_prediction_summary(predictions, test_labels_abalone)

print("\n\u001b[31;1mTest Decision Tree Scikit-learn:\u001b[0m")
print("\u001b[32;1mIris:\u001b[0m")
# predictions = np.array([nn_iris.predict(x) for x in test_iris])
predictions = decision_tree_iris_sklearn.predict(test_iris)
print_prediction_summary(predictions, test_labels_iris)

print("\n\u001b[32;1mWine:\u001b[0m")
# predictions = np.array([nn_wine.predict(x) for x in test_wine])
predictions =decision_tree_wine_sklearn.predict(test_wine)
print_prediction_summary(predictions, test_labels_wine)

print("\n\u001b[32;1mAbalones:\u001b[0m")
# predictions = np.array([nn_abalones.predict(x) for x in test_abalone])
predictions = decision_tree_abalones_sklearn.predict(test_abalone)
print_prediction_summary(predictions, test_labels_abalone)

print("\n\u001b[31;1mTest Decision Tree Pruning:\u001b[0m")
print("\u001b[32;1mIris:\u001b[0m")
# predictions = np.array([nn_iris.predict(x) for x in test_iris])
predictions = decision_tree_iris_pruning.predict(test_iris)
print_prediction_summary(predictions, test_labels_iris)

print("\n\u001b[32;1mWine:\u001b[0m")
# predictions = np.array([nn_wine.predict(x) for x in test_wine])
predictions = decision_tree_wine_pruning.predict(test_wine)
print_prediction_summary(predictions, test_labels_wine)

print("\n\u001b[32;1mAbalones:\u001b[0m")
# predictions = np.array([nn_abalones.predict(x) for x in test_abalone])
predictions = decision_tree_abalones_pruning.predict(test_abalone)
print_prediction_summary(predictions, test_labels_abalone)

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
error_vs_neurons_iris = []
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
    error_vs_neurons_iris.append(1 - score)
    if best_size_iris is None or score > best_size_iris[1]:
        best_size_iris = (hidden_layers, score)
    print(f"Hidden layers: {hidden_layers}, Accuracy: {score}")

plot_error_vs_neurons(
    hidden_layers_sizes_to_test,
    error_vs_neurons_iris,
    "Iris",
)

# optimisation du nombre de neurones pour le dataset wine
print("\nNeural Network layer size for wine:")
error_vs_neurons_wine = []
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
    error_vs_neurons_wine.append(1 - score)
    if best_size_wine is None or score > best_size_wine[1]:
        best_size_wine = (hidden_layers, score)
    print(f"Hidden layers: {hidden_layers}, Accuracy: {score}")

plot_error_vs_neurons(
    hidden_layers_sizes_to_test,
    error_vs_neurons_wine,
    "Wine",
)

# optimisation du nombre de neurones pour le dataset abalones
print("\nNeural Network layer size for abalones:")
error_vs_neurons_abalones = []
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
    error_vs_neurons_abalones.append(1 - score)
    if best_size_abalones is None or score > best_size_abalones[1]:
        best_size_abalones = (hidden_layers, score)
    print(f"Hidden layers: {hidden_layers}, Accuracy: {score}")

plot_error_vs_neurons(
    hidden_layers_sizes_to_test,
    error_vs_neurons_abalones,
    "Abalones",
)

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
error_vs_depth_iris = []
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
    error_vs_depth_iris.append(1 - score)
    if best_depth_iris is None or score > best_depth_iris[1]:
        best_depth_iris = (hidden_layers, score)
    print(f"Hidden layers: {hidden_layers}, Accuracy: {score}")

plot_error_vs_depth(
    [len(x) for x in hidden_layers_to_test_iris],
    error_vs_depth_iris,
    "Iris",
    bsi,
)


# optimisation de la profondeur pour le dataset wine
print("\nNeural Network depth for wine:")
error_vs_depth_wine = []
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
    error_vs_depth_wine.append(1 - score)
    if best_depth_wine is None or score > best_depth_wine[1]:
        best_depth_wine = (hidden_layers, score)
    print(f"Hidden layers: {hidden_layers}, Accuracy: {score}")

plot_error_vs_depth(
    [len(x) for x in hidden_layers_to_test_wine],
    error_vs_depth_wine,
    "Wine",
    bsw,
)

# optimisation de la profondeur pour le dataset abalones
print("\nNeural Network depth for abalones:")
error_vs_depth_abalones = []
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
    error_vs_depth_abalones.append(1 - score)
    if best_depth_abalones is None or score > best_depth_abalones[1]:
        best_depth_abalones = (hidden_layers, score)
    print(f"Hidden layers: {hidden_layers}, Accuracy: {score}")

plot_error_vs_depth(
    [len(x) for x in hidden_layers_to_test_abalones],
    error_vs_depth_abalones,
    "Abalones",
    bsa,
)

print(
    f"\nBest hidden layers for iris: {best_depth_iris[0]}, Accuracy: {best_depth_iris[1]}"
)
print(
    f"Best hidden layers for wine: {best_depth_wine[0]}, Accuracy: {best_depth_wine[1]}"
)
print(
    f"Best hidden layers for abalones: {best_depth_abalones[0]}, Accuracy: {best_depth_abalones[1]}"
)

nn_iris_layers = [4, *best_depth_iris[0], 3]
nn_wine_layers = [11, *best_depth_wine[0], 2]
nn_abalones_layers = [8, *best_depth_abalones[0], 3]
nn_iris = NeuralNet(nn_iris_layers, learning_rate=0.1, activation="relu")
nn_wine = NeuralNet(nn_wine_layers, learning_rate=0.01, activation="relu")
nn_abalones = NeuralNet(nn_abalones_layers, learning_rate=0.01, activation="relu")


print("\n\u001b[31;1mTrain Neural Network:\u001b[0m")
print("\u001b[32;1mIris:\u001b[0m")
time_nn_train_iris = time.time()
nn_iris.train(train_iris.T, np.eye(3)[train_labels_iris].T, epochs=1000)
time_nn_train_iris = time.time() - time_nn_train_iris
# predictions = np.array([nn_iris.predict(x) for x in train_iris])
predictions = nn_iris.predict(train_iris.T)
print_prediction_summary(predictions, train_labels_iris)

print("\n\u001b[32;1mWine:\u001b[0m")
time_nn_train_wine = time.time()
nn_wine.train(train_wine.T, np.eye(2)[train_labels_wine].T, epochs=3000)
time_nn_train_wine = time.time() - time_nn_train_wine
# predictions = np.array([nn_wine.predict(x) for x in train_wine])
predictions = nn_wine.predict(train_wine.T)
print_prediction_summary(predictions, train_labels_wine)

print("\n\u001b[32;1mAbalones:\u001b[0m")
time_nn_train_abalones = time.time()
nn_abalones.train(train_abalone.T, np.eye(3)[train_labels_abalone].T, epochs=8000)
time_nn_train_abalones = time.time() - time_nn_train_abalones
# predictions = np.array([nn_abalones.predict(x) for x in train_abalone])
predictions = nn_abalones.predict(train_abalone.T)
print_prediction_summary(predictions, train_labels_abalone)

print("\n\u001b[31;1mTest Neural Network:\u001b[0m")
print("\u001b[32;1mIris:\u001b[0m")
# predictions = np.array([nn_iris.predict(x) for x in test_iris])
predictions = nn_iris.predict(test_iris.T)
print_prediction_summary(predictions, test_labels_iris)

print("\n\u001b[32;1mWine:\u001b[0m")
# predictions = np.array([nn_wine.predict(x) for x in test_wine])
predictions = nn_wine.predict(test_wine.T)
print_prediction_summary(predictions, test_labels_wine)

print("\n\u001b[32;1mAbalones:\u001b[0m")
# predictions = np.array([nn_abalones.predict(x) for x in test_abalone])
predictions = nn_abalones.predict(test_abalone.T)
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
decision_tree_iris_pred = decision_tree_iris.predict(test_iris[5])
time_decision_tree_predict_iris = time.time() - time_decision_tree_predict_iris
accuracy_decision_tree_iris = decision_tree_iris.evaluate(test_iris, test_labels_iris)

# time predict decision tree wine
time_decision_tree_predict_wine = time.time()
decision_tree_wine_pred = decision_tree_wine.predict(test_wine[5])
time_decision_tree_predict_wine = time.time() - time_decision_tree_predict_wine
accuracy_decision_tree_wine = decision_tree_wine.evaluate(test_wine, test_labels_wine)

# time predict decision tree abalones
time_decision_tree_predict_abalones = time.time()
decision_tree_abalones_pred = decision_tree_abalones.predict(test_abalone[5])
time_decision_tree_predict_abalones = time.time() - time_decision_tree_predict_abalones
accuracy_decision_tree_abalones = decision_tree_abalones.evaluate(
    test_abalone, test_labels_abalone
)

# time predict neural network iris
time_nn_predict_iris = time.time()
nn_iris_pred = nn_iris.predict(test_iris[5])
time_nn_predict_iris = time.time() - time_nn_predict_iris
accuracy_nn_iris = nn_iris.evaluate(test_iris.T, np.eye(3)[test_labels_iris].T)

# time predict neural network wine
time_nn_predict_wine = time.time()
nn_wine_pred = nn_wine.predict(test_wine[5])
time_nn_predict_wine = time.time() - time_nn_predict_wine
accuracy_nn_wine = nn_wine.evaluate(test_wine.T, np.eye(2)[test_labels_wine].T)

# time predict neural network abalones
time_nn_predict_abalones = time.time()
nn_abalones_pred = nn_abalones.predict(test_abalone[5])
time_nn_predict_abalones = time.time() - time_nn_predict_abalones
accuracy_nn_abalones = nn_abalones.evaluate(
    test_abalone.T, np.eye(3)[test_labels_abalone].T
)

print("\n\u001b[31;1mComparison of the 4 models:\u001b[0m")
print("\n\u001b[32;1mIris:\u001b[0m")
print("\n\u001b[33;1mTraining time:\u001b[0m")
print(f"\tKnn: {time_knn_train_iris}")
print(f"\tNaive Bayes: {time_naive_bayes_train_iris}")
print(f"\tDecision Tree: {time_decision_tree_train_iris}")
print(f"\tNeural Network: {time_nn_train_iris}")

print("\n\u001b[33;1mPrediction time:\u001b[0m")
print(f"\tKnn: {time_knn_predict_iris}")
print(f"\tNaive Bayes: {time_naive_bayes_predict_iris}")
print(f"\tDecision Tree: {time_decision_tree_predict_iris}")
print(f"\tNeural Network: {time_nn_predict_iris}")

print("\n\u001b[33;1mAccuracy:\u001b[0m")
print(f"\tKnn: {accuracy_knn_iris}")
print(f"\tNaive Bayes: {accuracy_naive_bayes_iris}")
print(f"\tDecision Tree: {accuracy_decision_tree_iris}")
print(f"\tNeural Network: {accuracy_nn_iris}")

print("\n\u001b[32;1mWine:\u001b[0m")
print("\n\u001b[33;1mTraining time:\u001b[0m")
print(f"\tKnn: {time_knn_train_wine}")
print(f"\tNaive Bayes: {time_naive_bayes_train_wine}")
print(f"\tDecision Tree: {time_decision_tree_train_wine}")
print(f"\tNeural Network: {time_nn_train_wine}")

print("\n\u001b[33;1mPrediction time:\u001b[0m")
print(f"\tKnn: {time_knn_predict_wine}")
print(f"\tNaive Bayes: {time_naive_bayes_predict_wine}")
print(f"\tDecision Tree: {time_decision_tree_predict_wine}")
print(f"\tNeural Network: {time_nn_predict_wine}")

print("\n\u001b[33;1mAccuracy:\u001b[0m")
print(f"\tKnn: {accuracy_knn_wine}")
print(f"\tNaive Bayes: {accuracy_naive_bayes_wine}")
print(f"\tDecision Tree: {accuracy_decision_tree_wine}")
print(f"\tNeural Network: {accuracy_nn_wine}")

print("\n\u001b[32;1mAbalones:\u001b[0m")
print("\n\u001b[33;1mTraining time:\u001b[0m")
print(f"\tKnn: {time_knn_train_abalones}")
print(f"\tNaive Bayes: {time_naive_bayes_train_abalones}")
print(f"\tDecision Tree: {time_decision_tree_train_abalones}")
print(f"\tNeural Network: {time_nn_train_abalones}")

print("\n\u001b[33;1mPrediction time:\u001b[0m")
print(f"\tKnn: {time_knn_predict_abalones}")
print(f"\tNaive Bayes: {time_naive_bayes_predict_abalones}")
print(f"\tDecision Tree: {time_decision_tree_predict_abalones}")
print(f"\tNeural Network: {time_nn_predict_abalones}")

print("\n\u001b[33;1mAccuracy:\u001b[0m")
print(f"\tKnn: {accuracy_knn_abalones}")
print(f"\tNaive Bayes: {accuracy_naive_bayes_abalones}")
print(f"\tDecision Tree: {accuracy_decision_tree_abalones}")
print(f"\tNeural Network: {accuracy_nn_abalones}")

"""
Vous allez definir une classe pour chaque algorithme que vous allez développer,
votre classe doit contenir au moins les 3 méthodes definies ici bas, 
	* train 	: pour entraîner le modèle sur l'ensemble d'entrainement.
	* predict 	: pour prédire la classe d'un exemple donné.
	* evaluate 		: pour evaluer le classifieur avec les métriques demandées. 
vous pouvez rajouter d'autres méthodes qui peuvent vous etre utiles, mais la correction
se fera en utilisant les méthodes train, predict et evaluate de votre code.
"""

from classifieur import Classifier

import numpy as np


# le nom de votre classe
# DecisionTree pour l'arbre de décision
# NeuralNet pour le réseau de neurones


class NeuralNet(Classifier):  # nom de la class à changer
    def __init__(self, layer_sizes, learning_rate=0.01, activation="sigmoid"):
        self.layer_sizes = layer_sizes
        self.learning_rate = learning_rate
        self.activation = activation
        self.weights, self.biases = self._initialize_weights()

    def _initialize_weights(self):
        np.random.seed(42)  # For reproducibility
        weights = []
        biases = []
        for i in range(1, len(self.layer_sizes)):
            weights.append(
                np.random.randn(self.layer_sizes[i], self.layer_sizes[i - 1]) * 0.1
            )
            biases.append(np.zeros((self.layer_sizes[i], 1)))
        return weights, biases

    def _relu(self, Z):
        return np.maximum(0, Z)

    def _relu_derivative(self, Z):
        return Z > 0

    def _sigmoid(self, Z):
        return 1 / (1 + np.exp(-Z))

    def _sigmoid_derivative(self, A):
        return A * (1 - A)

    def _softmax(self, Z):
        expZ = np.exp(Z - np.max(Z))
        return expZ / expZ.sum(axis=0, keepdims=True)

    def _activation_fn(self, Z):
        if self.activation == "relu":
            return self._relu(Z)
        elif self.activation == "sigmoid":
            return self._sigmoid(Z)

    def _activation_derivative(self, A):
        if self.activation == "relu":
            return self._relu_derivative(A)
        elif self.activation == "sigmoid":
            return self._sigmoid_derivative(A)

    def _forward_propagation(self, X):
        activations = [X]
        A = X

        for i in range(len(self.layer_sizes) - 1):
            W = self.weights[i]
            b = self.biases[i]
            Z = np.dot(W, A) + b
            if i == len(self.layer_sizes) - 2:
                A = self._softmax(Z)
            else:
                A = self._activation_fn(Z)
            activations.append(A)

        return A, activations

    def _backward_propagation(self, activations, Y):
        grads_W = []
        grads_b = []
        m = Y.shape[1]
        L = len(self.layer_sizes) - 1
        A_final = activations[-1]

        dZ = A_final - Y
        grads_W.append(np.dot(dZ, activations[-2].T) / m)
        grads_b.append(np.sum(dZ, axis=1, keepdims=True) / m)

        for i in range(L - 2, -1, -1):
            dA = np.dot(self.weights[i + 1].T, dZ)
            dZ = dA * self._activation_derivative(activations[i + 1])
            grads_W.insert(0, np.dot(dZ, activations[i].T) / m)
            grads_b.insert(0, np.sum(dZ, axis=1, keepdims=True) / m)

        return grads_W, grads_b

    def _update_parameters(self, grads_W, grads_b):
        for i in range(len(self.layer_sizes) - 1):
            self.weights[i] -= self.learning_rate * grads_W[i]
            self.biases[i] -= self.learning_rate * grads_b[i]

    def train(self, X, Y, epochs=1000):
        for epoch in range(epochs):
            Y_pred, activations = self._forward_propagation(X)
            grads_W, grads_b = self._backward_propagation(activations, Y)
            self._update_parameters(grads_W, grads_b)
            epsilon = 1e-8
            loss = -np.mean(
                Y * np.log(Y_pred + epsilon) + (1 - Y) * np.log(1 - Y_pred + epsilon)
            )

            if epoch % 100 == 0:
                print(f"Epoch {epoch}/{epochs}, Loss: {loss}")

    def predict(self, X):
        Y_pred, _ = self._forward_propagation(X)
        return np.argmax(Y_pred, axis=0)

    def evaluate(self, X, Y):
        predictions = self.predict(X)
        accuracy = np.mean(predictions == np.argmax(Y, axis=0))
        return accuracy

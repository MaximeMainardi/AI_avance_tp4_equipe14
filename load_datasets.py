import numpy as np
import random


def load_iris_dataset(train_ratio):
    """Cette fonction a pour but de lire le dataset Iris

    Args:
        train_ratio: le ratio des exemples qui vont etre attribués à l'entrainement,
        le reste des exemples va etre utilisé pour les tests.
        Par exemple : si le ratio est 50%, il y aura 50% des exemple (75 exemples) qui vont etre utilisés
        pour l'entrainement, et 50% (75 exemples) pour le test.

    Retours:
        Cette fonction doit retourner 4 matrices de type Numpy: train, train_labels, test, et test_labels

        - train : une matrice numpy qui contient les exemples qui vont etre utilisés pour l'entrainement, chaque 
        ligne dans cette matrice représente un exemple d'entrainement.

        - train_labels : contient les labels (ou les étiquettes) pour chaque exemple dans train, de telle sorte
          que : train_labels[i] est l'etiquette pour l'exemple train[i]

        - test : une matrice numpy qui contient les exemples qui vont etre utilisés pour le test, chaque 
        ligne dans cette matrice représente un exemple de test.

        - test_labels : contient les étiquettes pour chaque exemple dans test, de telle sorte
          que : test_labels[i] est l'etiquette pour l'exemple test[i]
    """

    # Pour avoir les meme nombres aléatoires à chaque initialisation.
    random.seed(1)

    # Vous pouvez utiliser des valeurs numériques pour les différents types de classes, tel que :
    conversion_labels = {'Iris-setosa': 0,
                         'Iris-versicolor': 1, 'Iris-virginica': 2}

    # Le fichier du dataset est dans le dossier datasets en attaché
    f = open('datasets/bezdekIris.data', 'r')

    # TODO : le code ici pour lire le dataset

    # REMARQUE très importante :
    # remarquez bien comment les exemples sont ordonnés dans
    # le fichier du dataset, ils sont ordonnés par type de fleur, cela veut dire que
    # si vous lisez les exemples dans cet ordre et que si par exemple votre ration est de 60%,
    # vous n'allez avoir aucun exemple du type Iris-virginica pour l'entrainement, pensez
    # donc à utiliser la fonction random.shuffle pour melanger les exemples du dataset avant de séparer
    # en train et test.
    dataset = np.loadtxt(f, delimiter=',', dtype=float, converters={
                         4: lambda s: conversion_labels[s]})
    f.close()

    np.random.shuffle(dataset)

    X = dataset[:, :4]
    y = dataset[:, 4]
    y = y.astype(int)
    length = len(y)
    train_length = int(length * train_ratio)

    train = X[:train_length]
    train_labels = y[:train_length]

    test = X[train_length:]
    test_labels = y[train_length:]

    # Tres important : la fonction doit retourner 4 matrices (ou vecteurs) de type Numpy.
    return (train, train_labels, test, test_labels)


def load_wine_dataset(train_ratio):
    """Cette fonction a pour but de lire le dataset Binary Wine quality

    Args:
        train_ratio: le ratio des exemples (ou instances) qui vont servir pour l'entrainement,
        le rest des exemples va etre utilisé pour les test.

    Retours:
        Cette fonction doit retourner 4 matrices de type Numpy: train, train_labels, test, et test_labels

        - train : une matrice numpy qui contient les exemples qui vont etre utilisés pour l'entrainement, chaque 
        ligne dans cette matrice représente un exemple d'entrainement.

        - train_labels : contient les labels (ou les étiquettes) pour chaque exemple dans train, de telle sorte
          que : train_labels[i] est l'etiquette pour l'exemple train[i]

        - test : une matrice numpy qui contient les exemples qui vont etre utilisés pour le test, chaque 
        ligne dans cette matrice représente un exemple de test.

        - test_labels : contient les étiquettes pour chaque exemple dans test, de telle sorte
          que : test_labels[i] est l'etiquette pour l'exemple test[i]
    """

    # Pour avoir les meme nombres aléatoires à chaque initialisation.
    random.seed(1)

    # Le fichier du dataset est dans le dossier datasets en attaché
    f = open('datasets/binary-winequality-white.csv', 'r')

    # TODO : le code ici pour lire le dataset
    dataset = np.loadtxt(f, delimiter=',', dtype=float)
    random.shuffle(dataset)
    X = dataset[:, :11]
    y = dataset[:, 11]
    y = y.astype(int)
    length = len(y)
    train_length = int(length * train_ratio)
    train = X[:train_length]
    train_labels = y[:train_length]
    test = X[train_length:]
    test_labels = y[train_length:]
    f.close()

    # La fonction doit retourner 4 structures de données de type Numpy.
    return (train, train_labels, test, test_labels)


def load_abalone_dataset(train_ratio):
    """
    Cette fonction a pour but de lire le dataset Abalone-intervalles

    Args:
        train_ratio: le ratio des exemples (ou instances) qui vont servir pour l'entrainement,
        le rest des exemples va etre utilisé pour les test.

    Retours:
        Cette fonction doit retourner 4 matrices de type Numpy: train, train_labels, test, et test_labels

        - train : une matrice numpy qui contient les exemples qui vont etre utilisés pour l'entrainement, chaque 
        ligne dans cette matrice représente un exemple d'entrainement.

        - train_labels : contient les labels (ou les étiquettes) pour chaque exemple dans train, de telle sorte
          que : train_labels[i] est l'etiquette pour l'exemple train[i]

        - test : une matrice numpy qui contient les exemples qui vont etre utilisés pour le test, chaque 
        ligne dans cette matrice représente un exemple de test.

        - test_labels : contient les étiquettes pour chaque exemple dans test, de telle sorte
          que : test_labels[i] est l'etiquette pour l'exemple test[i]
    """
    f = open('datasets/abalone-intervalles.csv',
             # La fonction doit retourner 4 matrices (ou vecteurs) de type Numpy.
             'r')
    # Male : 0, Female : 1, Infant : 2
    conversion_sexe = {'M': 0, 'F': 1, 'I': 2}

    dataset = np.loadtxt(f, delimiter=',', dtype=float, converters={
                         0: lambda s: conversion_sexe[s]})
    random.shuffle(dataset)
    X = dataset[:, :8]
    # make the first column of X to be integers in X
    X[:, 0] = X[:, 0].astype(int)
    y = dataset[:, 8]
    y = y.astype(int)
    length = len(y)
    train_length = int(length * train_ratio)
    train = X[:train_length]
    train_labels = y[:train_length]
    test = X[train_length:]
    test_labels = y[train_length:]
    f.close()
    return (train, train_labels, test, test_labels)
import math
import numpy as np
import preprocessor
import symbols
import kNN
import neural_network
import logistic_regression

def predict(symbImgs, topology):
    # Flatten using knn first
    knn = kNN.kNN()
    flat_syms = knn.flatten_symbols(symbImgs)
    binary_symbs = preprocessor.binarize(flat_syms)
    knn.load_MNIST()
    m_data = logistic_regression.flatten(knn.mnist_data)
    m_labels = knn.mnist_labels
    nn = neural_network.NeuralNet(topology)
    print("> Fitting NN")
    nn.fit(m_data[:100], m_labels[:100])
    results = []
    print("> Predicting NN")
    for i in range(1, len(binary_symbs) + 1):
        if i % 1000 == 0:
            print(i)
        results.append(nn.predict(binary_symbs[i - 1:i]))
    return results
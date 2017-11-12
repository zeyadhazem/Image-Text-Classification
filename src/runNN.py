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
    binary_symbs = preprocessor.binarize(np.array(flat_syms))
    knn.load_MNIST()
    m_data = logistic_regression.flatten(knn.mnist_data)
    m_labels = knn.mnist_labels
    fb_symbs = logistic_regression.fix_data_len(logistic_regression.flatten(binary_symbs))
    nn = neural_network.NeuralNet(topology)
    print("Fitting NN")
    nn.fit(m_data[4800:], m_labels[4800:])
    results = []
    print("Predicting NN")
    for i in range(1, len(fb_symbs) + 1):
        if i % 1000 == 0:
            print(i)
        results.append(nn.predict(fb_symbs[i - 1:i]))
    return results
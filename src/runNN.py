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
    flattened = logistic_regression.flatten(binary_symbs)
    knn.load_MNIST()
    m_data = logistic_regression.flatten(knn.mnist_data)
    m_labels = knn.mnist_labels
    nn = neural_network.NeuralNet(topology)
    print("> Fitting NN")
    m_labels = toArray(m_labels[:10000])
    nn.fit(m_data[:10000], m_labels)

    results = []
    print("> Predicting NN")
    for i in range(1, len(flattened) + 1):
        if i % 1000 == 0:
            print(i)
        results.append(nn.predict(flattened[i - 1:i]))
    return results


# Takes in an an array of arrays
def apply_nn_operations(data):
    results = []
    for i in range(math.floor(len(data) / 3)):
        if i % 1000 == 0:
            print(i)
        els = data[i * 3:i * 3 + 3]
        digits = []
        ops = []
        for e in els:
            symbol = e.index(max(e))
            if symbol < 10:
                digits.append(e)
            else:
                ops.append(e)
        # If there are 3 digits and no operators
        if len(digits) >= 3:
            maxOp = 0
            newOp = 0
            for d in digits:
                score = max(d[9:])
                if score > maxOp:
                    newOp = d
                    maxOp = score
            digits.remove(newOp)
            ops.append(newOp)

        # If there's more than one operator do the opposite of above
        while len(ops) > 1:
            maxDig = 0
            newDig = 0
            for o in ops:
                score = max(o[:10])
                if score > maxDig:
                    newDig = o
                    maxDig = score
            ops.remove(newDig)
            digits.append(newDig)

        # Max value out of 10,11
        op = ops[0].index(max(ops[0][9:]))
        # Max value out of 0-9
        dig1 = digits[0].index(max(digits[0][:10]))
        dig2 = digits[1].index(max(digits[1][:10]))
        if op == 10:
            results.append(dig1 + dig2)
        else:
            results.append(dig1 * dig2)
    return results

def toArray(data):
    modData = []
    for row in data:
        aArray = [0]*12
        aArray[(row)] = 1
        modData.append(aArray)
    return modData


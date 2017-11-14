from __future__ import division, print_function, absolute_import

import numpy as np
import matplotlib.pyplot as plt
import preprocessor
import symbols
import runNN
import logistic_regression

import tflearn
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.normalization import local_response_normalization
from tflearn.layers.estimator import regression

import pandas as pd

from numpy import genfromtxt

# Data loading and preprocessing
import tflearn.datasets.mnist as mnist

import scipy.ndimage as ndimage
from scipy.misc import imresize

print("> Reading the text file")
x = np.loadtxt("../data/test_x.csv", delimiter=",") # load from text

print("> Shaping data into 64 * 64 pixel images")
images = x.reshape(-1, 64, 64).astype(int) # reshape

print("> Transforming images into binary images")
binaryImages = preprocessor.binarize(images)

print("> Extracting the centers of symbols from images")
symbolCenters = symbols.findCenters(binaryImages, 3)

print("> Getting the images of the symbols")

symbol_list = symbols.getSymbolImages(binaryImages, symbolCenters, 28, 28)

print("> Grouping symbols by similarity")
# result = runNN.predict(symbol_list, [784, 784, 12])

# print(result)
# print("> Apply operations")
# maxedResult = []
# for arr in result:
#     max = 0
#     for aInt in arr:
#         if(aInt > max):
#             max = aInt
#     maxedResult.append(max)
# modifiedResults = runNN.apply_nn_operations(result)
#
# err = 0
# for i in range(len(modifiedResults)):
#     if(modifiedResults[i] == y[i]):
#         err += 0
#         print("    Correct: ", i, modifiedResults[i],y[i])
#     else:
#         print("    Incorrect: ", i, modifiedResults[i], y[i])
#         err += 1
# print("> Classification: ", (1.0-(err/len(modifiedResults)))*100,"%")


def rotate_images (images_list, class_list, rotationDegree):
    rotated = []
    for image in images_list:
        tempMat = imresize(ndimage.rotate(image, rotationDegree), (28,28))
        tempMat = normalize (tempMat)
        rotated.append(tempMat)

    return np.array(rotated)

def normalize (image):
    image = (image - np.min(np.abs(image))).astype(float)
    image = image/np.max(np.abs(image))
    return image

X, Y, testX, testY = mnist.load_data(one_hot=False)

X = X.reshape([-1, 28, 28])

my_data = np.loadtxt("../data/m_dat.csv", delimiter=",") # load from text
my_data = my_data.reshape([-1, 28, 28])

my_data = my_data [:9600] # Only letters

rotated_letters = []
for index_1 in range(0, len(my_data)):
    rotated_letters.append(normalize(np.transpose(my_data[index_1])))

letters_classes = np.loadtxt("../data/m_lab.csv", delimiter=",") # load from text
letters_classes = letters_classes [:9600] # Only letters

X = np.append(X, rotated_letters)

X = X.reshape([-1, 28, 28])

Y = np.append(Y, letters_classes)

Y = Y.astype(int)

rotations = []

rotations.append(rotate_images (X, Y, 30))
rotations.append(rotate_images (X, Y, -30))
# rotations.append(rotate_images (X, Y, 45))
# rotations.append(rotate_images (X, Y, -45))

temp = Y
for rotation in rotations:
    X = np.append(X, rotation)
    Y = np.append(Y, temp)

X = X.reshape([-1, 28, 28, 1])
testX = testX.reshape([-1, 28, 28, 1])

num_classes = 12
temp = np.zeros((len(Y), num_classes))
temp[np.arange(len(Y)), Y] = 1
Y = temp

temp = np.zeros((len(testY), num_classes))
temp[np.arange(len(testY)), testY] = 1
testY = temp

# -*- coding: utf-8 -*-

""" Convolutional Neural Network for MNIST dataset classification task.

References:
    Y. LeCun, L. Bottou, Y. Bengio, and P. Haffner. "Gradient-based
    learning applied to document recognition." Proceedings of the IEEE,
    86(11):2278-2324, November 1998.

Links:
    [MNIST Dataset] http://yann.lecun.com/exdb/mnist/

"""

# Building convolutional network
network = input_data(shape=[None, 28, 28, 1], name='input')
network = conv_2d(network, 32, 3, activation='relu', regularizer="L2")
network = max_pool_2d(network, 2)
network = local_response_normalization(network)
network = conv_2d(network, 64, 3, activation='relu', regularizer="L2")
network = max_pool_2d(network, 2)
network = local_response_normalization(network)
network = fully_connected(network, 128, activation='tanh')
network = dropout(network, 0.8)
network = fully_connected(network, 256, activation='tanh')
network = dropout(network, 0.8)
network = fully_connected(network, 12, activation='softmax')
network = regression(network, optimizer='adam', learning_rate=0.01,
                     loss='categorical_crossentropy', name='target')

# Training
model = tflearn.DNN(network, tensorboard_verbose=0)
model.fit({'input': X}, {'target': Y}, n_epoch=10,
           validation_set=({'input': testX}, {'target': testY}),
           snapshot_step=100,
          show_metric=True,
          run_id='convnet')

temp_list = symbol_list

all_operations = []
prediction_list = []

for ternary_symbol in symbol_list:
    operators_and_operands = []
    for symbol_image in ternary_symbol:
        symbol_image = symbol_image/np.max(np.abs(symbol_image))
        symbol_cnn_input = symbol_image.reshape([-1,28,28,1])

        pred = model.predict(symbol_cnn_input)
        prediction_list.append(np.ndarray.tolist(pred[0]))
        operators_and_operands.append(np.argmax(pred[0], axis=0))

    all_operations.append(operators_and_operands)

# Takes in an an array of arrays
def apply_nn_operations(data):
    results = []
    for i in range((int) (len(data)/3)):
        if i % 1000 == 0:
            print(i)
        els = data[i*3:i*3+3]
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

all_predictions = apply_nn_operations(prediction_list)

all_predictions = []

for operation in all_operations:
    operator = []
    operands = []

    for symbol_prediction in operation:
        if symbol_prediction >= 10:
            operator.append(symbol_prediction)
        else:
             operands.append(symbol_prediction)

    if len(operator) > 1:
        if (10 in operator): #'A' -> '4'
            operands.append(4)
            operator.remove(10)
        elif (11 in operator): # 'm' -> '3'
            operands.append(3)
            operator.remove(11)

    elif len(operands) > 2:
        if (2 in operands): # 'a'
            operator.append(10)
            operands.remove(2)
        elif (4 in operands): # 'A'
            operator.append(10)
            operands.remove(4)
        elif (3 in operands):
            operator.append(11)
            operands.remove(3)

    if len(operator) == 1:
        if (operator[0] == 10):
            all_predictions.append(operands[0] + operands[1])
        else:
            all_predictions.append(operands[0] * operands[1])

    elif len(operator) == 0:
        all_predictions.append(operands[0] + operands[1])

    else:
        all_predictions.append(1 + 1)

df = pd.DataFrame(all_predictions)
df.index = np.arange(1, len(df) + 1)
df.to_csv("../pred.csv", index=True, header=True)

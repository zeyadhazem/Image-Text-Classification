# -*- coding: utf-8 -*-

""" Convolutional Neural Network for MNIST dataset classification task.
References:
    Y. LeCun, L. Bottou, Y. Bengio, and P. Haffner. "Gradient-based
    learning applied to document recognition." Proceedings of the IEEE,
    86(11):2278-2324, November 1998.
Links:
    [MNIST Dataset] http://yann.lecun.com/exdb/mnist/
"""

from __future__ import division, print_function, absolute_import

import tflearn
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.normalization import local_response_normalization
from tflearn.layers.estimator import regression

# Data loading and preprocessing
import tflearn.datasets.mnist as mnist

import cv2
import numpy as np
import matplotlib.pyplot as plt
import scipy.ndimage as ndimage
from scipy.misc import imresize




def rotate_images (images_list, class_list, rotationDegree):
    rotated = []
    for image in images_list:
        rotated.append(imresize(ndimage.rotate(image, rotationDegree), (28,28)))

    return (np.array(rotated), class_list)


X, Y, testX, testY = mnist.load_data(one_hot=True)

# X = X.reshape([-1, 28, 28, 1])
X = X.reshape([-1, 28, 28])

# 1) Need to extend the MNIST dataset to support rotations (Rotate matrix)
# 2) Need to extend the testX to reflect that change
# 3) Need to normalize the values in my dataset to something between 0 and 1

X_10, Y_10 = rotate_images (X, Y, 10)
X_n10, Y_n10 = rotate_images (X, Y, -10)

X = np.append(X, np.array(X_10))
Y = np.concatenate((Y, Y))

X = np.append(X, np.array(X_10))
Y = np.concatenate((Y, Y))


X = X.reshape([-1, 28, 28])


plt.imshow(X[55000], cmap='gray')
plt.show()
plt.imshow(X[110000], cmap='gray')
plt.show()
plt.imshow(X[1], cmap='gray')
plt.show()
plt.imshow(X[55001], cmap='gray')
plt.show()
plt.imshow(X[110001], cmap='gray')
plt.show()




testX = testX.reshape([-1, 28, 28, 1])

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
network = fully_connected(network, 10, activation='softmax')
network = regression(network, optimizer='adam', learning_rate=0.01,
                     loss='categorical_crossentropy', name='target')

# Training
model = tflearn.DNN(network, tensorboard_verbose=0)
model.fit({'input': X}, {'target': Y}, n_epoch=20,
           validation_set=({'input': testX}, {'target': testY}),
           snapshot_step=100, show_metric=True, run_id='convnet_mnist')

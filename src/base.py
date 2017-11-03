import numpy as np
import matplotlib.pyplot as plt
import preprocessor
import symbols
from kNN import kNN
from neural_network import NeuralNet

print("> Reading the text file")
x = np.loadtxt("../data/temp.csv", delimiter=",") # load from text

print("> Shaping data into 64 * 64 pixel images")
images = x.reshape(-1, 64, 64).astype(int) # reshape

print("> Transforming images into binary images")
binaryImages = preprocessor.binarize(images)

print("> Displaying binary images")
# for i in range(0, 1):
#     plt.imshow(np.uint8(binaryImages[i]), cmap='gray')
#     plt.show()

print("> Extracting the centers of symbols from images")
symbolCenters = symbols.findCenters(binaryImages, 3)

print("> Getting the images of the symbols")
symbols.getSymbolImages(binaryImages, symbolCenters, 30, 30)

print("> Grouping symbols by similarity")
groupingClassifier = kNN()
groupingClassifier2 = NeuralNet()

print("> Create grouping logic now that symbols are separated and validate")
print("> Do all of the above for the test set")
print("> Predict")


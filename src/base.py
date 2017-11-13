import numpy as np
import matplotlib.pyplot as plt
import preprocessor
import symbols
import kNN
import runNN
import logistic_regression

print("> Reading the text file")
x = np.loadtxt("../data/train_x.csv", delimiter=",") # load from text
y = np.loadtxt("../data/train_y.csv", delimiter=",")

print("> Shaping data into 64 * 64 pixel images")
images = x.reshape(-1, 64, 64).astype(int) # reshape

print("> Transforming images into binary images")
binaryImages = preprocessor.binarize(images)

# print("> Displaying binary images")
# # for i in range(0, 1):
# #     plt.imshow(np.uint8(binaryImages[i]), cmap='gray')
# #     plt.show()

print("> Extracting the centers of symbols from images")
symbolCenters = symbols.findCenters(binaryImages, 3)

print("> Getting the images of the symbols")
symbImgs = symbols.getSymbolImages(binaryImages, symbolCenters, 28, 28)

print("> Grouping symbols by similarity")
#groupingClassifier = kNN()
#groupingClassifier2 = NeuralNet()
result = runNN.predict(symbImgs, [784, 784, 12])

print(result)
print("> Apply operations")
maxedResult = []
for arr in result:
    max = 0
    for aInt in arr:
        if(aInt > max):
            max = aInt
    maxedResult.append(max)
modifiedResults = runNN.apply_nn_operations(result)

err = 0
for i in range(len(modifiedResults)):
    if(modifiedResults[i] == y[i]):
        err += 0
        print("    Correct: ", i, modifiedResults[i],y[i])
    else:
        print("    Incorrect: ", i, modifiedResults[i], y[i])
        err += 1
print("> Classification: ", (1.0-(err/len(modifiedResults)))*100,"%")

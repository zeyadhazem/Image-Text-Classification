from classifier import Classifier
import math
import numpy as np

class NeuralNet (Classifier):
    """
    This class will use Neural nets to find the similar symbols from a list of symbols
    """
    def __init__(self, topology):
        global net
        net = Network(topology)  # setup a network with the above topology
        Perceptron.learningRate = 0.09
        Perceptron.momentum = 0.01

    def fit(self, X, y):
        while True:
            err = 0
            inputs = X
            outputs = y
            for i in range(len(inputs)):
                net.setInput(inputs[i])
                net.feedForword()
                net.backPropagate(outputs[i])
                err = err + net.getError(outputs[i])
            print("error: ", err)
            if err < 0.01:
                break

    def predict(self, test_set):
        while True:
            test = test_set
            '''
            for i in range(len(test)):
                net.setInput(test)
                net.feedForword()
                print(net.getResults())
            '''
            a = input("type 1st input :")
            b = input("type 2nd input :")
            net.setInput([a, b])
            net.feedForword()
            print(net.getBinResults())

class Connection: # simple class for defining connection between perceptrons 
    def __init__(self, connectedNeuron):
        self.connectedNeuron = connectedNeuron
        self.weight = np.random.normal() #assign a random weight
        self.dWeight = 0.0


class Perceptron:
    learningRate = 0.001
    momentum = 0.01

    def __init__(self, layer):
        self.connects = []
        self.error = 0.0
        self.gradient = 0.0
        self.output = 0.0
        if layer is None:
            pass
        else:
            for perceptron in layer:
                connection = Connection(perceptron)
                self.connects.append(connection)

    def addError(self, err): # for backprop
        self.error = self.error + err

    def sigmoid(self, x): #activation function
        return 1 / (1 + math.exp(-x * 1.0))

    def dSigmoid(self, x): #derivative of sigmoid
        return x * (1.0 - x)

    def feedForword(self): #check for connected neurons, if no, it is input neuron
        sumOutput = 0
        if len(self.connects) == 0:
            return
        for aX in self.connects:
            sumOutput = sumOutput + aX.connectedNeuron.getOutput() * aX.weight
        self.output = self.sigmoid(sumOutput)

    def backPropagate(self): # backpropogation algorithm found at https://thecodacus.com/, wikipedia backpropagate
        self.gradient = self.error * self.dSigmoid(self.output);
        for aX in self.connects:
            aX.dWeight = Perceptron.learningRate * (
            aX.connectedNeuron.output * self.gradient) + self.momentum * aX.dWeight;
            aX.weight = aX.weight + aX.dWeight;
            aX.connectedNeuron.addError(aX.weight * self.gradient);
        self.error = 0;


# getter setters
    def setError(self, err):
        self.error = err

    def setOutput(self, output):
        self.output = output

    def getOutput(self):
        return self.output

class Network:
    def __init__(self, topology):
        self.layers = []
        for numOfPerceptron in topology:
            layer = []
            for i in range(numOfPerceptron):
                if (len(self.layers) == 0):
                    layer.append(Perceptron(None))
                else:
                    layer.append(Perceptron(self.layers[-1]))
            layer.append(Perceptron(None))
            layer[-1].setOutput(1)
            self.layers.append(layer)

    def setInput(self, inputs):
        for i in range(len(inputs)):
            self.layers[0][i].setOutput(inputs[i])

    def feedForword(self): #do feed forward
        for layer in self.layers[1:]:
            for perceptron in layer:
                perceptron.feedForword();

    def backPropagate(self, target):
        for i in range(len(target)):
            self.layers[-1][i].setError(target[i] - self.layers[-1][i].getOutput())
        for layer in self.layers[::-1]:
            for perceptron in layer:
                perceptron.backPropagate()

    def getError(self, target):
        err = 0
        for i in range(len(target)):
            e = (target[i] - self.layers[-1][i].getOutput())
            err = err + e ** 2
        err = err / len(target)
        err = math.sqrt(err)
        return err

    def getResults(self):
        output = []
        for perceptron in self.layers[-1]:
            output.append(perceptron.getOutput())
        output.pop()
        return output

    def getBinResults(self):
        output = []
        for perceptron in self.layers[-1]:
            o = perceptron.getOutput()
            if (o > 0.5):
                o = 1
            else:
                o = 0
            output.append(o)
        output.pop()
        return output

def main():
    sampleInputs = [[0, 1], [0, 1], [1, 0], [1, 1]]
    sampleOutputs = [[0, 0], [1, 0], [1, 0], [0, 1]]
    sampleTests = [[0, 1]]

    topology = [2,3,2]

    run = NeuralNet(topology)
    run.fit(sampleInputs, sampleOutputs)
    run.predict(sampleTests)


if __name__ == '__main__':
    main()
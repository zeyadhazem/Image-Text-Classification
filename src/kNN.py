from classifier import Classifier
import numpy as np
import copy
import cv2
import random
import preprocessor
import mnist as MNIST

class kNN (Classifier):
    """
    This class will use k nearest neighbours to find the similar symbols from a list of symbols
    """
    def __init__(self):
        self.knn = cv2.KNearest()
        self.results = []
        self.a = 10.0 #KNN can't train on letters so a = 10 and m = 11
        self.m = 11.0
        self.mnist_labels = []
        self.mnist_data = []
        return
    
    # The mnist data has to be loaded from the files with the right name
    # We have two files to load so we need two directories
    def load_MNIST(self):
        mdata = MNIST(path="../data/letter_MNIST")
        m_data, m_labels = mdata.load_training()
        m_data, m_labels = self.filter_operators(m_data, m_labels)
        m_data = np.array(m_data).reshape(-1,28,28).astype(float)
        m_data = preprocessor.binarize(m_data)
        self.mnist_labels += m_labels
        self.mnist_data += m_data
        mdata = MNIST(path="../data/digit_MNIST")
        m_data, m_labels = mdata.load_training()
        m_data = np.array(m_data).reshape(-1,28,28).astype(float)
        m_data = preprocessor.binarize(m_data)
        self.mnist_labels += m_labels
        self.mnist_data += m_data        

    def train_with_MNIST(self):
        self.fit(np.array(self.mnist_data), np.array(self.mnist_labels), r=28, c=28)

    # The training of the classifier
    # Accepts either a list or an np array
    # Works with the binaryImages array
    # Will work with digits as well but assumes that the digits are not in arrays of 3
    # So the digit arrays will be 3x the size of the original data
    def fit(self, x, y, r = 64, c = 64):
        if type(x).__module__ == np.__name__:
            train_data = copy.copy(x)
        else:
            train_data = np.array(x)
        if type(y).__module__ == np.__name__:
            train_labels = copy.copy(y)
        else:
            train_labels = np.array(y)
        train_data = self.convert(train_data, r, c)
        train_labels = train_labels.astype(np.float32)
        
        self.knn.train(train_data, train_labels)       

    def predict(self, test_set, neighbors = 5, r = 64,c = 64):
        if not type(test_set).__module__ == np.__name__:
            test_set = np.array(test_set)
        
        test_set = self.convert(test_set, r, c)
        ret, result, neighbors, dist = self.knn.find_nearest(test_set,k = neighbors)
        return result
    
    # Predicts the most probable symbols, every 3 symbols
    # Sees if it's possible to do an operation, if it is, do the operation
    # Otherwise, look through the neighbors until the most probable one is found
    def predict_digits(self, test_set, r = 28, c = 28, k = 5):
        
        # If the test_set is not an np array, convert to np array
        #if not type(test_set).__module__ == np.__name__:
        #    test_set = np.array(test_set)      
        try:
            test_set = self.convert(test_set, r, c)
        except:       
            for i in range(len(test_set)):
                el = test_set[i]
                test_set[i] = self.convert(test_set[i],len(el),len(el[0]))
        results = []
        # we classify every 3 examples
        # then use these symbols to do an operation
        # DCDC - make sure that I'm not somehow only taking 3 bits 
        for i in range(len(test_set)/3):
            # To know how far along we are
            if i % 1000 == 0:
                print(i)
            ret, result, neighbors, dist = self.knn.find_nearest(test_set[3*i:3*i+3], k = k)
            digits = []
            operators = []
            # The second tuple value shows which neighbor
            # Since it's the original value, it is not a neighbor so = -1
            for j in range(len(result)):
                d = result[j]
                # d > 9 means its 10 or 11 => an operator
                if d > 9:
                    operators.append((j,-1))
                else:
                    digits.append((j,-1))
            
            # This means that there are 2 or more operators
            # While loop in case of 3 operators
            while len(operators) > 1:
                newDig, theRest = self.MostProbable(operators, neighbors, dist, operator = False)
                operators = theRest
                digits.append(newDig)
            # This means that there are 3 digits
            if len(digits) > 2:
                newOp, theRest = self.MostProbable(digits, neighbors, dist, operator = True)
                digits = theRest
                operators.append(newOp)

            # find the operator again
            # If we aren't using a neighbor then the neighbor value is -1
            
            index1 = operators[0][0]
            index2 = operators[0][1]
            # This is the case that no operators were found
            if index1 == None or index2 == None:
                op = random.choice([self.a,self.m])
            else:
                if operators[0][1] >= 0:
                    op = neighbors[index1][index2]
                    print("Random op")
                else:
                    op = result[index1]
            index1 = digits[0][0]
            index2 = digits[0][1]
            # Less than 2 numbers found, so choose a random number
            if index1 == None or index2 == None:
                dig1 = random.choice([0,1,2,3,4,5,6,7,8,9])
                print("Random num 1")
            else:
                # If the index > -1, then it's a neighbor
                if index1 >= 0:
                    dig1 = int(neighbors[index1][index2])
                # Otherwise, it's the original value
                else:
                    dig1 = int(result[index1])
            index1 = digits[1][0]
            index2 = digits[1][1]
            if index1 == None or index2 == None:
                dig2 = random.choice([0,1,2,3,4,5,6,7,8,9])
                print("Random num 2")
            else:
                if index1 >= 0:
                    dig2 = int(neighbors[index1][index2])
                else:
                    dig2 = int(result[index1])

            if(op == self.a):
                results.append(dig1 + dig2)
            else:
                results.append(dig1 * dig2)
                
        self.results.append(results)
        return results
    
    # Looking through a list of digits to find the one that has the most
    # probability of being an operator
    # If operator is false, then look for most probable digit
    def MostProbable(self, indices, neighbors, distances, operator = True):
        idx = 0
        minDist = 10000000000 # Upper bound, none should be higher
        opIdx = 0
        nIdx = 0
        opFound = False
        options = ""

        if operator:
            options = [10,11]
        else:
            options = [0,1,2,3,4,5,6,7,8,9]
        while not opFound:
            # If we can't find an operator/digit
            if idx >= len(neighbors[0]) and not opFound:
                return (None, None), indices[:-1] # Remove the first index, return none
            for i in indices:
                # Indices store the index neighbor tuple
                n = int(neighbors[i[0]][idx])
                if n in options:
                    opFound = True
                    dist = distances[i[0]][idx]
                    if dist < minDist:
                        opIdx = i[0]
                        nIdx = idx # THe neighbor index
                        minDist = dist
            idx += 1
        # Returns 2 values, the first value is the new operator/digit
        # The second is the unchanged other digits/operators
        theRest = []
        for i in indices:
            if i[0] != opIdx:
                theRest.append(i)
            
        return (opIdx, nIdx), theRest

    # Filters out all the samples classified as 'a' or 'm'
    # Some of them are lowercase, not sure how to deal with them
    def filter_operators(self, data, labels):
        op_data = []
        op_labels = []
        for i in range(len(labels)):
            letter = labels[i]
            if letter == 1:
                op_data.append(data[i])
                op_labels.append(10)
            elif letter == 13:
                op_data.append(data[i])
                op_labels.append(11)
                
        return op_data, op_labels

    
    # Turns an np array of 64x64 matrices into one 4096 array of type float
    # r and c are the rows and columns
    def convert(self, data, r, c):        
        return data.reshape(-1, r * c).astype(np.float32)    

    # Takes an array of values of 3 and turns into array of 1 values
    def flatten_symbols(self, images):
        new_images = []
        for i in images:
            for img in i:
                new_images.append(img)
        return new_images

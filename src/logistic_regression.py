# -*- coding: utf-8 -*-
"""
Created on Fri Nov 10 18:16:03 2017

@author: Sacha Perry-Fagant
"""
import sklearn
from kNN import kNN
import random
import numpy as np
import preprocessor
from sklearn import linear_model

# turns a matrix into a long array
def flatten(data):
    all_data = []
    for d in data:
        dat = d.reshape(1, len(d) * len(d[0]))
        all_data.append(dat[0].tolist())
    return all_data

def floatify(data):
    all_data = []
    for d in data:
        row = []
        for c in d:
            row.append(float(c))
        all_data.append(row)
    return all_data

#Just loading the MNIST data here



# This is for data
def fix_data_len(data, length = 784):
    for i in range(len(data)):
        while len(data[i]) < length:
            data[i].append(0)

# Turn the flat symbols into a format that fits in the kNN
def npify( data):
    r_dat = np.array([[None]*len(data[0])]*len(data)).astype(np.float32)
    for i in range(len(data)):
        e = np.array(data[i]).astype(np.float32)
        r_dat[i] = e
    return r_dat

def predict():
    #symbImgs = symbols.getSymbolImages(binaryImages, symbolCenters, 28, 28)
    #Flatten using knn first    
    knn = kNN()
    print "knn"
    flat_syms = knn.flatten_symbols(symbImgs)
    print "flat"
    binary_symbs = preprocessor.binarize(np.array(flat_syms))
    print "bin"
    knn.load_MNIST()
    print "loaded"
    m_data = flatten(knn.mnist_data)
    m_labels = knn.mnist_labels
    print "Dat"
    f_symbs = flatten(binary_symbs)
    fix_data_len(f_symbs)
    print "fb"
    LR = linear_model.LogisticRegression()
    inst = LR.fit(m_data[4800:], m_labels[4800:])
    results = []
    for i in range(1,len(f_symbs) + 1):
        if i % 1000 == 0:
            print i
        results.append(inst.predict(f_symbs[i-1:i]))
    return results


def predict_whole_img(f_train, labels, f_test):
    #f_train = flatten(train)
    #f_test = flatten(test)
    LR = linear_model.LogisticRegression()
    inst = LR.fit(f_train,labels)
    results = []
    for i in range(1,len(f_test) + 1):
        if i % 1000 == 0:
            print i
        results.append(inst.predict(f_test[i-1:i]))
    return results

#results = inst.predict(fb_symbs)


# Might try with a bit more randomness
# Try reversing order of m and a to prioritize a more
def apply_operations(data):
    results = []
    for i in range(len(data)/3):
        if i % 1000 == 0:
            print i
        els = data[3*i:3*i+3]
        digits = []
        ops = []
        for e in els:
            if int(e) < 10:
                digits.append(int(e))
            else:
                ops.append(int(e))
        # If there are 3 digits and no operators      
        if len(digits) >= 3:
            # Then if there's a 3 change it into an m            
            if 3 in digits:
                digits.remove(3)
                ops.append(11)
            elif 9 in digits:
                # If there's a 9, change into an a
                digits.remove(9)
                ops.append(10)
            # Else get rid of a random digit and turn it into a random op            
            else:
                digits.remove(random.choice(digits))
                ops.append(random.choice([10,11]))
        # If there's more than one operator do the opposite of above
        while len(ops) > 1:
            if 11 in ops:
                ops.remove(11)
                digits.append(3) # 3 and m look similar
            elif 10 in ops:
                ops.remove(10)
                digits.append(9)      # a and 9 look similar
        if ops[0] == 10:
            results.append(digits[0] + digits[1])
        else:
            results.append(digits[0] * digits[1])
    return results
            
def save_mnist(m_data, m_labels, filename1, filename2):
    wfile = open(filename1, 'w')
    for i in range(len(m_data)):
        sent = ""
        for el in m_data[i]:
            sent += str(el)
            sent += ','
        #Delete the last ','
        sent = sent[:len(sent)-1]
        wfile.writelines(sent)
        wfile.writelines("\n")
    wfile.close()
    wf = open(filename2, 'w')
    for el in (m_labels):
        wf.write(str(el))
        wf.writelines("\n")
    wf.close()        


        
        
def save_results(data, filename):
    wfile = open(filename, 'w')
    wfile.write("Id,Label")
    for i in range(1, len(data) + 1):
        wfile.writelines("\n")        
        wfile.writelines(str(i) + "," + str(data[i-1]))
        
            

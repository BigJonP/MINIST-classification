import logging
import random
import math
import matplotlib.pyplot as plt
import numpy as np

from data_utils import *


#loading images and labels into variables via numpy, from the data_utils.py module 
trainingImages = np.array(load_images("train-images-idx3-ubyte.gz"))
trainingLabels = np.array(load_labels("train-labels-idx1-ubyte.gz"))
testImages = np.array(load_images("t10k-images-idx3-ubyte.gz"))
testLabels = np.array(load_labels("t10k-labels-idx1-ubyte.gz"))

class CustomNN:
    def __init__(self, input_size, hidden_layers, units_per_layer, activation_functions):
        self.input_size = input_size
        self.hidden_layers = hidden_layers
        self.units_per_layer = units_per_layer
        self.activation_functions = activation_functions
        self.weights = []
        self.biases = []
        self.init_weights_and_biases()

    def init_weights_and_biases(self):
        layer_sizes = [self.input_size] + self.units_per_layer

        for i in range(len(layer_sizes) - 1):
            self.weights.append(np.random.randn(layer_sizes[i], layer_sizes[i + 1]) * 0.01)
            self.biases.append(np.zeros((1, layer_sizes[i + 1])))
            
    def activation(self, x, af):
        if af == "1":
            #Sigmoid
            return 1 / (1 + np.exp(-x))
        elif af == "2":
            #ReLU
            return x * (x > 0)
    
    def activationDerivatiive(self, x, af):
        if af == "1":
            #Sigmoid derivative
            s = 1 / (1 + np.exp(-x))
            return s * (1.0 - s)
        elif af == "2":
            #ReLU derivative
            return (x >= 0) * 1
        
    def softmax(self, x):
        return np.exp(x) / np.sum(np.exp(x))
    
    def softmaxDerivative(self, x):
        #x is output numpy array from softmax
        s = x.reshape(-1, 1)
        return np.diagflat(s) - np.dot(s, s.T)

    def oneHotEncode(self, label):
        '''
        create new numpy array with shape (60000, 10)
        for every array in numpy array, traverse and change to 1 for corresponding num
        in label
        '''
        oneHotEncoded = np.zeros((60000, 10))
        for x, label in enumerate(label):
            oneHotEncoded[x, label] = 1
        
        return oneHotEncoded
    
    def dropout(self, x, dropout_rate):
        mask = np.random.binomial(1, 1 - dropout_rate, size=x.shape) / (1 - dropout_rate)
        self.dropout_masks.append(mask)
        return x * mask

        
    def cost(self, label, output):

        #Mean Squared Error
        cost = 1 / len(output) * np.sum((output - label) ** 2, axis = 0)

        return cost

    def crossEntropyLoss(self, actual, predicted):
        #actual value/label needs to be one-hot encoded
        loss = -np.sum(actual * np.log(predicted))
        return loss 
    
    def fit (self, lr, epochs, trainImg, trainLabels, activationFunc, dropout_rate=0.0):
        
        #required for cost/loss 
        oneHotLabels = self.oneHotEncode(trainLabels)

        #Extracting number of rows from oneHotLabels because that's the number of images
        numImages, columns = oneHotLabels.shape
        #Number of cycles required to create array to plot graph showing change in gradient
        numCycles  = numImages * epochs

        #Required to ensure the change in gradient matches with the current cycle of the training loop
        overallCycleNum = 0

        #empty arrays the size of the total cycles required for training, to store outputs for graphs
        layer1Grads = np.zeros((numCycles,1))
        layer2Grads = np.zeros((numCycles,1))
        w1Mags = np.zeros((numCycles,1))
        w2Mags = np.zeros((numCycles,1))
        a1Log = np.zeros((numCycles,1))
        a2Log = np.zeros((numCycles,1))
        
        #keep track of accuracies over epochs
        epoch_accuracies = np.zeros(epochs)

        #the number of times the NN correctly identifies a number during training
        correct = 0    
        
        for epoch in range (0, epochs):
            #Count of the current image being processed in the following for-loop
            #Required to plot graph
            imgCycle = 0

            for img, label in zip(trainImg, oneHotLabels):
                #Changing shape of img and label so they can be dot multipled
                img.shape += (1,)
                label.shape += (1,)

                #incrementing imgCycle so the gradients for the graphs are plotted in the correct positions
                imgCycle += 1

                ''' Forward prop '''
                #input layer to hidden layer - matrix dot multiplication of weights with input layer + bias
                z1 = np.dot(img.T, self.w1) + self.b1
                #feeding z1 into activation function for non-linearity
                a1 = self.activation(z1, activationFunc)
                
                #hidden layer to output layer - dot multiplication of weights with output of first layer + bias
                z2 = np.dot(a1, self.w2) + self.b2
                #feeding into activation function again
                a2 = self.activation(z2, activationFunc)
                


                ''' Softmax, then Cost/loss '''
                smOutput = self.softmax(a2)
                correct += int(np.argmax(a2) == np.argmax(label))
                
                ''' Back prop '''
                #delta a2 = dL/dS * dS/da2 * da2/dz2
                #dL/dS = cross entropy loss derivative = (smOutput - label.T)
                #dS/da2 = softmax derivative
                da2 = (smOutput - label.T) * self.activationDerivatiive(z2, activationFunc)

                #delta a1 = delta a2 * dz2/da1 * da1/dz1
                #da1/dz1 = w2
                da1 = da2.dot(self.w2.T) * self.activationDerivatiive(z1, activationFunc)

                #--Updating Weights and Biases--

                #dL/dw2 = da2 * dz2/dw2
                #da2 = dL/da2 * da2/dz2
                #dz2/dw2 = a1
                self.w2 -= lr * a1.T.dot(da2)

                #dL/db2 = da2 * dz2/db2
                #dz2/db2 = 1
                self.b2 -= lr * da2

                #dL/dw1 = delta a1 * dz1/dw1
                #dz1/dw1 = trainImg
                self.w1 -= lr * img.dot(da1)
                
                #dL/db1 = delta a1 * dz1/db1
                #dz1/db1 = 1
                self.b1 -= lr * da1

                #the total number of iterations of the training loop is the number of images * epochs
                #imgCycle is the current image training cycle within the greater for-loop of epochs
                overallCycleNum = (numImages * epoch) + imgCycle

                #gradient values updated
                if overallCycleNum < numCycles:
                    layer1Grads[overallCycleNum] = np.sum(img.dot(da1))
                    layer2Grads[overallCycleNum] = np.sum(a1.T.dot(da2))
                    w1Mags[overallCycleNum] = np.sum(np.absolute(self.w1))
                    w2Mags[overallCycleNum] = np.sum(np.absolute(self.w2))
                    a1Log[overallCycleNum] = np.mean(a1)
                    a2Log[overallCycleNum] = np.mean(a2)
            
            print("epoch: ", epoch)
            print(f"Accuracy: {round((correct / trainImg.shape[0]) * 100, 2)}%")
            #epoch_accuracies[epoch] = round((correct/trainImg.shape[0]) * 100,2)
            #reset so the accuracy is determined based on each epoch
            correct = 0
        
        #training returns gradients for plotting graphs
        print("overallCycleNum: ", overallCycleNum)
        gradients = np.array([layer1Grads, layer2Grads, w1Mags, w2Mags, a1Log, a2Log])
        #self.plot_metrics(gradients, epoch_accuracies, epochs, activationChoice, learningRate)
        return gradients
        
    def predict(self, x, activationFunc):
        z1 = np.dot(x, self.w1) + self.b1
        a1 = self.activation(z1, activationFunc)
        z2 = np.dot(a1, self.w2) + self.b2
        a2 = self.activation(z2, activationFunc)
        sm = self.softmax(a2)
        return sm
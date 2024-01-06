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
        if hidden_layers != len(units_per_layer) or hidden_layers + 1 != len(activation_functions):
            raise ValueError("Length of units_per_layer must be equal to hidden_layers, and activation_functions must be one more than hidden_layers")

        self.input_size = input_size
        self.hidden_layers = hidden_layers
        self.units_per_layer = units_per_layer
        self.activation_functions = activation_functions
        self.weights = []
        self.biases = []
        
        self.init_weights_and_biases()

    #trying out xavier initialisation for a dynamic number of hidden layers
    def init_weights_and_biases(self):
        #including the input layer size in the layer_sizes list
        layer_sizes = [self.input_size] + self.units_per_layer

        for i in range(len(layer_sizes) - 1):
            #xavier initialisation for weights
            weight = np.random.randn(layer_sizes[i], layer_sizes[i + 1]) * np.sqrt(2 / (layer_sizes[i] + layer_sizes[i + 1]))
            bias = np.zeros((1, layer_sizes[i + 1]))
            self.weights.append(weight)
            self.biases.append(bias)
            
    def activation(self, x, af):
        if af == "1":
            #Sigmoid
            return 1 / (1 + np.exp(-x))
        elif af == "2":
            #ReLU
            return x * (x > 0)
    
    def sigmoid(self,x):
        return 1/(1+np.exp(-x))
    
    def ReLU(self,x):
        return x * (x > 0)
    
    
    def activationDerivatiive(self, x, af):
        if af == "1":
            #Sigmoid derivative
            s = 1 / (1 + np.exp(-x))
            return s * (1.0 - s)
        elif af == "2":
            #ReLU derivative
            return (x >= 0) * 1
        
    def sigmoid_derivative(self,x):
        s = 1 / (1 + np.exp(-x))
        return s * (1.0 - s)
    
    def ReLU_derivative(self,x):
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
    
    def forward_prop(self, img):
        activations = [img]
        for i in range(len(self.weights)):
            z = np.dot(activations[-1], self.weights[i]) + self.biases[i]
            if self.activation_functions[i] == 'sigmoid':
                activation = self.sigmoid(z)
            elif self.activation_functions[i] == 'relu':
                activation = self.ReLU(z)
            elif self.activation_functions[i] == 'softmax':
                activation = self.softmax(z)
            activations.append(activation)
        return activations
    
    def back_prop(self, error, activations):
        gradients = [error * self.softmaxDerivative(activations[-1])]
        
        #loop through layers in reverse order starting from the second last layer
        for i in range(len(self.weights) - 1, 0, -1):
            if self.activation_functions[i - 1] == 'sigmoid':
                delta = np.dot(gradients[-1], self.weights[i].T) * self.sigmoid_derivative(activations[i])
            elif self.activation_functions[i - 1] == 'relu':
                delta = np.dot(gradients[-1], self.weights[i].T) * self.ReLU_derivative(activations[i])
            gradients.append(delta)

        #reverse gradients to match the order of the layers
        gradients.reverse()

        return gradients
    
    
    
    def fit(self, lr, epochs, trainImg, trainLabels, dropout_rate=0.0):
        oneHotLabels = self.oneHotEncode(trainLabels)
        numImages, _ = oneHotLabels.shape

        for epoch in range(epochs):
            correct = 0  #to track the number of correct predictions

            for img, label in zip(trainImg, oneHotLabels):
                img = img.reshape((1, -1))  #reshaping the image to be a row vector
                label = label.reshape((1, -1))  #reshaping the label to be a row vector

                #forwardpropagation
                activations = self.forward_prop(img)

                #calculating the error
                error = label - activations[-1]

                #backward propagation
                gradients = self.back_prop(error, activations)

                #update weights and biases
                for i in range(len(self.weights)):
                    self.weights[i] += lr * np.dot(activations[i].T, gradients[i])
                    self.biases[i] += lr * gradients[i]

                #check for correct prediction
                if np.argmax(activations[-1]) == np.argmax(label):
                    correct += 1

            accuracy = correct / numImages
            print(f"Epoch {epoch + 1}/{epochs}, Accuracy: {accuracy:.2f}")

input_size = 784  #for 28x28 pixel images
hidden_layers = 2
units_per_layer = [128, 10]  #two hidden layers with 128 and 64 neurons
activation_functions = ['relu', 'relu', 'softmax']  #activation functions for each layer including the output layer

nn = CustomNN(input_size, hidden_layers, units_per_layer, activation_functions) 
   

activationChoice = "1" #input("Choose an activation function\n 1 - Sigmoid\n 2 - ReLU")
learningRate = 0.2 #input("Enter a learning rate")
epochs = 5 #input("Enter number of epochs")

#converts image pixel values from 0 - 255 to 0 - 1 range, avoiding overflow from activation function
trainingImages = trainingImages / 255 

#training returns gradients for plotting graphs
print("training in progress...")
gradients = nn.fit(learningRate, epochs, trainingImages, trainingLabels, activationChoice)
print("training complete")




while True:
    index = int(input("Enter a number between 0 - 59999: "))
    yHat = nn.predict(trainingImages, activationChoice)
    print("prediction: ", yHat[index].argmax(), " | ", yHat[index])
    print("actual: ", trainingLabels[index], " | ", nn.oneHotEncode(trainingLabels)[index])



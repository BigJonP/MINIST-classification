#importing libraries
import logging
import random
import math
import matplotlib.pyplot as plt
import numpy as np

#importing data
from data_utils import *


#loading images and labels into variables via numpy, from the data_utils.py module 
trainingImages = np.array(load_images("train-images-idx3-ubyte.gz"))
trainingLabels = np.array(load_labels("train-labels-idx1-ubyte.gz"))
testImages = np.array(load_images("t10k-images-idx3-ubyte.gz"))
testLabels = np.array(load_labels("t10k-labels-idx1-ubyte.gz"))

'''
trainImages is 60,000 images/elements,
with each element itself being an array of length 784 (all pixels of a 28x28 image),
with each element in that array being a number 0 - 255 representing pixel brightness
For trainImages to be dot multiplied, it needs to be turned into a numpy array
'''

#Seed for reproducibility of RNG
seed = 1
np.random.seed(seed)
random.seed(seed)


class NeuralNetwork():
    def __init__(self, layer_sizes, activation_funcs):
        # Initializing the network with customizable layers and activation functions
        self.layer_sizes = layer_sizes
        self.activation_funcs = activation_funcs
        self.dropout_masks = []  # To store dropout masks for each layer

        #generate weights
        self.w1 = 2 * np.random.rand(784, 16) - 1 
        self.w2 = 2 * np.random.rand(16, 10) - 1 
        #generate biases    
        self.b1 = 2 * np.random.rand(1, 16) - 1
        self.b2 = 2 * np.random.rand(1, 10) - 1
    
    
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

    def dropout(self, x, dropout_rate, is_training=True):
        if is_training:
            # Inverted dropout mask
            mask = np.random.binomial(1, 1 - dropout_rate, size=x.shape) / (1 - dropout_rate)
            self.dropout_masks.append(mask)
            return x * mask
        else:
            return x  # No change to the input during testing

        
    def cost(self, label, output):

        #Mean Squared Error
        cost = 1 / len(output) * np.sum((output - label) ** 2, axis = 0)

        return cost

    def crossEntropyLoss(self, actual, predicted):
        #actual value/label needs to be one-hot encoded
        loss = -np.sum(actual * np.log(predicted))
        return loss    
    
    

    #create plots to compare different hyperparameters
    def plot_metrics(self, gradients, epoch_accuracies, epochs, activationChoice, learningRate):
        # just copying what's done in nn.fit
        layer1Grads, layer2Grads, w1Mags, w2Mags, a1Log, a2Log = gradients
        
        #add extra details in about the training
        activation_name = "Sigmoid" if activationChoice == "1" else "ReLU"
        #set up the plotting environment
        fig, axs = plt.subplots(3, 2, figsize=(15, 15))
        fig.suptitle(f'Training Metrics Over {epochs} Epochs\nActivation: {activation_name}, Learning Rate: {learningRate}')

        #plot accuracy per epoch, this is probably what we're most interested in
        axs[0, 0].plot(range(epochs), epoch_accuracies)
        axs[0, 0].set_title('Accuracy per Epoch')
        axs[0, 0].set_xlabel('Epoch')
        axs[0, 0].set_ylabel('Accuracy')

        #plot layer gradients, weights magnitudes, and activations? not sure if this is necessary but i guess ill add it anyway
        axs[0, 1].plot(layer1Grads, label='Layer 1 Gradients')
        axs[1, 0].plot(layer2Grads, label='Layer 2 Gradients')
        axs[1, 1].plot(w1Mags, label='Weights 1 Magnitudes')
        axs[2, 0].plot(w2Mags, label='Weights 2 Magnitudes')
        axs[2, 1].plot(a1Log, label='Activations 1 Log')
        axs[2, 1].plot(a2Log, label='Activations 2 Log')

        for ax in axs.flat:
            ax.label_outer()
            ax.legend()

        plt.show()
                
    def fit (self, lr, epochs, trainImg, trainLabels, activationFunc, dropout_rate=0.0, optimizer="sgd", momentum=0.9):
        
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
        
        
        # Initialize velocity for momentum
        v_w1 = np.zeros(self.w1.shape)
        v_w2 = np.zeros(self.w2.shape)
        v_b1 = np.zeros(self.b1.shape)
        v_b2 = np.zeros(self.b2.shape)
        
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
                a1 = self.dropout(a1, dropout_rate, is_training=True)  # Apply dropout after activation
                #hidden layer to output layer - dot multiplication of weights with output of first layer + bias
                z2 = np.dot(a1, self.w2) + self.b2
                #feeding into activation function again
                a2 = self.activation(z2, activationFunc)
                a2 = self.dropout(a2, dropout_rate, is_training=True)  # Apply dropout after activation



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

                '''
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
                '''
                # Update weights and biases
                if optimizer == "sgd":
                    self._update_parameters_sgd(lr, da1, da2, img, a1)
                elif optimizer == "sgd_momentum":
                    self._update_parameters_sgd_momentum(lr, da1, da2, img, a1, v_w1, v_w2, v_b1, v_b2, momentum)
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
            epoch_accuracies[epoch] = round((correct/trainImg.shape[0]) * 100,2)
            #reset so the accuracy is determined based on each epoch
            correct = 0
        
        #training returns gradients for plotting graphs
        print("overallCycleNum: ", overallCycleNum)
        gradients = np.array([layer1Grads, layer2Grads, w1Mags, w2Mags, a1Log, a2Log])
        self.plot_metrics(gradients, epoch_accuracies, epochs, activationChoice, learningRate)
        return gradients
        
    
    def _update_parameters_sgd(self, lr, da1, da2, img, a1):
        self.w2 -= lr * a1.T.dot(da2)
        self.b2 -= lr * da2
        self.w1 -= lr * img.dot(da1)
        self.b1 -= lr * da1

    def _update_parameters_sgd_momentum(self, lr, da1, da2, img, a1, v_w1, v_w2, v_b1, v_b2, momentum):
        # Update with momentum
        v_w2 = momentum * v_w2 + lr * a1.T.dot(da2)
        self.w2 -= v_w2

        v_b2 = momentum * v_b2 + lr * da2
        self.b2 -= v_b2

        v_w1 = momentum * v_w1 + lr * img.dot(da1)
        self.w1 -= v_w1

        v_b1 = momentum * v_b1 + lr * da1
        self.b1 -= v_b1
    
    def predict(self, x, activationFunc):
        z1 = np.dot(x, self.w1) + self.b1
        a1 = self.activation(z1, activationFunc)
        z2 = np.dot(a1, self.w2) + self.b2
        a2 = self.activation(z2, activationFunc)
        sm = self.softmax(a2)
        return sm
    
    



nn = NeuralNetwork([786,16,10],"1")
activationChoice = "1" #input("Choose an activation function\n 1 - Sigmoid\n 2 - ReLU")
learningRate = 0.2 #input("Enter a learning rate")
epochs = 2 #input("Enter number of epochs")
dropout_rate=0.5 #updating dropout rate

#converts image pixel values from 0 - 255 to 0 - 1 range, avoiding overflow from activation function
trainingImages = trainingImages / 255 

'''
#testing for sigmoid and derivative, create range of values from -10 to 10
plotVal = np.linspace(-10, 10, 200)
#run values through sigmoid and derivative
plot_sigmoid_values = [nn.activation(x, "1") for x in plotVal]
plot_sigmoid_derivative_values = [nn.activationDerivatiive(x, "1") for x in plotVal]
#relu plot values, same thing as above
plot_relu_values = [nn.activation(x, "2") for x in plotVal]
plot_relu_derivative_values =[nn.activationDerivatiive(x, "2") for x in plotVal]


plot_softmax_probabilities = nn.softmax(plotVal)

#create plots for both functions
plt.figure(figsize=(12, 5))

#sigmoid plot
plt.subplot(1, 2, 1)
plt.plot(plotVal, plot_sigmoid_values, label="Sigmoid")
plt.title("Sigmoid Function")
plt.xlabel("Input")
plt.ylabel("Output")
plt.grid(True)
plt.legend()
#backprop sigmoid plot
plt.subplot(1, 2, 2)
plt.plot(plotVal, plot_sigmoid_derivative_values, label="Sigmoid Derivative")
plt.title("Sigmoid derivative function")
plt.xlabel("Input")
plt.ylabel("Derivative")
plt.grid(True)
plt.legend()

#relu plot
plt.subplot(1, 2, 1)
plt.plot(plotVal, plot_relu_values, label="ReLU")
plt.title("ReLU function")
plt.xlabel("Input")
plt.ylabel("Output")
plt.grid(True)
plt.legend()
#relu backprop plot
plt.subplot(1, 2, 2)
plt.plot(plotVal, plot_relu_derivative_values, label="ReLU Derivative")
plt.title("ReLU derivative function")
plt.xlabel("Input")
plt.ylabel("Derivative")
plt.grid(True)
plt.legend()
#show plots
plt.tight_layout()
plt.show()

x = np.linspace(-10, 10, 100)
y = nn.softmax(x)
plt.scatter(x, y) 
plt.title('Softmax Function') 
plt.show()
'''
#training returns gradients for plotting graphs
print("training in progress...")
gradients = nn.fit(learningRate, epochs, trainingImages, trainingLabels, activationChoice, optimizer="sgd")
print("training complete")




while True:
    index = int(input("Enter a number between 0 - 59999: "))
    yHat = nn.predict(trainingImages, activationChoice)
    print("prediction: ", yHat[index].argmax(), " | ", yHat[index])
    print("actual: ", trainingLabels[index], " | ", nn.oneHotEncode(trainingLabels)[index])



'''
Trying to match with example Lab solution, trying to do all images at once instead of loop method
x:  (5, 3) | trainingImages is (60000, 784) 
y:  (1, 5) [transpose to (5, 1)] | trainingLabels is (60000,) [converts to (60000, 10)]
w1: (3, 4) | (784, 16) 
z1: (5, 3) x (3, 4) = (5, 4) | (60000, 784) x (784, 16) = (60000, 16) | (1, 784) x (784, 16) = (1, 16)
w2: (4, 1) | (16, 10) | (16, 10) x (10, 10) = (16, 10) 
z2: (5, 4) x (4, 1) = (5, 1) | (60000, 16) x (16, 10) = (60000, 10) 
'''

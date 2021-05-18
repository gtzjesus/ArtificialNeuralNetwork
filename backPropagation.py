# @author: Jesus Gutierrez
# @dateLastModified: 05/08/2020
# @description: Artificial Neural Networks on forward-propagation and backward-propagation

import numpy as np 

# X = (Hours sleeping, hours studying), Y = Test score of the student
X = np.array(([4,5],[3,7],[5,8]), dtype=float)
Y = np.array(([98],[72],[81]), dtype=float)

# Scale units
X = X/np.amax(X, axis=0) # Maximum of X array
Y = Y/100 # Maxmimum test score is 100

class NeuralNetwork(object):
    def __init__(self):
        # Parameters for creating the matrices
        self.inputSize = 2
        self.outputSize = 1
        self.hiddenSize = 3

        # Weights
        self.weightOne = np.random.randn(self.inputSize, self.hiddenSize) # 3x2 weight matrix
        self.weightTwo = np.random.randn(self.hiddenSize, self.outputSize) # 3x1 weight matrix

    def backward(self, X, Y, output):
        # Backward propagation through the network
        self.output_error =  Y - output # The error in the output
        self.output_delta = self.output_error * self.sigmoid(output, deriv=True) # Calculates delta of the output

        self.z2_error = self.output_delta.dot(self.weightTwo.T) # Calculates how much hidden later weights. Generates the transpose of the 3x1
        self.z2_delta = self.z2_error * self.sigmoid(self.z2, deriv=True) # Applies derivative of sigmoid to z2 error

        self.weightOne += X.T.dot(self.z2_delta) # Adjusts the first set (input -> hidden) weights
        self.weightTwo += self.z2.T.dot(self.output_delta) # Adjusts second set (hidden -> output) weights

    def forward(self, X):
        # Forward propagation through the network
        self.z = np.dot(X, self.weightOne) # Multiplies two matrices --> (3x2)
        self.z2 = self.sigmoid(self.z) # Activation function
        self.z3 = np.dot(self.z2, self.weightTwo) # --> (3x1)
        output = self.sigmoid(self.z3)
        return output

    def sigmoid(self, s, deriv=False):
        # Function that takes a number and turns it into a probability 
        if (deriv == True):
            return s * (1-s)
        return 1/(1 + np.exp(-s))

    def invoke(self, X, Y):
        # Function that invokes both propagations
        output = self.forward(X)
        self.backward(X, Y, output)

# Creates object Neural Network
NN = NeuralNetwork()

for i in range(1000): # Invokes the NN 1000 times
    NN.invoke(X, Y)

print("\n")
print("Input: " + str(X))
print("\n")
print("Actual Output: " + str(Y))
print("\n")
print("Loss: " + str(np.mean(np.square(Y - NN.forward(X)))))
print("\n")
print("Predicted Output: " + str(NN.forward(X)))
print("\n")


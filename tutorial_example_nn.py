from pprint import pprint
import numpy as np
import pandas as pd
import torch
import os
import io
import torch.nn as nn
import matplotlib as plt
from torch.autograd import Variable
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from compas_dataset import CompasDataset
from data_preproc_functions import load_preproc_data_compas


X = torch.tensor(([2, 9, 1], [1, 5, 1], [3, 6, 1], [20, 2, 10]),
                 dtype=torch.float)  # 3 X 2 tensor
y = torch.tensor(([92], [100], [89], [1]), dtype=torch.float)  # 3 X 1 tensor
xPredicted = torch.tensor(([0, 1, 1]), dtype=torch.float)  # 1 X 2 tensor


# scale units
X_max, _ = torch.max(X, 0)
xPredicted_max, _ = torch.max(xPredicted, 0)

X = torch.div(X, X_max)
xPredicted = torch.div(xPredicted, xPredicted_max)
y = y / 100  # max test score is 100


class Neural_Network(nn.Module):
    def __init__(self, ):
        super(Neural_Network, self).__init__()
        # parameters
        # TODO: parameters can be parameterized instead of declaring them here
        self.inputSize = 3
        self.outputSize = 1
        self.hiddenSize = 3

        # weights
        self.W1 = torch.randn(self.inputSize, self.hiddenSize)  # 3 X 2 tensor
        self.W2 = torch.randn(self.hiddenSize, self.outputSize)  # 3 X 1 tensor

    def forward(self, X):
        # 3 X 3 ".dot" does not broadcast in PyTorch
        self.z = torch.matmul(X, self.W1)
        self.z2 = self.sigmoid(self.z)  # activation function
        self.z3 = torch.matmul(self.z2, self.W2)
        o = self.sigmoid(self.z3)  # final activation function
        return o

    def sigmoid(self, s):
        return 1 / (1 + torch.exp(-s))

    def sigmoidPrime(self, s):
        # derivative of sigmoid
        return s * (1 - s)

    def backward(self, X, y, o):
        self.o_error = y - o  # error in output
        # derivative of sig to error
        self.o_delta = self.o_error * self.sigmoidPrime(o)
        self.z2_error = torch.matmul(self.o_delta, torch.t(self.W2))
        self.z2_delta = self.z2_error * self.sigmoidPrime(self.z2)
        self.W1 += torch.matmul(torch.t(X), self.z2_delta)
        self.W2 += torch.matmul(torch.t(self.z2), self.o_delta)

    def train(self, X, y):
        # forward + backward pass for training
        o = self.forward(X)
        self.backward(X, y, o)

    def saveWeights(self, model):
        # we will use the PyTorch internal storage functions
        torch.save(model, "NN")
        # you can reload model with all the weights and so forth with:
        # torch.load("NN")

    def predict(self):
        print("Predicted data based on trained weights: ")
        print("Input (scaled): \n" + str(xPredicted))
        print("Output: \n" + str(self.forward(xPredicted)))


NN = Neural_Network()
for i in range(1000):  # trains the NN 1,000 times
    # mean sum squared loss
    print("#" + str(i) + " Loss: " +
          str(torch.mean((y - NN(X))**2).detach().item()))
    NN.train(X, y)
NN.saveWeights(NN)
NN.predict()

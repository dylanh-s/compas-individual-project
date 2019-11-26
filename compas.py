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


dataset = load_preproc_data_compas(['race'])
dataset_train, dataset_test = dataset.split([0.7], shuffle=True)

frame, dic = dataset_train.convert_to_dataframe()
frame_test, dic_test = dataset_test.convert_to_dataframe()
# frame.to_csv('out.csv')

Z = frame.to_numpy()
# Nums = Z[:, 0]
X_np = Z[:, :-1]
Y_np = Z[:, -1]

# pprint(Z.shape)
# pprint(Y_np.shape)
# pprint(X_np.shape)
X = torch.tensor(X_np, dtype=torch.float)
Y = torch.tensor(Y_np, dtype=torch.float)
# pprint(X.size())
# pprint(Y.size())
xPredicted = torch.tensor(X_np[0, :], dtype=torch.float)
pprint(xPredicted)


class MyClassifier(nn.Module):
    def __init__(self):
        super(MyClassifier, self).__init__()

        self.fc1 = nn.Linear(10, 100)

        # This applies linear transformation to produce output data
        self.fc2 = nn.Linear(100, 1)

    def forward(self, x):

        # Output of the first layer
        x = self.fc1(x)

        # Activation function
        x = torch.tanh(x)

        # output
        x = self.fc2(x)
        return x

    # predicts the class (0 == low recid or 1 == high recid)
    def predict(self, x):
        # Apply softmax to output.
        pred = F.softmax(self.forward(x))

        return pred


# Initialize the model
model = MyClassifier()
# Define loss criterion
# criterion = nn.CrossEntropyLoss()
criterion = nn.BCEWithLogitsLoss()
# criterion = nn.SoftMarginLoss()
# Define the optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# Number of epochs
epochs = 4000
# List to store losses
losses = []

Y = Y.unsqueeze(1)
for i in range(epochs):
    # Precit the output for Given input
    y_pred = model.forward(X)
    # Compute Cross entropy loss
    loss = criterion(y_pred, Y)
    # Add loss to the list
    losses.append(loss.item())
    # Clear the previous gradients
    optimizer.zero_grad()
    # Compute gradients
    loss.backward()
    # Adjust weights
    optimizer.step()
    if (i % 1000 == 0):
        print(str(i/epochs * 100) + "%")

#out = model.predict(xPredicted)
# pprint(out.item())
# pprint(Y_np[70])

Z_test = frame_test.to_numpy()
# Nums = Z[:, 0]
X_test_np = Z_test[:, :-1]
Y_test_np = Z_test[:, -1]
successes = 0
inputs, _ = X_test_np.shape
for i in range(0, inputs):
    xPred = torch.tensor(X_np[i, :], dtype=torch.float)
    y_star = Y_test_np[i]
    y_hat = model.predict(xPred)
    if (y_hat.item() == y_star):
        successes += 1
print("accuracy = ")
pprint(successes/inputs)

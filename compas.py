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

from_csv = True
if (not from_csv):
    dataset = load_preproc_data_compas(['race'])
    dataset_train, dataset_test = dataset.split([0.7], shuffle=True)

    frame, dic = dataset_train.convert_to_dataframe()
    frame_test, dic_test = dataset_test.convert_to_dataframe()
    # frame.to_csv('out.csv')
    Z = frame.to_numpy()
    Z_test = frame_test.to_numpy()
else:
    Z = np.loadtxt('compas_data.csv', delimiter=',')
    Z = Z[1:, 1:]
    (rows, cols) = Z.shape
    train_rows = round(0.7*rows)
    Z_train = Z[:, :train_rows]
    Z_test = Z[:, train_rows+1:]


X_np = Z[:, :-1]
Y_np = Z[:, -1]

pprint(Z.shape)
# pprint(Y_np.shape)
# pprint(X_np.shape)
X_train_np = X_np[:, :-1]
X = torch.tensor(X_np, dtype=torch.float)
Y = torch.tensor(Y_np, dtype=torch.float)
# pprint(X.size())
# pprint(Y.size())
xPredicted = torch.tensor(X_np[0, :], dtype=torch.float)
pprint(xPredicted)
OLD_AGE_COL = 3
YOUNG_AGE_COL = 4
GENDER_COL = 0
FEMALE = 1.0
MALE = 0.0
(rows, cols) = X.shape
old_female_count = 0
old_female_reoffend = 0
young_male_count = 0
young_male_reoffend = 0


for i in range(rows):
    if (X_np[i, OLD_AGE_COL] == 1.0 and X_np[i, GENDER_COL] == FEMALE):
        old_female_count += 1
        if (Y_np[i] == 1.0):
            old_female_reoffend += 1
    if (X_np[i, YOUNG_AGE_COL] == 1.0 and X_np[i, GENDER_COL] == MALE):
        young_male_count += 1
        if (Y_np[i] == 1.0):
            young_male_reoffend += 1
pprint("")
print(old_female_count)
print(old_female_reoffend/old_female_count)
print(young_male_count)
print(young_male_reoffend/young_male_count)


class MyClassifier(nn.Module):
    def __init__(self):
        super(MyClassifier, self).__init__()

        self.fc1 = nn.Linear(10, 100)

        # This applies linear transformation to produce output data
        self.fc2 = nn.Linear(100, 100)
        self.fc3 = nn.Linear(100, 100)
        self.fc4 = nn.Linear(100, 1)

    def forward(self, x):

        # Output of the first layer
        x = self.fc1(x)

        # Activation function
        x = torch.tanh(x)
        # layer 2
        x = self.fc2(x)
        # layer 3
        x = self.fc3(x)
        # output
        x = self.fc4(x)
        return x

    # predicts the class (0 == low recid or 1 == high recid)
    def predict(self, x):
        # Apply softmax to output.
        pred = F.softmax(self.forward(x), dim=0)

        return pred


# Initialize the model
model = MyClassifier()
# Define loss criterion

# criterion = nn.BCEWithLogitsLoss()
criterion = nn.BCEWithLogitsLoss()

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
        print(loss)
        print(str(i/epochs * 100) + "%")

# out = model.predict(xPredicted)
# pprint(out.item())
# pprint(Y_np[70])

# Nums = Z[:, 0]
X_test_np = Z_test[:, :-1]
Y_test_np = Z_test[:, -1]
successes = 0
inputs, _ = X_test_np.shape

old_female_count = 0
old_female_reoffend = 0
young_male_count = 0
young_male_reoffend = 0

for i in range(0, inputs):
    xPred = torch.tensor(X_np[i, :], dtype=torch.float)
    y_star = Y_test_np[i]

    y_hat = model.predict(xPred)

    if (X_test_np[i, OLD_AGE_COL] == 1.0 and X_test_np[i, GENDER_COL] == FEMALE):
        old_female_count += 1
        if (Y_test_np[i] == 1.0):
            old_female_reoffend += 1
    if (X_test_np[i, YOUNG_AGE_COL] == 1.0 and X_test_np[i, GENDER_COL] == MALE):
        young_male_count += 1
        if (Y_test_np[i] == 1.0):
            young_male_reoffend += 1

    if (y_hat.item() == y_star):
        successes += 1
print("accuracy = "+str(successes/inputs))

print(old_female_count)
print(old_female_reoffend/old_female_count)
print(young_male_count)
print(young_male_reoffend/young_male_count)

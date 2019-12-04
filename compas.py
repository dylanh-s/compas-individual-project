from pprint import pprint
import numpy as np
import pandas as pd
import torch
import os
import io
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.autograd import Variable
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from compas_dataset import CompasDataset
from data_preproc_functions import load_preproc_data_compas

OLD_AGE_COL = 3
YOUNG_AGE_COL = 4
GENDER_COL = 0
RACE_COL = 1
FEMALE = 1
MALE = 0
WHITE = 1
NON_WHITE = 0
MODEL_RACE_BLIND_PATH = 'model_race_blind.pt'
MODEL_PATH = 'model.pt'


def plot_hist(X, Y_hat):
    white_ys = []
    non_white_ys = []
    # bins = np.linspace(0, 1, 15)
    for i in range(len(Y_hat)):
        if (int(X[i, RACE_COL]) == WHITE):
            white_ys.append(Y_hat[i])
        else:
            non_white_ys.append(Y_hat[i])
    fig = plt.figure()
    ax = fig.add_subplot(111)
    n, bins, rectangles = ax.hist(np.asarray(
        white_ys), 10, alpha=0.8, density=True, label='white')
    n, bins, rectangles = ax.hist(np.asarray(
        non_white_ys), 10, alpha=0.5, density=True, label='non_white')
    # plt.hist(np.asarray(white_ys),bins,alpha=0.5,label='white')
    # plt.hist(np.asarray(non_white_ys),bins,alpha=0.5,label='non_white')
    plt.legend(loc='upper right')
    fig.canvas.draw()
    plt.show()


def print_matrix(M):
    print("         | MALE | FEMALE    |")
    print("-------------------------------")
    print("NON-WHITE| "+str(round(M[0, 0], 3)) +
          " | "+str(round(M[0, 1], 3))+"    | ")
    print("-------------------------------")
    print("    WHITE| "+str(round(M[1, 0], 3)) +
          " | "+str(round(M[1, 1], 3))+"    | ")
    print("-------------------------------")


def getAccuracies(X, Y_hat, Y):
    count_matrix = np.zeros((2, 2))
    true_matrix = np.zeros((2, 2))
    false_matrix = np.zeros((2, 2))
    # positive -> will recid
    # true -> correct prediction
    # Different actors want different minimisations
    # The individual wants to minimise False Positives i.e. Chance of wrongly being predicted to reoffend
    # The system wants to minimise False Negatives i.e. Chance of letting someone go who would reoffend
    true_positive_matrix = np.zeros((2, 2))
    false_positive_matrix = np.zeros((2, 2))

    true_negative_matrix = np.zeros((2, 2))
    false_negative_matrix = np.zeros((2, 2))

    for i in range(len(Y_hat)):
        # prediction 1 = recid, 0 = no recid
        y_hat = round(Y_hat[i])
        group = [int(X[i, GENDER_COL]), int(X[i, RACE_COL])]
        count_matrix[group[0], group[1]] += 1
        # if correct
        if (y_hat == Y[i]):
            true_matrix[group[0], group[1]] += 1
            # and will recid
            if (y_hat == 1):
                true_positive_matrix[group[0], group[1]] += 1
            # and will not recid
            else:
                true_negative_matrix[group[0], group[1]] += 1
        # if incorrect
        else:
            # and will recid
            if (y_hat == 1):
                false_positive_matrix[group[0], group[1]] += 1
            # and will not recid
            else:
                false_negative_matrix[group[0], group[1]] += 1
    false_matrix = count_matrix-true_matrix
    print("------------------------------")
    print("accuracy matrix: ")
    print_matrix(true_matrix/count_matrix)
    print("------------------------------")
    # print("TPR matrix - : ")
    # pprint(true_positive_matrix/(true_positive_matrix + false_negative_matrix))
    # print("------------------------------")
    print("FPR matrix - predicted to reoffend, but didn't: ")
    print_matrix(false_positive_matrix /
                 (true_positive_matrix + false_negative_matrix))
    print("------------------------------")
    print("FNR matrix - predicted NOT to reoffend, but did: ")
    print_matrix(false_positive_matrix /
                 (false_positive_matrix + true_negative_matrix))
    print("------------------------------")


def getDatasetBaseRates(X, Y):

    (rows, cols) = X.shape
    # M  F
    # 0  0 NON-WHITE
    # 0  0 WHITE
    count_matrix = np.zeros((2, 2))
    reoffend_matrix = np.zeros((2, 2))
    recid_rate_matrix = np.zeros((2, 2))

    for i in range(rows):
        if (X[i, GENDER_COL] == MALE and X[i, RACE_COL] == WHITE):
            count_matrix[MALE, WHITE] += 1
            if (Y[i] >= 0.5):
                reoffend_matrix[MALE, WHITE] += 1
        if (X[i, GENDER_COL] == MALE and X[i, RACE_COL] == NON_WHITE):
            count_matrix[MALE, NON_WHITE] += 1
            if (Y[i] >= 0.5):
                reoffend_matrix[MALE, NON_WHITE] += 1

        if (X[i, GENDER_COL] == FEMALE and X[i, RACE_COL] == WHITE):
            count_matrix[FEMALE, WHITE] += 1
            if (Y[i] >= 0.5):
                reoffend_matrix[FEMALE, WHITE] += 1
        if (X[i, GENDER_COL] == FEMALE and X[i, RACE_COL] == NON_WHITE):
            count_matrix[FEMALE, NON_WHITE] += 1
            if (Y[i] >= 0.5):
                reoffend_matrix[FEMALE, NON_WHITE] += 1
    # pprint(count_matrix)
    # pprint(reoffend_matrix)

    recid_rate_matrix = reoffend_matrix/count_matrix
    # pprint(recid_rate_matrix)
    print("white male recid rate: " +
          str(recid_rate_matrix[MALE, WHITE]))
    print("non-white male recid rate: " +
          str(recid_rate_matrix[MALE, NON_WHITE]))
    print("")
    print("white female data recid rate: " +
          str(recid_rate_matrix[FEMALE, WHITE]))
    print("non-white female data recid rate: " +
          str(recid_rate_matrix[FEMALE, NON_WHITE]))
    print("------------------------------")


from_csv = True
race_blind = True
if (not from_csv):
    dataset = load_preproc_data_compas(['race'])
    dataset_train, dataset_test = dataset.split([0.7], shuffle=True)

    frame, dic = dataset_train.convert_to_dataframe()
    frame_test, dic_test = dataset_test.convert_to_dataframe()
    # frame.to_csv('out.csv')
    Z_train = frame.to_numpy()
    Z_test = frame_test.to_numpy()
else:
    Z = np.loadtxt('compas_data.csv', delimiter=',')
    Z = Z[1:, 1:]
    print("------------------------------")
    print("Dataset base rates:")
    getDatasetBaseRates(Z, Z[:, -1])
    (rows, cols) = Z.shape
    train_rows = round(0.7*rows)
    if (race_blind):
        c = 9
        Z_protected_removed = np.delete(Z, RACE_COL, axis=1)
        Z_train = Z_protected_removed[:train_rows, :]
        Z_test = Z_protected_removed[train_rows:, :]
    else:
        c = 10
        Z_train = Z[:train_rows, :]
        Z_test = Z[train_rows:, :]

    Z_test_race_included = Z[train_rows:, :]
    # print(Z_test.shape)

X_train_np = Z_train[:, :-1]
Y_train_np = Z_train[:, -1]

X = torch.tensor(X_train_np, dtype=torch.float)
Y = torch.tensor(Y_train_np, dtype=torch.float)
# pprint(X.size())
# pprint(Y.size())

(rows, cols) = X.shape


class MyClassifier(nn.Module):
    def __init__(self, c):
        super(MyClassifier, self).__init__()

        self.fc1 = nn.Linear(c, 100)

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
        pred = torch.sigmoid(self.forward(x))
        return pred


# Initialize the model
model = MyClassifier(c)
# Define loss criterion
training = False
if (training):
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
        if (race_blind):
            torch.save(model.state_dict(), MODEL_RACE_BLIND_PATH)
        else:
            torch.save(model.state_dict(), MODEL_PATH)

else:
    if (race_blind):
        model.load_state_dict(torch.load(MODEL_RACE_BLIND_PATH))
    else:
        model.load_state_dict(torch.load(MODEL_PATH))
    model.eval()

    # Nums = Z[:, 0]
X_test_np = Z_test[:, :-1]
Y_test_np = Z_test[:, -1]
successes = 0
inputs, _ = X_test_np.shape
predictions = []
for i in range(0, inputs):
    X_pred = torch.tensor(X_test_np[i, :], dtype=torch.float)
    # pprint(X_pred)
    y_star = Y_test_np[i]
    prediction = model.predict(X_pred)
    y_hat = prediction.item()
    predictions.append(y_hat)
plot_hist(Z_test_race_included, predictions)


print("------------------------------")
print("Prediction rates:")
getDatasetBaseRates(Z_test_race_included, np.asarray(predictions))

getAccuracies(Z_test_race_included, np.asarray(predictions), Y_test_np)

# print("------------------------------")
# print("Test data rates:")
# getDatasetBaseRates(Z_test_race_included, Y_test_np)

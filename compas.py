from pprint import pprint
import numpy as np
import pandas as pd
import torch
import os
import io
import torch.nn as nn
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from torch.autograd import Variable
import torch.optim as optim
import torch.nn.functional as F
import enum
from torch.utils.data import Dataset, DataLoader
from compas_dataset import CompasDataset
from data_preproc_functions import load_preproc_data_compas
from matplotlib import cm

OLD_AGE_COL = 3
YOUNG_AGE_COL = 4
GENDER_COL = 0
RACE_COL = 1
FEMALE = 1
MALE = 0
WHITE = 1
NON_WHITE = 0


class Metrics(enum.Enum):

    dTPR = 0
    dFPR = 1
    SP = 2


MODEL_RACE_BLIND_PATH = 'model_race_blind.pt'
MODEL_PATH = 'model.pt'
# thresholds as a function of protected characteristics
# cost is number of people classified differently based on change to the classifier
# calibration
# equity
# causal fairness paper (maybe ucl), arguing its in mportant to include protected characteristics in classifier
# can compensate for other problems

# backprop cost functions,


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
    n, bins, rectangles = ax.hist(
        np.asarray(white_ys),
        10, alpha=0.8, density=True, label='white')
    n, bins, rectangles = ax.hist(
        np.asarray(non_white_ys),
        10, alpha=0.5, density=True, label='non_white')
    # plt.hist(np.asarray(white_ys),bins,alpha=0.5,label='white')
    # plt.hist(np.asarray(non_white_ys),bins,alpha=0.5,label='non_white')
    plt.legend(loc='upper right')
    fig.canvas.draw()
    plt.show()


def fairness_2D_graphs(metric, thresholds, zs):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    xs, ys = [], []
    for ts in thresholds:
        xs.append(ts[0])
        ys.append(ts[1])
    axes = plt.gca()
    axes.set_xlim([0, 1])
    axes.set_ylim([0, 1])
    p = ax.scatter(np.asarray(xs), np.asarray(ys), c=np.log(zs), cmap=cm.jet, s=5, marker='s')
    fig.colorbar(p)
    plt.show()


def fairness_3D_graphs(metric, thresholds, zs):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    xs, ys = [], []
    for ts in thresholds:
        xs.append(ts[0])
        ys.append(ts[1])
    print("xs"+str(len(xs)))
    print("ys"+str(len(ys)))
    print("zs"+str(len(zs)))
    # xs, ys = np.meshgrid(xs, ys)

    p = ax.scatter(np.asarray(xs), np.asarray(ys), np.log(zs), c=zs, zdir='z')
    # p = ax.bar3d(
    #	 np.asarray(xs),
    #	 np.asarray(ys),
    #	 np.ones_like(zs),
    #	 0.01*np.ones_like(xs),
    #	 0.01*np.ones_like(ys),np.log(zs),shade=True)
    # fig.canvas.draw()
    plt.show()


def print_confusion_matrix(M):
    print("          | PRED: NO | PRED: YES |")
    print("-------------------------------")
    print("ACTL:   NO| "+str(round(M[0, 0], 3)) +
          "	| "+str(round(M[0, 1], 3))+"	 |")
    print("-------------------------------")
    print("ACTL:  YES| "+str(round(M[1, 0], 3)) +
          "	| "+str(round(M[1, 1], 3))+"	 |")
    print("-------------------------------")


def print_matrix(M):
    print("		 | MALE | FEMALE	|")
    print("-------------------------------")
    print("NON-WHITE| "+str(round(M[0, 0], 3)) +
          " | "+str(round(M[0, 1], 3))+"	| ")
    print("-------------------------------")
    print("	WHITE| "+str(round(M[1, 0], 3)) +
          " | "+str(round(M[1, 1], 3))+"	| ")
    print("-------------------------------")


def get_cost_of_switch(Y_hat_0, Y_hat_1, neg_to_pos_cost=1, pos_to_neg_cost=1):
    cost = 0

    for i in range(len(Y_hat_0)):
        if (Y_hat_0[i] == 1 and Y_hat_1[i] == 0):
            cost += pos_to_neg_cost
            # print("cost_pos")
        elif (Y_hat_0[i] == 0 and Y_hat_1[i] == 1):
            cost += neg_to_pos_cost
            # print("cost_neg")
    return cost


def probability_to_outcome(X, Y_hat, thresholds):
    outcomes = np.zeros_like(Y_hat)
    for i in range(len(Y_hat)):
        # prediction 1 = recid, 0 = no recid
        race = int(X[i, RACE_COL])
        if (Y_hat[i] > thresholds[race]):
            y_hat = 1
        else:
            y_hat = 0
        outcomes[i] = y_hat
    return outcomes


def get_best_pairs(X, Y_hat, target, precision=100):
    li = []
    paired_ys = []
    recids = 0
    for big_i in range(precision):
        i = big_i/precision
        best = 1000000
        best_j = -1

        for big_j in range(precision):
            j = big_j/precision
            thresholds = [i, j]
            recids = 0

            for x in range(len(Y_hat)):
                # prediction 1 = recid, 0 = no recid
                race = int(X[x, RACE_COL])
                if (Y_hat[x] > thresholds[race]):
                    recids += 1
            # if number predicted to recid is approx equal to target
            if(abs(recids - target) < best):
                # print(str(i)+","+str(j)+" gives "+str(recids)+" positives")
                best = abs(recids-target)
                best_j = j
        li.append([i, best_j])
        # print(str(i)+" , "+str(best_j))
        paired_ys.append(best_j)
    # print(len(li))

    # go through again filling the ys with no corresponding xs
    for big_y in range(precision):
        y = big_y/100
        if (y not in paired_ys):
            count = 0
            best = 1000000
            best_j = -1
            for big_j in range(precision):
                j = big_j/precision
                thresholds = [j, y]
                recids = 0

                for x in range(len(Y_hat)):
                    # prediction 1 = recid, 0 = no recid
                    race = int(X[x, RACE_COL])
                    if (Y_hat[x] > thresholds[race]):
                        recids += 1
                # if number predicted to recid is approx equal to target
                if(abs(recids - target) < best):
                    # print(str(i)+","+str(x)+" gives "+str(recids)+" positives")
                    best = abs(recids-target)
                    best_j = j
            # print(str(best_j)+" , "+str(y))
            li.append([best_j, y])

    return li


def get_threshold_pairs(X, Y_hat, target, max_relaxation_of_target=0.1, precision=100):
    li = []
    recids = 0
    # epsilon gives percentage distance we are willing to deviate from our target number of arrests in order to consider a threshold pair
    epsilon = max_relaxation_of_target
    for big_i in range(precision):
        i = big_i/precision
        for big_j in range(precision):
            j = big_j/precision
            thresholds = [i, j]
            recids = 0
            for x in range(len(Y_hat)):
                # prediction 1 = recid, 0 = no recid
                race = int(X[x, RACE_COL])
                if (Y_hat[x] > thresholds[race]):
                    recids += 1
            # if number predicted to recid is approx equal to target
            if(abs(recids - target) < epsilon*target):
                # print(str(i)+","+str(j)+" gives "+str(recids)+" positives")
                li.append(thresholds)
    # print(len(li))
    return li


def get_statistical_parity(X, Y_hat, Y, thresholds):
    confusion_matrices = [np.zeros((2, 2)), np.zeros((2, 2))]

    pos_outcomes = np.zeros(2)

    counts = np.zeros(2)
    # positive -> predicted to recid
    # true -> correct prediction
    # Different actors want different minimisations

    for i in range(len(Y_hat)):
        # prediction 1 = recid, 0 = no recid
        race = int(X[i, RACE_COL])
        if (Y_hat[i] > thresholds[race]):
            y_hat = 1
        else:
            y_hat = 0
        pos_outcomes[race] += y_hat

        counts[race] += 1
    P_favoured_outcome_non_white = ((counts[NON_WHITE]-pos_outcomes[NON_WHITE])/counts[NON_WHITE])
    P_favoured_outcome_white = ((counts[WHITE]-pos_outcomes[WHITE])/counts[WHITE])
    epsilon = (P_favoured_outcome_non_white/P_favoured_outcome_white)
    # print("p(Y_hat|non_white) "+str(P_favoured_outcome_non_white))
    # print("p(Y_hat|white) "+str(P_favoured_outcome_white))
    # print(str(abs(epsilon*100))+" percent fair")

    return abs(1-epsilon)


def get_confusion_matrices(X, Y_hat, Y, thresholds):
    confusion_matrices = [
        np.zeros((2, 2)),
        np.zeros((2, 2))]

    counts = np.zeros(2)
    trues = np.zeros(2)
    falses = np.zeros(2)
    true_positives = np.zeros(2)
    true_negatives = np.zeros(2)
    false_positives = np.zeros(2)
    false_negatives = np.zeros(2)
    # positive -> will recid
    # true -> correct prediction
    # Different actors want different minimisations
    # The individual wants to minimise False Positives i.e. Chance of wrongly being predicted to reoffend
    # The system wants to minimise False Negatives i.e. Chance of letting someone go who would reoffend
    for i in range(len(Y_hat)):
        # prediction 1 = recid, 0 = no recid
        race = int(X[i, RACE_COL])
        if (Y_hat[i] > thresholds[race]):
            y_hat = 1
        else:
            y_hat = 0
        counts[race] += 1
        # if correct
        if (y_hat == Y[i]):
            trues[race] += 1
            # and will recid
            if (y_hat == 1):
                true_positives[race] += 1
            # and will not recid
            else:
                true_negatives[race] += 1
        # if incorrect
        else:
            # and will recid
            if (y_hat == 1):
                false_positives[race] += 1
            # and will not recid
            else:
                false_negatives[race] += 1
    falses = counts-trues
    for i in range(2):
        race = i
        # print("------------------------------")
        # print("RACE = " + str(i))
        # print("COUNT = " + str(counts[race]))
        CM = confusion_matrices[race]

        # true negatives
        CM[0, 0] = true_negatives[race] / (
            true_negatives[race]+false_positives[race])

        # false positives
        CM[0, 1] = false_positives[race] / (
            true_negatives[race]+false_positives[race])

        # false negatives
        CM[1, 0] = false_negatives[race] / (
            true_positives[race]+false_positives[race])

        # true positives
        CM[1, 1] = true_positives[race] / (
            true_positives[race]+false_positives[race])
        confusion_matrices[race] = CM
        # print_confusion_matrix(CM)
    return confusion_matrices


def get_equalised_odds(X, Y_hat, Y, thresholds):
    confusion_matrices = get_confusion_matrices(X, Y_hat, Y, thresholds)
    CM_DIFF = confusion_matrices[NON_WHITE] - confusion_matrices[WHITE]
    # for dTPR, this is actually false negatives because we want the odds of a positive outcome for the users across groups to be equal.

    # dTPR = CM_DIFF[1, 1]
    # dFPR = CM_DIFF[0, 1]

    dTPR = CM_DIFF[0, 0]
    dFPR = CM_DIFF[1, 0]

    # print("dTPR = " + str(dTPR))
    # print("dFPR = " + str(dFPR))
    return abs(dTPR), abs(dFPR)


def getAccuracies(X, Y_hat, Y, thresholds):
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
        race = int(X[i, RACE_COL])
        count_matrix[group[0], group[1]] += 1
        # if correct
        if (y_hat == Y[i]):
            true_matrix[group[0], group[1]] += 1
            # and will recid
            if (y_hat > thresholds[race]):
                true_positive_matrix[group[0], group[1]] += 1
            # and will not recid
            else:
                true_negative_matrix[group[0], group[1]] += 1
        # if incorrect
        else:
            # and will recid
            if (y_hat > thresholds[race]):
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
    print_matrix(
        false_positive_matrix /
        (true_positive_matrix + false_negative_matrix))
    print("------------------------------")
    print("FNR matrix - predicted NOT to reoffend, but did: ")
    print_matrix(
        false_positive_matrix /
        (false_positive_matrix + true_negative_matrix))
    print("------------------------------")


def get_dataset_base_rates_intersectional(X, Y):
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


def get_weighted_average_base(X, Y):

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

    recid_rate_matrix = reoffend_matrix/count_matrix
    total_count = np.sum(count_matrix)
    weighted_base_rate = (count_matrix[MALE, WHITE]*recid_rate_matrix[MALE, WHITE]
                          + count_matrix[FEMALE, WHITE]*recid_rate_matrix[FEMALE, WHITE]
                          + count_matrix[MALE, NON_WHITE]*recid_rate_matrix[MALE, NON_WHITE]
                          + count_matrix[FEMALE, NON_WHITE]*recid_rate_matrix[FEMALE, NON_WHITE]) / total_count
    print("Weighted Average Base Rate: "+str(weighted_base_rate))
    return weighted_base_rate


from_csv = True
race_blind = True
# if (not from_csv):
#	 dataset = load_preproc_data_compas(['race'])
#	 dataset_train, dataset_test = dataset.split([0.7], shuffle=True)

#	 frame, dic = dataset_train.convert_to_dataframe()
#	 frame_test, dic_test = dataset_test.convert_to_dataframe()
#	 # frame.to_csv('out.csv')
#	 Z_train = frame.to_numpy()
#	 Z_test = frame_test.to_numpy()
# else:
Z = np.loadtxt('compas_data.csv', delimiter=',')
Z = Z[1:, 1:]
print("------------------------------")
print("Dataset base rates:")
get_dataset_base_rates_intersectional(Z, Z[:, -1])
BASE = get_weighted_average_base(Z, Z[:, -1])
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

        self.fc1 = nn.Linear(c, 20)

        # This applies linear transformation to produce output data
        self.fc2 = nn.Linear(20, 20)
        self.fc3 = nn.Linear(20, 20)
        self.fc4 = nn.Linear(20, 20)
        self.fc5 = nn.Linear(20, 1)

    def forward(self, x):

        # Output of the first layer
        x = self.fc1(x)

        # Activation function
        x = torch.sigmoid(x)
        # layer 2
        x = self.fc2(x)
        # layer 3
        x = self.fc3(x)
        # layer 4
        x = self.fc4(x)
        # output
        x = self.fc5(x)
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
    optimizer = torch.optim.Adam(
        model.parameters(), lr=0.01)

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
            torch.save(
                model.state_dict(),
                MODEL_RACE_BLIND_PATH)
        else:
            torch.save(model.state_dict(), MODEL_PATH)
else:
    if (race_blind):
        model.load_state_dict(
            torch.load(MODEL_RACE_BLIND_PATH))
    else:
        model.load_state_dict(torch.load(MODEL_PATH))
    model.eval()

X_test_np = Z_test[:, :-1]
Y_test_np = Z_test[:, -1]
successes = 0
inputs, _ = X_test_np.shape
predictions = []
for i in range(0, inputs):
    X_pred = torch.tensor(X_test_np[i, :], dtype=torch.float)
    # pprint(X_pred)
    # y_star = Y_test_np[i]
    prediction = model.predict(X_pred)
    y_hat = prediction.item()
    predictions.append(y_hat)
print("done")

# plot_hist(Z_test_race_included, predictions)
thresholds = get_threshold_pairs(Z_test_race_included, np.asarray(predictions), BASE * len(predictions))
# thresholds = get_best_pairs(Z_test_race_included, np.asarray(predictions), BASE * len(predictions))
# thresholds = get_threshold_pairs(Z_test_race_included, np.asarray(predictions), BASE * len(predictions), 1.0)

min = []
min_ts = []
zs = []
currents = []
for met in Metrics:
    min.append(10000)
    min_ts.append([0, 0])
    zs.append([])

count = 0
for ts in thresholds:
    currents = [0, 0, 0]
    currents[Metrics.dTPR.value], currents[Metrics.dFPR.value] = get_equalised_odds(
        Z_test_race_included, np.asarray(predictions), Y_test_np, ts)
    currents[Metrics.SP.value] = get_statistical_parity(Z_test_race_included, np.asarray(predictions), Y_test_np, ts)

    for met in Metrics:
        zs[met.value].append(currents[met.value])

        if (min[met.value] > currents[met.value]):
            min[met.value] = currents[met.value]
            min_ts[met.value] = ts

    count += 1

print("count= "+str(count))
zs_all = []
for met in Metrics:
    zs_all.append(zs[met.value])

print("")
switch_costs = np.zeros((len(Metrics), len(Metrics)))
for met_i in Metrics:

    fairness_2D_graphs(met_i, thresholds, np.asarray(zs_all[met_i.value]))

    print("min "+met_i.name+" = "+str(min[met_i.value])+" at "+str(min_ts[met_i.value]))
    cms = get_confusion_matrices(Z_test_race_included, np.asarray(predictions), Y_test_np, min_ts[met_i.value])
    print("non_white")
    print_confusion_matrix(cms[0])
    print("white")
    print_confusion_matrix(cms[1])
    print("")
    print("")
    met_val = met_i.value
    for met_j in Metrics:
        # if (met_j.value > met_i.value):
        outcomes_i = probability_to_outcome(Z_test_race_included, np.asarray(predictions), min_ts[met_i.value])
        outcomes_j = probability_to_outcome(Z_test_race_included, np.asarray(predictions), min_ts[met_j.value])
        switch_costs[met_i.value, met_j.value] = get_cost_of_switch(outcomes_i, outcomes_j)
pprint(switch_costs)
costs = np.sum(switch_costs, 1)
print(costs)
optimal = np.argmin(costs)
print("best metric for cost minimisation is " + Metrics(optimal).name)

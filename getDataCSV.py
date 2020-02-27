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

dataset = load_preproc_data_compas(['race'])
dataset_train, dataset_test = dataset.split([0.7], shuffle=True)

frame, dic = dataset_train.convert_to_dataframe()
frame_test, dic_test = dataset_test.convert_to_dataframe()
frame.to_csv('out2.csv')
Z_train = frame.to_numpy()
Z_test = frame_test.to_numpy()

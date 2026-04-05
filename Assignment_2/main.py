import numpy as np
import pickle
import copy
import matplotlib.pyplot as plt

from torch_gradient_computations import ComputeGradsWithTorch

from ann import *
from data_handling import *
from plotting import *
from ann_sigmoid import *


# ------- Load data ------------

cifar_dir = '../Datasets/cifar-10-batches-py/'


# For 1 batch
trainX, trainY, trainy = LoadBatch(cifar_dir +  'data_batch_1')
validX, validY, validy = LoadBatch(cifar_dir +  'data_batch_2')

"""
# For all batches
X, Y, y = LoadAll(cifar_dir)
trainX = X[:, 1000:]
trainY = Y[:, 1000:]
trainy = y[1000:]
validX = X[:, :1000]
validY = Y[:, :1000]
validy = y[:1000]
"""

testX, testY, testy = LoadBatch(cifar_dir +  'test_batch')

d = trainX.shape[0]
n = trainX.shape[1]
K = trainY.shape[0]


# ------- Normalize data -------------

mean_X = np.mean(trainX, axis=1).reshape(d, 1)
std_X = np.std(trainX, axis=1).reshape(d, 1)

trainX = NormalizeData(trainX, mean_X, std_X)
validX = NormalizeData(validX, mean_X, std_X)
testX = NormalizeData(testX, mean_X, std_X)


# -------- Initialize ----------------

m = 50

net = InitializeNet(d, m, K)

print(trainX.shape, trainY.shape, trainy.shape)
print(net['W'][0].shape, net['b'][0].shape)
print(net['W'][1].shape, net['b'][1].shape)


# ---------- Forward pass --------

X_small = trainX[:, 0:5]
P, fp_data = ApplyNetwork(X_small, net)

print("P shape:", P.shape)                 # (10, 5)
print("s1 shape:", fp_data['S1'].shape)    # (50, 5)
print("h shape:", fp_data['H'].shape)      # (50, 5)
print("s shape:", fp_data['S'].shape)      # (10, 5)
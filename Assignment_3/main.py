import numpy as np
import pickle
import copy
import matplotlib.pyplot as plt
from math import floor

from torch_gradient_computations import ComputeGradsWithTorch

from ann import *
from data_handling import *
from convolution import *


# ------- Load data ----------------------------------------------------

cifar_dir = '../Datasets/cifar-10-batches-py/'


# --------- For 1 batch

trainX, trainY, trainy = LoadBatch(cifar_dir +  'data_batch_1')
validX, validY, validy = LoadBatch(cifar_dir +  'data_batch_2')


# --------- For all batches
"""
n_val = 1000

X, Y, y = LoadAll(cifar_dir)
trainX = X[:, n_val:]
trainY = Y[:, n_val:]
trainy = y[n_val:]
validX = X[:, :n_val]
validY = Y[:, :n_val]
validy = y[:n_val]

"""

# ---------

testX, testY, testy = LoadBatch(cifar_dir +  'test_batch')

d = trainX.shape[0]
n = trainX.shape[1]
K = trainY.shape[0]


# ------- Normalize data ------------------------------------------------

mean_X = np.mean(trainX, axis=1).reshape(d, 1)
std_X = np.std(trainX, axis=1).reshape(d, 1)

trainX = NormalizeData(trainX, mean_X, std_X)
validX = NormalizeData(validX, mean_X, std_X)
testX = NormalizeData(testX, mean_X, std_X)


data = {'trainX': trainX, 'trainY': trainY, 'trainy': trainy,
        'validX': validX, 'validY': validY, 'validy': validy}


# ------------ Build/store MX -------------------

f = 4

trainMX = MXFromX(trainX, f)
validMX = MXFromX(validX, f)
testMX = MXFromX(testX, f)

SaveMX(trainMX, 'trainMX_' + str(n) + '.npy')
SaveMX(validMX, 'validMX_' + str(n) + '.npy')
SaveMX(testMX, 'testMX_' + str(n) + '.npy')


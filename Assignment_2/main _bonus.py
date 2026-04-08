import numpy as np
import pickle
import copy
import matplotlib.pyplot as plt
from math import floor

from torch_gradient_computations import ComputeGradsWithTorch

from ann import *
from data_handling import *
from plotting import *
from ann_sigmoid import *


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


# -------------------------- Final training ----------------------
#"""

lam_best = 0.00199526

m = 100
n_batch = 100
n_s = 2 * floor(n / n_batch)
eta_min = 1e-5
eta_max = 1e-1
n_cycles = 1

GDparams = {
    'n_batch': n_batch,
    'eta_min': eta_min,
    'eta_max': eta_max,
    'n_s': n_s,
    'n_cycles': n_cycles
}


net = InitializeNet(d, m, K)

trained_net, history = MiniBatchGD(
    trainX, trainY, trainy,
    validX, validY, validy,
    GDparams, net, lam_best, seed=42
)

P_train = ApplyNetwork(trainX, trained_net)['P']
train_acc = ComputeAccuracy(P_train, trainy)

P_val = ApplyNetwork(validX, trained_net)['P']
val_acc = ComputeAccuracy(P_val, validy)

P_test = ApplyNetwork(testX, trained_net)['P']
test_acc = ComputeAccuracy(P_test, testy)

print(f"Best lambda: {lam_best:.8f}")
print(f"Training accuracy:   {100 * train_acc:.2f}%")
print(f"Validation accuracy: {100 * val_acc:.2f}%")
print(f"Test accuracy:       {100 * test_acc:.2f}%")


#"""
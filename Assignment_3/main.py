import numpy as np
import pickle
import copy
import matplotlib.pyplot as plt
from math import floor

from torch_gradient_computations import ComputeGradsWithTorch

from plotting import PlotPerformance
from ann import ComputeAccuracy
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


# ------------ Network shape --------------------

f = 4
nf = 20 
nh = 50


# ------------ Build/store MX -------------------

trainMX = MXFromX(trainX, f)
validMX = MXFromX(validX, f)
testMX = MXFromX(testX, f)

SaveMX(trainMX, 'trainMX_' + str(n) + '.npy')
SaveMX(validMX, 'validMX_' + str(n) + '.npy')
SaveMX(testMX, 'testMX_' + str(n) + '.npy')

data = {'trainMX': trainMX, 'trainY': trainY, 'trainy': trainy,
        'validMX': validMX, 'validY': validY, 'validy': validy}


# -------------- Initialize Net -------------------

init_net = InitializeCNN(f, nf, nh, K, seed=42)


# ------------------ Train  ------------------------

lam = 0.001

eta_min = 1e-5
eta_max = 1e-1
n_s = 800
n_cycles = 2

GDparams = {'n_batch': 100, 'eta_min': eta_min, 'eta_max': eta_max,
             'n_s': n_s, 'n_cycles': n_cycles}



trained_net, history = MiniBatchGDConv(data, GDparams, init_net, lam, seed=42, n_rec=10)


# ---------------- Evaluate -----------------------

P_train = ForwardConv(trainX, trained_net)['P']
train_acc = ComputeAccuracy(P_train, trainy)

P_val = ForwardConv(validX, trained_net)['P']
val_acc = ComputeAccuracy(P_val, validy)

P_test = ForwardConv(testX, trained_net)['P']
test_acc = ComputeAccuracy(P_test, testy)

print(f"Training accuracy:   {100 * train_acc:.2f}%")
print(f"Validation accuracy: {100 * val_acc:.2f}%")
print(f"Test accuracy:       {100 * test_acc:.2f}%")


# ............ Plot performance

PlotPerformance(history['step'], history['train_cost'], history['val_cost'],
                title='Cost plot', ylabel='cost', file_name='ex3_cost')

PlotPerformance(history['step'], history['train_loss'], history['val_loss'],
                title='Loss plot', ylabel='loss', file_name='ex3_loss')

PlotPerformance(history['step'], history['train_acc'], history['val_acc'],
                title='Accuracy plot', ylabel='accuracy', file_name='ex3_acc')
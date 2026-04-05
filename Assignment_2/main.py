import numpy as np
import pickle
import copy
import matplotlib.pyplot as plt

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

X, Y, y = LoadAll(cifar_dir)
trainX = X[:, 1000:]
trainY = Y[:, 1000:]
trainy = y[1000:]
validX = X[:, :1000]
validY = Y[:, :1000]
validy = y[:1000]

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


# -------- Initialization test -----------------------------------------
"""

m = 50

net = InitializeNet(d, m, K)

print(trainX.shape, trainY.shape, trainy.shape)
print(net['W'][0].shape, net['b'][0].shape)
print(net['W'][1].shape, net['b'][1].shape)

"""


# ---------- Forward pass test -----------------------------------------
"""

X_small = trainX[:, 0:5]
fp_data = ApplyNetwork(X_small, net)

print("P shape:", fp_data['P'].shape)                 # (10, 5)
print("s1 shape:", fp_data['S1'].shape)    # (50, 5)
print("h shape:", fp_data['H'].shape)      # (50, 5)
print("s shape:", fp_data['S'].shape)      # (10, 5)

"""


# --------- Backward pass test ----------------------------------------


# ------------ Test with PyTorch

"""

d_small = 5
n_small = 3
m = 6
lam = 0
small_net = InitializeNet(d_small, m, K)


X_small = trainX[0:d_small, 0:n_small]
Y_small = trainY[:, 0:n_small]
fp_data = ApplyNetwork(X_small, small_net)
my_grads = BackwardPass(X_small, Y_small, fp_data, small_net, lam)

torch_grads = ComputeGradsWithTorch(X_small, trainy[0:n_small], small_net)

# Print comparison error between analytic and PyTorch gradients
for layer in range(len(my_grads['W'])):
    abs_diff_W = np.max(np.abs(my_grads['W'][layer] - torch_grads['W'][layer]))
    abs_diff_b = np.max(np.abs(my_grads['b'][layer] - torch_grads['b'][layer]))

    rel_diff_W = abs_diff_W / np.maximum(
        1e-12,
        np.max(np.abs(my_grads['W'][layer]) + np.abs(torch_grads['W'][layer]))
    )
    rel_diff_b = abs_diff_b / np.maximum(
        1e-12,
        np.max(np.abs(my_grads['b'][layer]) + np.abs(torch_grads['b'][layer]))
    )

    print(f"Layer {layer+1}")
    print(f"  W max abs diff: {abs_diff_W:.10e}")
    print(f"  W max rel diff: {rel_diff_W:.10e}")
    print(f"  b max abs diff: {abs_diff_b:.10e}")
    print(f"  b max rel diff: {rel_diff_b:.10e}")

"""


# ---------- Test overfitting 
"""

m = 50
lam = 0
small_net = InitializeNet(d, m, K)

X_tiny = trainX[:, :100]
Y_tiny = trainY[:, :100]
y_tiny = trainy[:100]

X_tiny_val = validX[:, :100]
Y_tiny_val = validY[:, :100]
y_tiny_val = validy[:100]


GDparams = {'n_batch': 10, 'eta': 0.01, 'n_epochs': 200}

trained_net, history = MiniBatchGD(X_tiny, Y_tiny, y_tiny,
                                X_tiny_val, Y_tiny_val, y_tiny_val,
                                GDparams, small_net, lam, seed=42)

# training accuracy
P_train = ApplyNetwork(X_tiny, trained_net)['P']
train_acc = ComputeAccuracy(P_train, y_tiny)
print(f"Training accuracy: {100 * train_acc:.2f}%")

# validation accuracy
P_val = ApplyNetwork(X_tiny_val, trained_net)['P']
val_acc = ComputeAccuracy(P_val, y_tiny_val)
print(f"Validation accuracy: {100 * val_acc:.2f}%")

"""


import numpy as np

from data_handling import *
from RNN import *
from torch_gradient_computations_column_wise import *

book_data, unique_chars, char_to_ind, ind_to_char = LoadBookData()

K = len(unique_chars)
m = 10
eta = 0.001
seq_length = 25

RNN = InitializeRNN(K, m)

X_chars = book_data[0:seq_length]
Y_chars = book_data[1:seq_length+1]

X = StrToOneHot(X_chars, char_to_ind, K)
Y = StrToOneHot(Y_chars, char_to_ind, K)

h0 = np.zeros((m,1))

loss, fp = ForwardPass(X, Y, RNN, h0)
grads = BackwardPass(X, Y, RNN, fp)

# Y = one hot labels ==> y = integer labels
y = np.argmax(Y, axis=0)

torch_grads = ComputeGradsWithTorch(X, y, h0, RNN)

for kk in grads.keys():
    diff = np.max(np.abs(grads[kk] - torch_grads[kk]))
    print(kk)
    print("max abs diff:", diff)


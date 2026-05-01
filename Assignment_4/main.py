import numpy as np

from data_handling import *
from RNN import *

# ---------- Exercise 0.1: Read in the data ---------------------------------

book_data, unique_chars, char_to_ind, ind_to_char = LoadBookData()


# ---------- Exercise 0.2:  Hyper-parameters & Initilization ----------------

K = len(unique_chars)
m = 100
eta = 0.001
seq_length = 25

RNN = InitializeRNN(K, m)

# --------------- Exercise 0.3:  Synthesize text ------------------------
"""
rng = np.random.default_rng(42)

n = 200

h0 = np.zeros((m, 1))
x0 = np.zeros((K, 1))
x0[char_to_ind['.']] = 1

Y = Synthesize(RNN, h0, x0, n, rng)

print("Generated text: ")
print(OneHotToStr(Y, ind_to_char))
print("------------------")
"""

# --------------- Exercise 0.4: Forward & backward pass ------------------------

"""
X_chars = book_data[0:seq_length]
Y_chars = book_data[1:seq_length+1]

X = StrToOneHot(X_chars, char_to_ind, K)
Y = StrToOneHot(Y_chars, char_to_ind, K)

h0 = np.zeros((m,1))

loss, fp = ForwardPass(X, Y, RNN, h0)
grads = BackwardPass(X, Y, RNN, fp)

print("loss: ", loss)
"""

# --------------- Exercise 0.4: Forward & backward pass ------------------------

n_updates = 100000

RNN, smooth_losses = TrainRNN(
    book_data,
    char_to_ind,
    ind_to_char,
    RNN,
    eta,
    seq_length,
    n_updates)



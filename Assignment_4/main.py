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

rng = np.random.default_rng(42)

n = 200

h0 = np.zeros((m, 1))
x0 = np.zeros((K, 1))
x0[char_to_ind['.']] = 1

Y = Synthesize(RNN, h0, x0, n, rng)

print("Generated text: ")
print(YtoString(Y, ind_to_char))
print("------------------")

# --------------- Exercise 0.4: Forward & backward pass ------------------------


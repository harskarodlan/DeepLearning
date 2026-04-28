import numpy as np

from data_handling import *
from RNN import *

# ---------- Exercise 0.1: Read in the data -------------

book_data, unique_chars, char_to_ind, ind_to_char = LoadBookData()


# ---------- Exercise 0.2:  Hyper-parameters & Initilization ------------------------

K = len(unique_chars)
m = 100
eta = 0.001
seq_length = 25

RNN = InitializeRNN(K, m)

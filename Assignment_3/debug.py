import numpy as np

from data_handling import *
from convolution import *


# ------------ Exercise 1 -------------------------

# ---------- Slow convolution

X_ims, Fs, conv_out_true = LoadDebugData()
conv_out = SlowConv(X_ims, Fs)

print("max abs diff: ", np.max(np.abs(conv_out - conv_out_true)))


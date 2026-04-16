import numpy as np

from data_handling import *
from convolution import *


# ------------ Exercise 1 -------------------------

# ---------- Slow convolution

X_ims, debug_data = LoadDebugData()
Fs = debug_data['Fs']
conv_out_true = debug_data['conv_outputs']
conv_out_slow = SlowConv(X_ims, Fs)

print("slow conv_out diff: ", np.max(np.abs(conv_out_slow - conv_out_true)))


# --------- Fast convolution

f = Fs.shape[0]
MX = BuildMX(X_ims, f)
Fs_flat = FlattenFilters(Fs)

conv_out_mat = Conv(MX, Fs_flat)

n_p, _, n = MX.shape
nf = Fs_flat.shape[1]

conv_out_slow_flat =  conv_out_slow.reshape((n_p, nf, n), order='C')

print("MX diff: ", np.max(np.abs(MX-debug_data['MX'])))
print("Fs_flat diff: ", np.max(np.abs(Fs_flat - debug_data['Fs_flat'])))
print("conv_outputs_mat diff: ", np.max(np.abs(conv_out_mat-debug_data['conv_outputs_mat'])))
print("conv slow vs fast diff: ", np.max(np.abs(conv_out_mat-conv_out_slow_flat)))


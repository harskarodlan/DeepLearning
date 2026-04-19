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



# --------------- Exercise 2 -----------------------------------

# ----------- Forward Pass

debug_network = {'Fs_flat': Fs_flat, 
                 'W': [debug_data['W1'], debug_data['W2']],
                 'b': [debug_data['b1'], debug_data['b2']]}

fp_data = ForwardConv(MX, debug_network)

print("conv_flat diff: ", 
      np.max(np.abs(fp_data['conv_flat'] - debug_data['conv_flat'])))
print("X1 diff: ",
      np.max(np.abs(fp_data['X1'] - debug_data['X1'])))
print("P diff: ",
      np.max(np.abs(fp_data['P'] - debug_data['P'])))


# ----------- Backward Pass

Y = debug_data['Y']

grads = BackwardConv(MX, Y, fp_data, debug_network, lam=0)

print("grads_Fs_flat diff: ",
      np.max(np.abs(grads['Fs_flat'] - debug_data['grad_Fs_flat'])))



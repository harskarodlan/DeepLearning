import numpy as np

from data_handling import *
from convolution import *
from torch_gradient_computations import *

# ------------ Exercise 1 -------------------------
"""

# ---------- Slow convolution

X_ims, debug_data = LoadDebugData()
Fs = debug_data['Fs'].astype(np.float32)
conv_out_true = debug_data['conv_outputs'].astype(np.float32)
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
"""

# ---------------- Torch gradient check ------------------------------

cifar_dir = '../Datasets/cifar-10-batches-py/'

trainX, trainY, trainy = LoadBatch(cifar_dir +  'data_batch_1')

f = 4

trainMX = MXFromX(trainX, f)

n_small = 3
MX_small = trainMX[:, :, :n_small]
Y_small = trainY[:, :n_small]
y_small = trainy[:n_small]


nf = 2
nh = 5
K = trainY.shape[0]
lam = 0

init_net = InitializeCNN(f, nf, nh, K, seed=42)

fp_data = ForwardConv(MX_small, init_net)
grads = BackwardConv(MX_small, Y_small, fp_data, init_net, lam)
torch_grads = ComputeGradsWithTorch(MX_small, y_small, init_net)

print('Torch comparison: ')
print("grad_Fs_flat diff:",
      np.max(np.abs(grads['Fs_flat'] - torch_grads['Fs_flat'])))
print("grad_W1 diff:",
      np.max(np.abs(grads['W'][0] - torch_grads['W'][0])))
print("grad_W2 diff:",
      np.max(np.abs(grads['W'][1] - torch_grads['W'][1])))
print("grad_b1 diff:",
      np.max(np.abs(grads['b'][0] - torch_grads['b'][0])))
print("grad_b2 diff:",
      np.max(np.abs(grads['b'][1] - torch_grads['b'][1])))
print("grad_b_conv diff:",
      np.max(np.abs(grads['b_conv'] - torch_grads['b_conv'])))





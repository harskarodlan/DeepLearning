import torch
import numpy as np

def ComputeGradsWithTorch(MX, y, network_params):
    
    MXt = torch.from_numpy(MX)
    MXt = MXt.to(torch.float32)

    L = len(network_params['W'])

    # will be computing the gradient w.r.t. these parameters
    Fs_flat = torch.tensor(network_params['Fs_flat'], requires_grad=True)    
    W = [None] * L
    b = [None] * L    
    for i in range(len(network_params['W'])):
        W[i] = torch.tensor(network_params['W'][i], requires_grad=True)
        b[i] = torch.tensor(network_params['b'][i], requires_grad=True)        

    ## give informative names to these torch classes        
    apply_relu = torch.nn.ReLU()
    apply_softmax = torch.nn.Softmax(dim=0)

    #### BEGIN your code ###########################
    
    # Apply the scoring function corresponding to equations (1-3) in assignment description 
    # If X is d x n then the final scores torch array should have size 10 x n 

    n_p, _, n = MX.shape
    nf = network_params['Fs_flat'].shape[1]

    conv_outputs_mat = torch.zeros((n_p, nf, n))

    for i in range(n):
        conv_outputs_mat[:, :, i] = torch.matmul(MXt[:, :, i], Fs_flat)
    
    b_conv = torch.tensor(network_params['b_conv'], requires_grad=True)

    conv_outputs_mat += b_conv
    
    # ReLU + flatten
    conv_flat = apply_relu(conv_outputs_mat.reshape(n_p*nf, n))
    
    S1 = torch.matmul(W[0], conv_flat) + b[0]     
    X1 = apply_relu(S1)                     
    scores = torch.matmul(W[1], X1) + b[1]  

    #### END of your code ###########################            

    # apply SoftMax to each column of scores     
    P = apply_softmax(scores)
    
    # compute the loss
    n = MX.shape[2]
    loss = torch.mean(-torch.log(P[y, np.arange(n)]))
    
    # compute the backward pass relative to the loss and the named parameters 
    loss.backward()

    # extract the computed gradients and make them numpy arrays 
    grads = {}
    grads['Fs_flat'] = Fs_flat.grad.numpy()
    grads['b_conv'] = b_conv.grad.numpy()
    grads['W'] = [None] * L
    grads['b'] = [None] * L
    for i in range(L):
        grads['W'][i] = W[i].grad.numpy()
        grads['b'][i] = b[i].grad.numpy()

    return grads

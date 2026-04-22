import numpy as np
import time
import matplotlib.pyplot as plt
import copy
import pickle
import torch



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


def PlotPerformance(steps, train_values, val_values, title, ylabel, file_name=None):
    plt.figure()
    plt.plot(steps, train_values, label='training')
    plt.plot(steps, val_values, label='validation')
    plt.xlabel('update step')
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.grid(True)
    if file_name:
        plt.savefig("./images/"+file_name)
    plt.show()


def LoadBatch(filename):
    """
    Loads data batch.

    Args:
        filename: path of data batch to be loaded
    Returns:
        X: image data, (d, n)
        Y: one-hot encoded image labels, (K, n)
        y: integer (int64) image labels, (n, )
    """

    # Load a batch of training data
    with open(filename, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')

    # Extract the image data and cast to float from the dict dictionary
    # n = num images, d = image data dimension = 32*32*3
    # normalize to rgb values to range [0,1]
    X = dict[b'data'].astype(np.float32) / 255.0    # n x d = (10000, 3072)
    X = X.transpose()                               # d x n = (3072, 10000)
    #print(X.shape)

    # Extract integer (int64) image labels, y[i] is int 0-9
    y = np.array(dict[b'labels'])                   # (n,) = (10000,)
    #print(y.shape)
    #print(type(y[0]))

    # Extract one-hot encoded image labels
    K = 10              # num of labels
    n = X.shape[1]      # num of images
    Y = np.zeros((K, n), dtype=X.dtype)     # (K, n) = (10, 10000), float32
    # arange(n) = [0, 1, ..., n-1]
    Y[y, np.arange(n)] = 1      # for each column i, set row y[i] to 1

    return X, Y, y




def LoadAll(dir):
    """
    Loads all 5 data batches.

    Args:
        dir: directory of data batches to be loaded
    Returns:
        X_all: image data, (d, 5n)
        Y_all: one-hot encoded image labels, (K, 5n)
        y_all: integer (int64) image labels, (5n, )
    """
    X_batches = []
    Y_batches = []
    y_batches = []

    for i in range(1,6):
        X, Y, y = LoadBatch(dir + f'data_batch_{i}')
        X_batches.append(X.astype(np.float32))     # X is (d, n)
        Y_batches.append(Y)     # Y is (K, n)
        y_batches.append(y)     # y is (n,)

    X_all = np.concatenate(X_batches, axis=1)     # (d, 5n)
    Y_all = np.concatenate(Y_batches, axis=1)     # (K, 5n)
    y_all = np.concatenate(y_batches, axis=0)     # (5n,)

    return X_all, Y_all, y_all




def NormalizeData(data, mean, std):
    """
    Normalizes image data

    Args:
        data: d x n numpy array
    """
    return (data - mean) / std





def LoadDebugData():
    """
    Loads debugging data.

    Returns:
        X_ims: image data (32, 32, 3, n) = (32, 32, 3, 5)
        debug_data: dict with true matrices Fs, conv_outputs, MX, etc...
    """
    debug_file = 'debug_info.npz'
    debug_data = np.load(debug_file)
    X = debug_data['X'].astype(np.float32)

    n = X.shape[1]

    X_ims = np.transpose(X.reshape((32, 32, 3, n), order='F'), (1, 0, 2, 3))

    return X_ims, debug_data


def SaveMX(MX, filename):
    np.save('./matrices/'+filename, MX)

def LoadMX(filename):
    return np.load('./matrices/'+filename)


def Softmax(S):
    """
    Applies softmax.

    Args:
        S: scores W * X + b,   (K, n)
           Each column is s = Wx + b, where x is one image data vector
    Returns:
        P: probability for each class for each image, (K, n)
    """
    # Shift each score vector s (columns of S) by subtracting max score,
    # improves numerical stability by preventing overflow
    S_shift = S - np.max(S, axis=0, keepdims=True)

    S_exp = np.exp(S_shift)  # (K, n)
    # denumerator is exp(S) with each column summed up (summing all classes scores)
    denum = np.sum(S_exp, axis = 0, keepdims=True)  # (1, n)
    P = S_exp / denum  #  broadcasting divides columnwise: (K,n)/(1, n) = (K, n)
    return P


def ComputeLoss(P, y):
    """
    Computes cross-entropy loss w/o regularization.

    Args:
        P: probability for each class for each image, (K, n)
        y: integer labels, (n, )
    Returns:
        L: cross-entropy loss (scalar)
    """
    n = P.shape[1]

    eps = 1e-15     # for numerical stability: ensure log(0) never evaluated

    # P[y, np.arange(n)]) extracts probilities for correct class
    # apply -log
    # get mean of each -log(py)
    L = -np.mean(np.log(P[y, np.arange(n)] + eps))
    return L



def ComputeAccuracy(P, y):
    """
    Computes accuracy of classifier.

    Args:
        P: probability for each class for each image, (K, n)
        y: integer labels, (n, )
    Returns:
        acc: accuracy = percentage correctly classified images
    """
    # get predicted labels as those with highest probability
    preds = np.argmax(P, axis=0) # (n, )
    # get % correct classifications
    acc = np.mean(preds == y)

    return acc



def CyclicEta(t, eta_min, eta_max, n_s):
    """
    Computes cyclic learning rate eta_t.

    Args:
        t: step in eta cycle
        eta_min: minimum learning rate
        eta_max: maximum learning rate
        n_s: step size

    Returns:
        eta_t: learning rate at step t
    """

    l = t // (2 * n_s)

    if 2*l*n_s <= t <= (2*l + 1) * n_s:
        eta_t = eta_min + ((t - 2*l*n_s) / n_s)*(eta_max - eta_min)
    else:
        eta_t = eta_max - ((t - (2*l + 1)*n_s) / n_s)*(eta_max - eta_min)

    return eta_t


def PrintProgress(t, eta, history):
    """
        Prints current performance statistics.
    """
    print(f"step {t}: eta = {eta:.6f}, "
            f"train loss = {history['train_loss'][-1]:.6f}, "
            f"train cost = {history['train_cost'][-1]:.6f}, "
            f"train acc = {history['train_acc'][-1]:.4f}, "
            f"val loss = {history['val_loss'][-1]:.6f}, "
            f"val cost = {history['val_cost'][-1]:.6f}, "
            f"val acc = {history['val_acc'][-1]:.4f}")


def SlowConv(X_ims, Fs):
    """
    Performs slow convolution (for-loops).

    Args:
        X_ims: image data (32, 32, 3, n) = (32, 32, 3, 5)
        Fs: filters (f, f, 3, nf) = (4, 4, 3, 2)
    Returns:
        conv_out: convolution result 
                  (32/f, 32/f, nf, n) = (8, 8, 2, 5)
    """

    h, w, _, n = X_ims.shape
    f, _, _, nf = Fs.shape

    out_h = h // f
    out_w = w // f
    conv_out = np.zeros((out_h, out_w, nf, n), dtype=X_ims.dtype)

    # for each image
    for i in range(n):
        # for each filter
        for k in range(nf):
            # for each patch row
            for r in range(out_h):
                # for each patch column
                for c in range(out_w):
                    # idx of beginning of patch
                    r0 = r * f
                    c0 = c * f

                    patch = X_ims[r0: r0 + f, c0: c0+f, :, i]

                    # apply k:th filter to patch
                    conv_out[r, c, k, i] = np.sum(patch * Fs[:, :, :, k])
    return conv_out


def BuildMX(X_ims, f):
    """
    Builds matrix MX for convolution of image data.

    Args:
        X_ims: image data (32, 32, 3, n)
        f: filter width
    Returns:
        MX: (n_p, f*f*3, n)
            n_p = 64
    """

    h, w, _, n = X_ims.shape
    out_h = h // f
    out_w = w // f
    n_p = out_h*out_w # num of patches

    # allocate space for MX
    MX = np.zeros((n_p, f*f*3, n), dtype=X_ims.dtype)

    # for each image
    for i in range(n):
        # for each patch, flatten it into a row in MX
        l = 0   # row num in MX
        for r in range(out_h): # patch rows
            for c in range(out_w): # patch columns
                r0 = r*f
                c0 = c*f
                X_patch = X_ims[r0:r0+f, c0:c0+f,:,i]

                MX[l, :, i] = X_patch.reshape((1, f*f*3), order='C')
                
                l += 1

    return MX

def MXFromX(X, f):
    n = X.shape[1]
    X_ims = np.transpose(X.reshape((32, 32, 3, n), order='F'), (1, 0, 2, 3))
    return BuildMX(X_ims, f).astype(np.float32)


def FlattenFilters(Fs):
    """
    Args:
        Fs: unflattened filters (f, f, 3, nf)
    Returns:
        Fs_flat: flattened filters (f*f*3, nf)
    """
    f, _, _, nf = Fs.shape
    Fs_flat = Fs.reshape((f*f*3, nf), order='C')    
    return Fs_flat



def Conv(MX, Fs_flat):
    """
    Performs fast (matrix) convolution.

    Args:
        MX: (n_p, f*f*3, n)
        Fs_flat: filters flattened (f*f*3, nf)
    Returns:
        conv_outputs_mat: convolution result
                          (n_p, nf, n)
    """
    conv_outputs_mat = np.einsum('ijn, jl ->iln', MX, Fs_flat, optimize=True)
    
    return conv_outputs_mat


def ReLU(X):
    return np.maximum(0, X)



def ForwardConv(MX, network):
    """
    Forward pass of assignment 3 (w/ convolution).

    Args:
        MX: image convolution matrix (n_p, f*f*3, n)
        network: dict of network parameters, with
            network['Fs_flat'] = flattened filter for 1st layer
                                 (f*f*3, nf)
            network['W'][0] = W1 - (nh, n_p*nf)
            network['b'][0] = b1 - (nh, 1)
            network['W'][1] = W2 - (10, nh)
            network['b'][1] = b2 - (10, 1)
    Returns:
        fp_data: dict of intermediate forward-pass values, with
            fp_data['conv_outputs_mat']
            fp_data['conv_flat']
            fp_data['S1']
            fp_data['X1]
            fp_data['S']
            fp_data['P]
    """

    W1 = network['W'][0]
    b1 = network['b'][0]
    W2 = network['W'][1]
    b2 = network['b'][1]
    Fs_flat = network['Fs_flat']

    n_p, _, n = MX.shape
    nf = Fs_flat.shape[1]
    
    # 1st layer: convolve X and Fs
    b_conv = network['b_conv']  # (nf, 1)
    b_conv = b_conv.reshape((1, nf, 1))
    conv_outputs_mat = Conv(MX, Fs_flat)    # (n_p, nf, n)
    conv_outputs_mat += b_conv

    # ReLU & flatten convolution result
    # (n_p*nf, n)
    conv_flat = np.fmax(conv_outputs_mat.reshape((n_p*nf, n), order='C'), 0)

    # 2nd layer: fully connected
    S1 = W1 @ conv_flat + b1    # (nh, n)
    X1 = ReLU(S1)               # (nh, n)

    # 3rd layer: fully connected
    S = W2 @ X1 + b2     # (10, n)
    P = Softmax(S)       # (10, n)

    fp_data = {'conv_outputs_mat': conv_outputs_mat, 'conv_flat': conv_flat,
                'S1': S1, 'X1': X1, 'S': S, 'P': P}
    
    return fp_data




def BackwardConv(MX, Y, fp_data, network, lam=0):
    """
    Computes gradients of cost wrt weights W, biases b, filters Fall.

    Args:
        MX: image data convoluton matrix, (n_p, f*f*3, n)
        Y: one-hot encoded image labels, (K, n)
        fp_data: dict of intermediete forward pass values
        network: dict of network parameters, with
            network['Fs_flat'] = flattened filter for 1st layer
                                 (f*f*3, nf)
            network['W'][0] = W1 - (nh, n_p*nf)
            network['b'][0] = b1 - (nh, 1)
            network['W'][1] = W2 - (10, nh)
            network['b'][1] = b2 - (10, 1)
        lam: regularizataion coefficient lambda
    Returns:
        grads: dict of gradients, with
               grads['Fs_flat'] = dJ/dFall
               grads['W'][i] = dJ/dWi, (K, d)
               grads['b'][i] = dJ/dbi, (K, 1)
    """

    W1 = network['W'][0]
    W2 = network['W'][1]
    Fs_flat = network['Fs_flat']

    n_p, _, n = MX.shape
    nf = Fs_flat.shape[1]

    conv_flat = fp_data['conv_flat']      # (n_p*nf, n)
    S1 = fp_data['S1']                    # (nh, n)
    X1 = fp_data['X1']                    # (nh, n)
    P = fp_data['P']                      # (K, n)

    # Follow procedure from lecture 4, slide 34-37:

    # step 1: G_batch = - (Y_batch - P_batch)
    G = P - Y

    # step 2: Add gradient of l wrt b2 & W2
    dJdW2 = (G @ X1.T) / n + 2*lam*W2                # (K, nh)
    # dJ/db2 = 1/nb * G * 1_nb
    dJdb2 = np.sum(G, axis=1, keepdims=True) / n    # (K, 1)

    # step 3: backprop gradient through 2nd layer
    G = W2.T @ G                                    # (nh, n)
    # G = G*ind(S1>0)
    G = G * (S1 > 0)                                # (nh, n)

    # step 4: Add gradient of l wrt b1 & W1
    dJdW1 = (G @ conv_flat.T) / n + 2*lam*W1        # (nh, n_p*nf)
    # dJ/db1 = 1/nb * G * 1_nb
    dJdb1 = np.sum(G, axis=1, keepdims=True) / n    # (nh, 1) 

    # backprop through conv_flat
    G_batch = W1.T @ G                              # (n_p*nf, n)

    # undo reshape from fp
    GG = G_batch.reshape((n_p, nf, n), order='C')

    # backprop through ReLU of conv layer
    GG = GG * (fp_data['conv_outputs_mat'] > 0)

    # gradient wrt flattened filters
    MXt = np.transpose(MX, (1, 0, 2))
    grad_Fs_flat = np.einsum('ijn, jln ->il', MXt, GG, optimize=True) / n

    # account for regularization in filters
    grad_Fs_flat += 2 * lam * Fs_flat

    grad_b_conv = np.sum(GG, axis=(0, 2)).reshape((nf, 1)) / n

    grads = {}
    grads['Fs_flat'] = grad_Fs_flat
    grads['b_conv'] = grad_b_conv
    grads['W'] = [dJdW1, dJdW2]
    grads['b'] = [dJdb1, dJdb2]

    return grads




def InitializeCNN(f, nf, nh, K, seed=42):
  """
  He innitialization of conv net parameters.

  Args:
    f: filter width
    nf: num filters
    nh: num hidden nodes
    K: num classes
 Returns:
    net_params: dict of He initialized network parameters
  """
  rng = np.random.default_rng(seed)

  n_conv = 3*f*f        # num of filter params per filter
  n_p = (32 // f) ** 2  # num of patches
  n_fc1 = n_p*nf        # num of weights for 1st layer

  net_params = {}
  net_params['Fs_flat'] = (
      np.sqrt(2.0/n_conv) * rng.standard_normal((n_conv, nf), dtype=np.float32))
  
  # conv layer bias
  net_params['b_conv'] = np.zeros((nf,1), dtype=np.float32)

  net_params['W'] = [None] * 2
  net_params['b'] = [None] * 2

  net_params['W'][0] = (
      np.sqrt(2.0/n_fc1) * rng.standard_normal((nh, n_fc1), dtype=np.float32))
  net_params['b'][0] = np.zeros((nh, 1), dtype=np.float32)

  net_params['W'][1] = (
      np.sqrt(2.0/nh) * rng.standard_normal((K, nh), dtype=np.float32))
  net_params['b'][1] = np.zeros((K, 1), dtype=np.float32)

  return net_params



def ComputeCostConv(P, y, network, lam):
    """
    Computes cost = loss + regularization term.

    Args:
        P: probability for each class for each image, (K, n)
        y: integer labels, (n, )
        network:  dict of network parameters
        lam: regularizataion coefficient lambda
    Returns:
        cost = loss + regularization term
    """
    loss = ComputeLoss(P, y)
    reg = lam * (sum(np.sum(W ** 2) for W in network['W']) +
          np.sum(network['Fs_flat'] ** 2))
    return loss + reg



def RecordHistoryConv(data, net, lam, eta, t, history):
    """
        Appends current performance statistics to history.
    """

    y = data['trainy']
    y_val = data['validy']

    P_train = ForwardConv(data['trainMX'], net)['P']
    P_val = ForwardConv(data['validMX'], net)['P']

    history['train_loss'].append(ComputeLoss(P_train, y))
    history['train_cost'].append(ComputeCostConv(P_train, y, net, lam))
    history['train_acc'].append(ComputeAccuracy(P_train, y))

    history['val_loss'].append(ComputeLoss(P_val, y_val))
    history['val_cost'].append(ComputeCostConv(P_val, y_val, net, lam))
    history['val_acc'].append(ComputeAccuracy(P_val, y_val))

    history['eta'].append(eta)
    history['step'].append(t)


def Step(trained_net, grads, eta):
    trained_net['Fs_flat'] -= eta * grads['Fs_flat']
    trained_net['b_conv'] -= eta * grads['b_conv']
    trained_net['W'][0] -= eta*grads['W'][0]
    trained_net['b'][0] -= eta*grads['b'][0]
    trained_net['W'][1] -= eta*grads['W'][1]
    trained_net['b'][1] -= eta*grads['b'][1]

    return trained_net


def MiniBatchGDConv(data, GDparams, init_net, lam,
                 seed=None, n_rec=10):
    """
    Performs mini-batch gradient descent to train network parameters.

    Args:
        data: dict with training and validation data
        GDparams: dict of GD parameter values, keys
                  'n_batch' - mini batch size
                  'eta_min' - minimum learning rate of cycle
                  'eta_max' - maximum learning rate of cycle
                  'n_s' - stepsize
                  'n_cycles' - num of cycles
        init_net: initial network parameters, dict with  
        lam: regularization coefficient lambda
        seed: random generator seed for shuffling
        n_rec: num times per cycle performance is recorded
    Returns:
        trained_net: dict of trained network parameters
        history: dict of performance statistics for each epoch, keys
                 'train_loss' - loss after each epoch
                 'train_cost' - cost after each epoch
                 'train_acc' - accuracy after each epoch
    """
    # reset rng for each GD
    local_rng = np.random.default_rng(seed)

    trained_net = copy.deepcopy(init_net)

    MX = data['trainMX']
    Y = data['trainY']
    y = data['trainy']
    MX_val = data['validMX']
    Y_val = data['validY']
    y_val = data['validy']

    n_batch = GDparams['n_batch']
    eta_min = GDparams['eta_min']
    eta_max = GDparams['eta_max']
    n_s = GDparams['n_s']
    n_cycles = GDparams['n_cycles']

    n = MX.shape[2]

    t = 0
    t_end = 2 * n_s * n_cycles

    record_rate = (2 * n_s) // n_rec

    history = {'train_loss': [], 'train_cost': [], 'train_acc': [],
                'val_loss': [], 'val_cost': [], 'val_acc': [],
                'eta': [], 'step': []}

    # run until all cycles done
    while t <= t_end:
        # shuffle dataset before each epoch
        perm = local_rng.permutation(n)
        MX_epoch = MX[:, :, perm]
        Y_epoch = Y[:, perm]

        for j in range(n//n_batch): # go through mini-batches
            if t > t_end:
                break

            # mini batch indices
            j_start = j*n_batch
            j_end = (j+1)*n_batch

            MX_batch = MX_epoch[:, :, j_start:j_end]
            Y_batch = Y_epoch[:, j_start:j_end]

            eta = CyclicEta(t, eta_min, eta_max, n_s)

            # apply mini batch
            fp_data = ForwardConv(MX_batch, trained_net)
            # backprop mini batch
            grads = BackwardConv(MX_batch, Y_batch, fp_data, trained_net, lam)

            # update parameters using GD with mini batch
            trained_net = Step(trained_net, grads, eta)

            if t % record_rate == 0:
                RecordHistoryConv(data, trained_net, lam, eta, t, history)
                PrintProgress(t, eta, history)

            t += 1

    return trained_net, history




def IncreasingCyclicEta(t, eta_min, eta_max, step_1):
    """
    Computes cyclic learning rate eta_t that doubles number 
    of steps each cycle.

    Args:
        t: step in eta cycle
        eta_min: minimum learning rate
        eta_max: maximum learning rate
        step_1: num of steps in first half cycle

    Returns:
        eta_t: learning rate at step t
    """
    c = 0  # cycle number
    step = step_1   # steps until peak of cycle
    tc0 = 0          # start step of cycle

    # find which cycle t is on
    while t >= tc0 + 2*step:
        tc0 += 2*step
        c += 1
        step *= 2   # double steps each cycle

    tc = t - tc0    # cycle-local step counter

    if tc <= step:  
        # increase eta
        eta_t = eta_min + (tc / step) * (eta_max-eta_min)
    else:
        # decrease eta
        eta_t = eta_max - ((tc - step) / step) * (eta_max-eta_min)        

    return eta_t


def RecordHistoryConvSparse(data, test_data, net, eta, t, history, eval_size=1000):
    """
        Appends current performance statistics to history.
    """
    trainMX_eval = data['trainMX'][:,:,:eval_size] 
    trainy_eval = data['trainy'][:eval_size] 

    P_train = ForwardConv(trainMX_eval, net)['P']
    P_test = ForwardConv(test_data['testMX'], net)['P']

    history['train_loss'].append(ComputeLoss(P_train, trainy_eval))
    history['train_acc'].append(ComputeAccuracy(P_train, trainy_eval))

    history['test_loss'].append(ComputeLoss(P_test, test_data['testy']))
    history['test_acc'].append(ComputeAccuracy(P_test, test_data['testy']))

    history['eta'].append(eta)
    history['step'].append(t)



def MiniBatchGDConvLong(data, test_data, GDparams, init_net, lam, seed=42, smooth=False, eps=0.1):
    """
    Performs mini-batch gradient descent to train network parameters.
    Used in the "train for longer" part of exercise 3.
    With increasing cyclic eta and sparse recording.

    Args:
        data: dict with training and validation data
        GDparams: dict of GD parameter values, keys
                  'n_batch' - mini batch size
                  'eta_min' - minimum learning rate of cycle
                  'eta_max' - maximum learning rate of cycle
                  'step_1' - num of steps for first half cycle
                  'n_cycles' - num of cycles
        init_net: initial network parameters, dict with  
        lam: regularization coefficient lambda
        seed: random generator seed for shuffling
        smooth: uses label smoothing if True
        eps: epsilon used in label smoothing
    Returns:
        trained_net: dict of trained network parameters
        history: dict of performance statistics for each epoch, keys
                 'train_loss' - loss after each epoch
                 'train_cost' - cost after each epoch
                 'train_acc' - accuracy after each epoch
    """
    # reset rng for each GD
    local_rng = np.random.default_rng(seed)

    trained_net = copy.deepcopy(init_net)

    MX = data['trainMX']
    Y = data['trainY']
    y = data['trainy']
    MX_val = data['validMX']
    Y_val = data['validY']
    y_val = data['validy']

    n_batch = GDparams['n_batch']
    eta_min = GDparams['eta_min']
    eta_max = GDparams['eta_max']
    step_1 = GDparams['step_1']
    n_cycles = GDparams['n_cycles']

    n = MX.shape[2]

    history = {'train_loss': [], 'train_acc': [],
               'test_loss': [], 'test_acc': [],
                'eta': [], 'step': []}

    t = 0

    # calculate total num steps to take
    t_end = 0
    step = step_1
    for _ in range(n_cycles):
        t_end += 2*step
        step *= 2


    record_rate = step_1 // 2       # record every (step_1 / 2)'th step

    # run until all cycles done
    while t <= t_end:
        # shuffle dataset before each epoch
        perm = local_rng.permutation(n)
        MX_epoch = MX[:, :, perm]
        Y_epoch = Y[:, perm]

        for j in range(n//n_batch): # go through mini-batches
            if t > t_end:
                break

            # mini batch indices
            j_start = j*n_batch
            j_end = (j+1)*n_batch

            MX_batch = MX_epoch[:, :, j_start:j_end]
            Y_batch = Y_epoch[:, j_start:j_end]

            if smooth:
                Y_batch = SmoothLabels(Y_batch, eps)

            eta = IncreasingCyclicEta(t, eta_min, eta_max, step_1)

            # apply mini batch
            fp_data = ForwardConv(MX_batch, trained_net)
            # backprop mini batch
            grads = BackwardConv(MX_batch, Y_batch, fp_data, trained_net, lam)

            # update parameters using GD with mini batch
            trained_net = Step(trained_net, grads, eta)

            if t % record_rate == 0: 
                RecordHistoryConvSparse(data, test_data, trained_net, eta, t, history)
                print(f"step {t}: eta = {eta:.6f}, "
                    f"train loss = {history['train_loss'][-1]:.6f}, "
                    f"train acc = {history['train_acc'][-1]:.4f}, "
                    f"test loss = {history['test_loss'][-1]:.6f}, "
                    f"test acc = {history['test_acc'][-1]:.4f}")

            t += 1

    return trained_net, history


def SmoothLabels(Y, eps):
    """
    Label smoothens one-hot encoded labels Y.
    """

    K = Y.shape[0]
    Y_smooth = (1.0 - eps)*Y + (eps/(K-1))*(1.0-Y)
    return Y_smooth.astype(np.float32)



def Evaluate(trained_net, data, test_data):
        P_train = ForwardConv(data['trainMX'], trained_net)['P']
        train_acc = ComputeAccuracy(P_train, data['trainy'])

        P_val = ForwardConv(data['validMX'], trained_net)['P']
        val_acc = ComputeAccuracy(P_val, data['validy'])

        P_test = ForwardConv(test_data['testMX'], trained_net)['P']
        test_acc = ComputeAccuracy(P_test, test_data['testy'])

        print(f"Training accuracy:   {100 * train_acc:.2f}%")
        print(f"Validation accuracy: {100 * val_acc:.2f}%")
        print(f"Test accuracy:       {100 * test_acc:.2f}%")

        return train_acc, val_acc, test_acc


def BuildAllMX(data, test_data, f):
        trainMX = MXFromX(data['trainX'], f)
        #del trainX      # to free memory
        validMX = MXFromX(data['validX'], f)
        #del validX      # to free memory
        testMX = MXFromX(test_data['testX'], f)
        #del testX       # to free memory

        data['trainMX'] = trainMX
        data['validMX'] = validMX
        test_data['testMX'] = testMX

        return data, test_data


def StoreAllMX(data, test_data, f):
        n = data['trainMX'].shape[2]

        SaveMX(data['trainMX'], 'trainMX_n' + str(n) + '_f' + str(f) + '.npy')
        SaveMX(data['validMX'], 'validMX_n' + str(n) + '_f' + str(f) + '.npy')
        SaveMX(data['testMX'], 'testMX_n' + str(n) + '_f' + str(f) + '.npy')

def LoadAllMX(data, test_data, n, f):
        data['trainMX'] = LoadMX('trainMX_n' + str(n) + '_f' + str(f) + '.npy')
        data['validMX'] = LoadMX('validMX_n' + str(n) + '_f' + str(f) + '.npy')
        test_data['testMX'] = LoadMX('testMX_n' + str(n) + '_f' + str(f) + '.npy')

        return data, test_data


def Ex3InitialRun(data, test_data):
        # Network shape 
        f = 4
        nf = 10 
        nh = 50
        K = data['trainY'].shape[0]

        # Build MX 
        data, test_data = BuildAllMX(data, test_data, f)

        # Initialize net 
        init_net = InitializeCNN(f, nf, nh, K, seed=42)

        # ------------------ Train parameters ------------------------
        lam = 0.003

        eta_min = 1e-5
        eta_max = 1e-1
        n_s = 800
        n_cycles = 3

        GDparams = {'n_batch': 100, 'eta_min': eta_min, 'eta_max': eta_max,
                'n_s': n_s, 'n_cycles': n_cycles}

        # -------------- Train -----------------------------------------
        t0 = time.perf_counter()

        trained_net, history = MiniBatchGDConv(data, GDparams, init_net, lam, seed=42, n_rec=10)

        train_time = time.perf_counter() - t0
        print(f"Training time: {train_time:.2f} s")

        # ---------------- Evaluate -----------------------
        Evaluate(trained_net, data, test_data)

        # ............ Plot performance -------------------
        PlotPerformance(history['step'], history['train_cost'], history['val_cost'],
                        title='Cost plot', ylabel='cost', file_name='ex3_cost')

        PlotPerformance(history['step'], history['train_loss'], history['val_loss'],
                        title='Loss plot', ylabel='loss', file_name='ex3_loss')

        PlotPerformance(history['step'], history['train_acc'], history['val_acc'],
                        title='Accuracy plot', ylabel='accuracy', file_name='ex3_acc')


def RunArchitecture(data, test_data, f, nf, nh, lam, GDparams, n_rec=10):
        K = data['trainY'].shape[0]

        # Build MX 
        data, test_data = BuildAllMX(data, test_data, f)

        # Initialize net 
        init_net = InitializeCNN(f, nf, nh, K, seed=42)

        t0 = time.perf_counter()

        trained_net, history = MiniBatchGDConv(data, GDparams, init_net, lam, seed=42, n_rec=n_rec)

        train_time = time.perf_counter() - t0
        print(f"Training time: {train_time:.2f} s")

        train_acc, val_acc, test_acc  = Evaluate(trained_net, data, test_data)

        del data['trainMX'], data['validMX'], test_data['testMX']       # to free memory

        return history, train_acc, val_acc, test_acc, train_time


def Ex3Comparisons(data, test_data):
        lam = 0.003

        GDparams = {'n_batch': 100, 'eta_min': 1e-5, 'eta_max': 1e-1,
        'n_s': 800, 'n_cycles': 3}

        architectures = [{'label': 'A1', 'f': 2,  'nf': 3, 'nh': 50},
                         {'label': 'A2', 'f': 4,  'nf': 10, 'nh': 50},
                         {'label': 'A3', 'f': 8,  'nf': 40, 'nh': 50},
                         {'label': 'A4', 'f': 16, 'nf': 160, 'nh': 50},]
        
        test_accuracies = []
        train_times = []
        
        for arch in architectures:
                print(f"\nRunning {arch['label']}: f={arch['f']}, nf={arch['nf']}, nh={arch['nh']}")
                _, _, _, test_acc, train_time = RunArchitecture(data, test_data,
                                                                 arch['f'], arch['nf'], arch['nh'],
                                                                 lam, GDparams, n_rec=10)
                
                test_accuracies.append(test_acc)
                train_times.append(train_time)
        
        return test_accuracies, train_times


def Ex3PlotBars(test_accuracies, train_times):
        labels = ['A1', 'A2', 'A3', 'A4']

        plt.figure()
        plt.bar(labels, [100 * acc for acc in test_accuracies])
        plt.ylabel('Test accuracy (%)')
        plt.title('Test accuracy per architecture')
        plt.savefig('./images/test_acc_bar')
        plt.show()

        plt.figure()
        plt.bar(labels, train_times)
        plt.ylabel('Training time (s)')
        plt.title('Training time per architecture')
        plt.savefig('./images/train_time_bar')
        plt.show()



def RunArchitectureLong(data, test_data, f, nf, nh, lam, GDparams, smooth=False, eps=0.1):
        K = data['trainY'].shape[0]

        # Build MX 
        data, test_data = BuildAllMX(data, test_data, f)

        # Initialize net 
        init_net = InitializeCNN(f, nf, nh, K, seed=42)

        t0 = time.perf_counter()

        trained_net, history = MiniBatchGDConvLong(data, test_data, GDparams, init_net, lam,
                                                    seed=42, smooth=smooth, eps=eps)

        train_time = time.perf_counter() - t0
        print(f"Training time: {train_time:.2f} s")

        train_acc, val_acc, test_acc  = Evaluate(trained_net, data, test_data)

        del data['trainMX'], data['validMX'], test_data['testMX']       # to free memory

        return history, train_acc, val_acc, test_acc, train_time


def PlotTrainTestLoss(history, title, filename):
    plt.figure()
    plt.plot(history['step'], history['train_loss'], label='training loss')
    plt.plot(history['step'], history['test_loss'], label='test loss')
    plt.xlabel('update step')
    plt.ylabel('loss')
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.savefig('./images/'+filename)
    plt.close()


def Ex3TrainForLonger(data, test_data):
        lam = 0.003

        GDparams = {'n_batch': 100, 'eta_min': 1e-5, 'eta_max': 1e-1,
        'step_1': 800, 'n_cycles': 3}


        architectures = [{'label': 'A2', 'f': 4,  'nf': 10, 'nh': 50},
                         {'label': 'A3', 'f': 8,  'nf': 40, 'nh': 50},
                         {'label': 'A2 wide', 'f': 4,  'nf': 40, 'nh': 50}]

        test_accuracies = {}
        
        for arch in architectures:
                print(f"\nRunning {arch['label']}: f={arch['f']}, nf={arch['nf']}, nh={arch['nh']}")
                history, _, _, test_acc, _ = RunArchitectureLong(data, test_data,
                                                                 arch['f'], arch['nf'], arch['nh'],
                                                                 lam, GDparams)
                PlotTrainTestLoss(history, arch['label'] + ' loss', arch['label']+'_loss_long')
                test_accuracies[arch['label']] = test_acc
        
        print()
        for label in test_accuracies.keys():
                print(f"{label} : test accuracy = {test_accuracies[label]}")
        
        return


def Ex4(data, test_data):
        f = 4
        nf = 40
        nh = 300

        GDparams = {'n_batch': 100, 'eta_min': 1e-5, 'eta_max': 1e-1,
        'step_1': 800, 'n_cycles': 4}

        lam = 0.0025

        #print("Running archictecture 5 without smoothing")
        #history, _, _, test_acc, _ = RunArchitectureLong(data, test_data, f, nf, nh,
        #                                                 lam, GDparams)
        #PlotTrainTestLoss(history, 'A5 loss, no smoothing', 'A5_loss_no_smoothing')

        
        #print(f"No smoothing: test accuracy = {100*test_acc}")

        lam = 0.0025

        print("Running archictecture 5 with smoothing")
        history, _, _, test_acc, _ = RunArchitectureLong(data, test_data, f, nf, nh,
                                                         lam, GDparams,smooth=True, eps=0.1)
        PlotTrainTestLoss(history, 'A5 loss, with smoothing', 'A5_loss_smoothing')

        
        print(f"Smoothing: test accuracy = {100*test_acc}")
        
        return


############### DEBUGGING PART ##########################################

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

"""

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

"""

############ END OF DEBUGGING PART #############################################


# ------- Load data ----------------------------------------------------

cifar_dir = '../Datasets/cifar-10-batches-py/'


# --------- For 1 batch

#trainX, trainY, trainy = LoadBatch(cifar_dir +  'data_batch_1')
#validX, validY, validy = LoadBatch(cifar_dir +  'data_batch_2')


# --------- For all batches
#"""
n_val = 1000

X, Y, y = LoadAll(cifar_dir)
trainX = X[:, n_val:]
trainY = Y[:, n_val:]
trainy = y[n_val:]
validX = X[:, :n_val]
validY = Y[:, :n_val]
validy = y[:n_val]

del X, Y, y     # to free memory

#"""

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

data = {'trainX': trainX, 'trainY': trainY, 'trainy': trainy,
        'validX': validX, 'validY': validY, 'validy': validy}

test_data = {'testX': testX, 'testY': testY, 'testy':testy}


# ------------------------ Runs -------------------

#Ex3InitialRun(data, test_data)

#test_accuracies, train_times = Ex3Comparisons(data, test_data)
#Ex3PlotBars(test_accuracies, train_times)

#Ex3TrainForLonger(data, test_data)

Ex4(data, test_data)
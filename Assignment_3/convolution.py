import numpy as np
import copy

from ann import Softmax, ComputeLoss, ComputeAccuracy, CyclicEta, PrintProgress

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



def MiniBatchGDConvLong(data, test_data, GDparams, init_net, lam, seed=42):
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
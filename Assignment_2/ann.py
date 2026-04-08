import numpy as np
import copy


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


def ApplyNetwork(X, network, training=False, p_keep=1.0, rng=None):
    """
    Forwards pass.

    Args:
        X: image data, (d, n)
        network: network parameters, dict with  
                 network['W'][0] = W1, shape (m, d)
                 network['b'][0] = b1, shape (m, 1)
                 network['W'][1] = W2, shape (K, m)
                 network['b'][1] = b2, shape (K, 1)
        training: True when training
        p_keep: probability of not dropping a hidden activation
        rng: random generator for dropout

    Returns:
        fp_data: intermediary forward-pass values, dict with keys
                 S1 - (m, n)
                 H - (m, n)
                 S - (K, n)
                 P - (K, n) : probability for each class for each image
                 U - (m, n) : 1 if node kept, 0 if node dropped out 
                              (in hidden layer for each image)
    """

    W1 = network['W'][0]
    b1 = network['b'][0]
    W2 = network['W'][1]
    b2 = network['b'][1]
  
    n = X.shape[1]

    S1 = W1 @ X + b1    # (m, n)
    H = ReLU(S1)        # (m, n)

    U = None
    # DropOut
    if training and p_keep < 1.0:
        U = (rng.random(H.shape) < p_keep) / p_keep
        H = H * U

    S = W2 @ H + b2     # (K, n)
    P = Softmax(S)      # (K, n)

    fp_data = {'S1': S1, 'H': H, 'S': S, 'P': P, 'U': U}

    return fp_data


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


def ComputeCost(P, y, network, lam):
    """
    Computes cost = loss + regularization term.

    Args:
        P: probability for each class for each image, (K, n)
        y: integer labels, (n, )
        network:  network parameters, dict with keys 'W', 'b'
        lam: regularizataion coefficient lambda
    Returns:
        cost = loss + regularization term
    """
    loss = ComputeLoss(P, y)
    reg = lam * sum(np.sum(W ** 2) for W in network['W'])
    return loss + reg


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


def BackwardPass(X, Y, fp_data, network, lam):
    """
    Computes gradients of cost wrt weights W, biases b.

    Args:
        X: image data, (d, n)
        Y: one-hot encoded image labels, (K, n)
        P: probability for each class for each image, (K, n)
        network: network parameters, dict with  
                 network['W'][0] = W1, shape (m, d)
                 network['b'][0] = b1, shape (m, 1)
                 network['W'][1] = W2, shape (K, m)
                 network['b'][1] = b2, shape (K, 1)
        lam: regularizataion coefficient lambda
    Returns:
        grads: dict of gradients, keys 'W, 'b'
                 W - dJ/dW, (K, d)
                 b - dJ/db, (K, 1)
    """
    n = X.shape[1]

    W1 = network['W'][0]
    W2 = network['W'][1]

    H = fp_data['H']      # (m, n)
    P = fp_data['P']      # (K, n)
    U = fp_data['U']      # (m, n)
    S1 = fp_data['S1']    # (m, n)

    # Follow procedure from lecture 4, slide 34-37:

    # step 1: G_batch = - (Y_batch - P_batch)
    G = P - Y

    # step 2: Add gradient of l wrt b2 & W2
    dJdW2 = (G @ H.T) / n + 2*lam*W2                # (K, m)
    # dJ/db2 = 1/nb * G * 1_nb
    dJdb2 = np.sum(G, axis=1, keepdims=True) / n    # (K, 1)

    # step 3: backprop gradient through 2nd layer
    G = W2.T @ G                                    # (m, n)

    # if used dropout in forward pass 
    # backprop through dropout mask
    if U is not None:
        G = G * U
        
    # OBS: with dropout ind(H>0) != ind(S1>0)
    # ReLU uses S1, so use ind(S1>0) instead
    # G = G*ind(S1>0)
    G = G * (S1 > 0)                                 # (m, n)

    # step 4: Add gradient of l wrt b1 & W1
    dJdW1 = (G @ X.T) / n + 2*lam*W1                # (m, d)
    # dJ/db1 = 1/nb * G * 1_nb
    dJdb1 = np.sum(G, axis=1, keepdims=True) / n    # (m, 1)

    grads = {}
    grads['W'] = [dJdW1, dJdW2]
    grads['b'] = [dJdb1, dJdb2]

    return grads




def MiniBatchGD(X, Y, y,  X_val, Y_val, y_val, GDparams, init_net, lam,
                 seed=None, n_rec=10, flip=False, p_keep=1.0):
    """
    Performs mini-batch gradient descent to train network parameters.

    Args:
        X: image data for training, (d, n)
        Y: one-hot encoded image labels for training, (K, n)
        y: integer (int64) image labels for training, (n, )
        X_val: image data for validation, (d, nt)
        Y_val: one-hot encoded image labels for validation, (K, n)
        y_val: integer (int64) image labels for validation, (n, )
        GDparams: dict of GD parameter values, keys
                  'n_batch' - mini batch size
                  'eta_min' - minimum learning rate of cycle
                  'eta_max' - maximum learning rate of cycle
                  'n_s' - stepsize
                  'n_cycles' - num of cycles
        init_net: initial network parameters, dict with  
                 network['W'][0] = W1, shape (m, d)
                 network['b'][0] = b1, shape (m, 1)
                 network['W'][1] = W2, shape (K, m)
                 network['b'][1] = b2, shape (K, 1)
        lam: regularization coefficient lambda
        seed: random generator seed for shuffling
        n_rec: num times per cycle performance is recorded
        flip: if True, training data is flipped w/ 50% chance
        p_keep: probability of not dropping a hidden activation
    Returns:
        trained_net: dict of trained network parameters
        history: dict of performance statistics for each epoch, keys
                 'train_loss' - loss after each epoch
                 'train_cost' - cost after each epoch
                 'train_acc' - accuracy after each epoch
    """
    trained_net = copy.deepcopy(init_net)

    n_batch = GDparams['n_batch']
    eta_min = GDparams['eta_min']
    eta_max = GDparams['eta_max']
    n_s = GDparams['n_s']
    n_cycles = GDparams['n_cycles']

    n = X.shape[1]

    t = 0
    t_end = 2 * n_s * n_cycles

    record_rate = (2 * n_s) // n_rec

    history = {'train_loss': [], 'train_cost': [], 'train_acc': [],
                'val_loss': [], 'val_cost': [], 'val_acc': [],
                'eta': [], 'step': []}

    # reset rng for each GD
    if seed is not None:
        local_rng = np.random.default_rng(seed)
    else:
        local_rng = None

    # get data indices for flipping image
    if flip:
        inds_flip = GetFlipIndices()

    # run until all cycles done
    while t <= t_end:
        # shuffle dataset before each epoch
        if seed is not None:
            perm = local_rng.permutation(n)
            X_epoch = X[:, perm]
            Y_epoch = Y[:, perm]
        else:
            X_epoch = X
            Y_epoch = Y

        for j in range(n//n_batch): # go through mini-batches
            if t > t_end:
                break

            # mini batch indices
            j_start = j*n_batch
            j_end = (j+1)*n_batch

            if flip:
                X_batch = X_epoch[:, j_start:j_end].copy()  # copy for flipping
            else:
                X_batch = X_epoch[:, j_start:j_end]
            Y_batch = Y_epoch[:, j_start:j_end]

            # flip each image with 0.5 chance
            if flip and local_rng is not None:
                flip_mask = local_rng.random(X_batch.shape[1]) < 0.5
                X_batch[:, flip_mask] = X_batch[inds_flip][:, flip_mask]

            eta = CyclicEta(t, eta_min, eta_max, n_s)

            # apply mini batch
            fp_data = ApplyNetwork(X_batch, trained_net, training=True, p_keep=p_keep, rng=local_rng)
            # backprop mini batch
            grads = BackwardPass(X_batch, Y_batch, fp_data, trained_net, lam)

            # update parameters using GD with mini batch
            trained_net['W'][0] -= eta*grads['W'][0]
            trained_net['b'][0] -= eta*grads['b'][0]

            trained_net['W'][1] -= eta*grads['W'][1]
            trained_net['b'][1] -= eta*grads['b'][1]

            if t % record_rate == 0:
                RecordHistory(X, y, X_val, y_val, trained_net, lam, eta, t, history)
                PrintProgress(t, eta, history)

            t += 1

    return trained_net, history



def InitializeNet(d, m, K, seed=42):
  """
  Initialize network parameters.
  """
  rng = np.random.default_rng(seed)

  net_params = {}
  net_params['W'] = [None] * 2
  net_params['b'] = [None] * 2

  net_params['W'][0] = (1 / np.sqrt(d)) * rng.standard_normal((m, d))
  net_params['b'][0] = np.zeros((m, 1))

  net_params['W'][1] = (1 / np.sqrt(m)) * rng.standard_normal((K, m))
  net_params['b'][1] = np.zeros((K, 1))

  return net_params



def ReLU(S):
  return np.maximum(0, S)



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


def RecordHistory(X, y, X_val, y_val, net, lam, eta, t, history):
    """
        Appends current performance statistics to history.
    """
    P_train = ApplyNetwork(X, net)['P']
    P_val = ApplyNetwork(X_val, net)['P']

    history['train_loss'].append(ComputeLoss(P_train, y))
    history['train_cost'].append(ComputeCost(P_train, y, net, lam))
    history['train_acc'].append(ComputeAccuracy(P_train, y))

    history['val_loss'].append(ComputeLoss(P_val, y_val))
    history['val_cost'].append(ComputeCost(P_val, y_val, net, lam))
    history['val_acc'].append(ComputeAccuracy(P_val, y_val))

    history['eta'].append(eta)
    history['step'].append(t)



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
    


def LambdaSearch(lamdas, data, d, m, K, GDparams):

    results = []

    for lam in lamdas:
        print(f"\nTraining with lambda = {lam:.8f}")

        init_net = InitializeNet(d, m, K)

        trained_net, history = MiniBatchGD(
            data['trainX'], data['trainY'], data['trainy'],
            data['validX'], data['validY'], data['validy'],
            GDparams, init_net, lam, seed=42
        )

        best_val_acc = np.max(history['val_acc'])
        best_train_acc = np.max(history['train_acc'])

        results.append({
            'lam': lam,
            'best_val_acc': best_val_acc,
            'best_train_acc': best_train_acc
        })

        print(f"best val acc  = {best_val_acc:.4f}")
    
    results = sorted(results, key=lambda res: res['best_val_acc'], reverse=True)

    return results


def PrintResults(results):
    for res in results:
        print(f"lambda = {res['lam']:.8f}, ", 
              f"best_val_acc = {res['best_val_acc']:.4f}, ", 
              f"best_train_acc = {res['best_train_acc']:.4f}")


def SaveResults(results, file_name):
    with open('./results/' + file_name, "w") as f:
        for res in results:
            f.write(f"lambda={res['lam']:.8f}, "
                f"best_val_acc={res['best_val_acc']:.6f}, "
                f"best_train_acc={res['best_train_acc']:.6f}\n"
            )


def GetFlipIndices():
    """
        Returns indexes of an image flipped.
    """
    aa = np.int32(np.arange(32)).reshape((32, 1))
    bb = np.int32(np.arange(31, -1, -1)).reshape((32, 1))
    vv = np.tile(32 * aa, (1, 32))
    ind_flip = vv.reshape((32 * 32, 1)) + np.tile(bb, (32, 1))
    inds_flip = np.vstack((ind_flip, 1024 + ind_flip))
    inds_flip = np.vstack((inds_flip, 2048 + ind_flip))     # (d, 1)
    inds_flip = inds_flip.flatten()                         # (d,)
    return inds_flip





def MiniBatchGDNesterov(X, Y, y,  X_val, Y_val, y_val, GDparams, init_net, lam,
                 seed=None, n_rec=10, flip=False, p_keep=1.0):
    """
    Performs mini-batch gradient descent to train network parameters.
    With nesterov optimizer.

    Args:
        X: image data for training, (d, n)
        Y: one-hot encoded image labels for training, (K, n)
        y: integer (int64) image labels for training, (n, )
        X_val: image data for validation, (d, nt)
        Y_val: one-hot encoded image labels for validation, (K, n)
        y_val: integer (int64) image labels for validation, (n, )
        GDparams: dict of GD parameter values, keys
                  'n_batch' - mini batch size
                  'eta_min' - minimum learning rate
                  'eta_max' - maximum learning rate
                  'n_s' - stepsize
                  'n_cycles' - num of cycles
                  'gamma' - nesterov parameter
        init_net: initial network parameters, dict with  
                 network['W'][0] = W1, shape (m, d)
                 network['b'][0] = b1, shape (m, 1)
                 network['W'][1] = W2, shape (K, m)
                 network['b'][1] = b2, shape (K, 1)
        lam: regularization coefficient lambda
        seed: random generator seed for shuffling
        n_rec: num times per cycle performance is recorded
        flip: if True, training data is flipped w/ 50% chance
        p_keep: probability of not dropping a hidden activation
    Returns:
        trained_net: dict of trained network parameters
        history: dict of performance statistics for each epoch, keys
                 'train_loss' - loss after each epoch
                 'train_cost' - cost after each epoch
                 'train_acc' - accuracy after each epoch
    """
    trained_net = copy.deepcopy(init_net)

    n_batch = GDparams['n_batch']
    eta_min = GDparams['eta_min']
    eta_max = GDparams['eta_max']
    n_s = GDparams['n_s']
    n_cycles = GDparams['n_cycles']
    gamma = GDparams['gamma']

    n = X.shape[1]

    t = 0
    t_end = 2 * n_s * n_cycles

    record_rate = (2 * n_s) // n_rec

    history = {'train_loss': [], 'train_cost': [], 'train_acc': [],
                'val_loss': [], 'val_cost': [], 'val_acc': [],
                'eta': [], 'step': []}

    # reset rng for each GD
    if seed is not None:
        local_rng = np.random.default_rng(seed)
    else:
        local_rng = None

    # get data indices for flipping image
    if flip:
        inds_flip = GetFlipIndices()

    vel = {}
    vel['W'] = [np.zeros_like(W) for W in trained_net['W']]
    vel['b'] = [np.zeros_like(b) for b in trained_net['b']]

    # run until all cycles done
    while t <= t_end:
        # shuffle dataset before each epoch
        if seed is not None:
            perm = local_rng.permutation(n)
            X_epoch = X[:, perm]
            Y_epoch = Y[:, perm]
        else:
            X_epoch = X
            Y_epoch = Y

        for j in range(n//n_batch): # go through mini-batches
            if t > t_end:
                break

            # mini batch indices
            j_start = j*n_batch
            j_end = (j+1)*n_batch

            if flip:
                X_batch = X_epoch[:, j_start:j_end].copy()  # copy for flipping
            else:
                X_batch = X_epoch[:, j_start:j_end]
            Y_batch = Y_epoch[:, j_start:j_end]

            # flip each image with 0.5 chance
            if flip and local_rng is not None:
                flip_mask = local_rng.random(X_batch.shape[1]) < 0.5
                X_batch[:, flip_mask] = X_batch[inds_flip][:, flip_mask]

            eta = LinearEta(t, eta_min, eta_max, t_end)

            trained_net, vel = NesterovStep(
                X_batch, Y_batch, trained_net, vel,
                eta, gamma, lam,
                p_keep=p_keep, rng=local_rng
            )

            if t % record_rate == 0:
                RecordHistory(X, y, X_val, y_val, trained_net, lam, eta, t, history)
                PrintProgress(t, eta, history)

            t += 1

    return trained_net, history


def LinearEta(t, eta_min, eta_max, t_end):
    """
    Computes linearly decayed eta.
    """
    return eta_max - (eta_max - eta_min) * (t / t_end)



def NesterovStep(X_batch, Y_batch, trained_net, vel, eta, gamma, lam,
                 p_keep=1.0, rng=None):
    """
    Performs one Nesterov step.
    """
    # From lecture 5 slides: let x(t) be parameters (trained_net)
    # First compute estimated parameters for next step:
    # e(t+1) = x(t) - gamma * v(t)
    estimated_net = copy.deepcopy(trained_net)
    estimated_net['W'][0] -= gamma * vel['W'][0]
    estimated_net['b'][0] -= gamma * vel['b'][0]
    estimated_net['W'][1] -= gamma * vel['W'][1]
    estimated_net['b'][1] -= gamma * vel['b'][1]

    # get gradients using estimated parameters e(t+1)
    fp_data = ApplyNetwork(X_batch, estimated_net, training=True, p_keep=p_keep, rng=rng)
    grads = BackwardPass(X_batch, Y_batch, fp_data, estimated_net, lam)

    # update velocity:
    # v(t+1) = gamma * v(t) + eta * grad(e(t+1))
    vel['W'][0] = gamma * vel['W'][0] + eta * grads['W'][0]
    vel['b'][0] = gamma * vel['b'][0] + eta * grads['b'][0]
    vel['W'][1] = gamma * vel['W'][1] + eta * grads['W'][1]
    vel['b'][1] = gamma * vel['b'][1] + eta * grads['b'][1]

    # update parameters
    trained_net['W'][0] -= vel['W'][0]
    trained_net['b'][0] -= vel['b'][0]
    trained_net['W'][1] -= vel['W'][1]
    trained_net['b'][1] -= vel['b'][1]

    return trained_net, vel
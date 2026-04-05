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


def ApplyNetwork(X, network):
    """
    Forwards pass.

    Args:
        X: image data, (d, n)
        network: network parameters, dict with keys 'W', 'b'
                 W - (K, d) weights
                 b - (K, 1) biases
    Returns:
        P: probability for each class for each image, (K, n)
    """
    W = network['W']
    b = network['b']
    n = X.shape[1]

    S = W @ X + b
    P = Softmax(S)

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
    reg = lam * np.sum(network['W'] ** 2)
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


def BackwardPass(X, Y, P, network, lam):
    """
    Computes gradients of cost wrt weights W, biases b.

    Args:
        X: image data, (d, n)
        Y: one-hot encoded image labels, (K, n)
        P: probability for each class for each image, (K, n)
        network: network parameters, dict with keys 'W', 'b'
                 W - (K, d) weights
                 b - (K, 1) biases
        lam: regularizataion coefficient lambda
    Returns:
        grads: dict of gradients, keys 'W, 'b'
                 W - dJ/dW, (K, d)
                 b - dJ/db, (K, 1)
    """
    n = X.shape[1]
    W = network['W']


    # G_batch = - (Y_batch - P_batch)
    G = P - Y

    # formula from lec 3, slide 101
    dJdW = (G @ X.T) / n + 2*lam*W
    # dJ/db = 1//nb* G * 1_nb
    dJdb = np.sum(G, axis=1, keepdims=True) / n

    grads = {'W': dJdW, 'b': dJdb}

    return grads




def MiniBatchGD(X, Y, y,  X_val, Y_val, y_val, GDparams, init_net, lam, seed=None, flip=False):
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
                  'eta' - training rate
                  'n_epochs' - num of epochs
        init_net: dict of initial network parameters, keys
                  'W' - (K, d) weights
                  'b' - (K, 1) biases
        lam: regularization coefficient lambda
        rng: random generator for shuffling
    Returns:
        trained_net: dict of trained network parameters, keys
                     'W' - (K, d) weights
                     'b' - (K, 1) biases
        history: dict of performance statistics for each epoch, keys
                 'train_loss' - loss after each epoch
                 'train_cost' - cost after each epoch
                 'train_acc' - accuracy after each epoch
    """
    trained_net = copy.deepcopy(init_net)

    n_batch = GDparams['n_batch']
    eta = GDparams['eta']
    n_epochs = GDparams['n_epochs']

    n = X.shape[1]

    history = {'train_loss': [], 'train_cost': [], 'train_acc': [],
                'val_loss': [], 'val_cost': [], 'val_acc': []}

    # reset rng for each GD
    if seed is not None:
        local_rng = np.random.default_rng(seed)
    else:
        local_rng = None

    # for 2.1b: flip augmentation
    # get data indices for flipping image
    if flip:
        inds_flip = GetFlipIndices()

    # 1 epoch = 1 run through entire dataset
    for epoch in range(n_epochs):
        # improvement 2.2d: step decay
        #if epoch in [20, 30]:
        #    eta = eta / 10

        # shuffle dataset before each epoch
        if seed is not None:
            perm = local_rng.permutation(n)
            X_epoch = X[:, perm]
            Y_epoch = Y[:, perm]
        else:
            X_epoch = X
            Y_epoch = Y

        for j in range(n//n_batch): # go through mini-batches
            # mini batch indices
            j_start = j*n_batch
            j_end = (j+1)*n_batch

            # mini batch
            if flip:
              X_batch = X_epoch[:, j_start:j_end].copy()      # copy for flipping
            else:
              X_batch = X_epoch[:, j_start:j_end]
            Y_batch = Y_epoch[:, j_start:j_end]

            # for 2.1b: flip augmentation
            # flip each image with 0.5 chance
            if flip and local_rng is not None:
                flip_mask = local_rng.random(X_batch.shape[1]) < 0.5
                X_batch[:, flip_mask] = X_batch[inds_flip][:, flip_mask]

            # apply mini batch
            P_batch = ApplyNetwork(X_batch, trained_net)
            # backprop mini batch
            grads = BackwardPass(X_batch, Y_batch, P_batch, trained_net, lam)

            # update parameters using GD with mini batch
            trained_net['W'] -= eta*grads['W']
            trained_net['b'] -= eta*grads['b']

        # evaluate trained net on original training data after each epoch
        P_epoch = ApplyNetwork(X, trained_net)

        train_loss = ComputeLoss(P_epoch, y)
        train_cost = ComputeCost(P_epoch, y, trained_net, lam)
        train_acc = ComputeAccuracy(P_epoch, y)

        history['train_loss'].append(train_loss)
        history['train_cost'].append(train_cost)
        history['train_acc'].append(train_acc)

        # evaluate trained net on validation data after each epoch
        P_epoch_val = ApplyNetwork(X_val, trained_net)

        val_loss = ComputeLoss(P_epoch_val, y_val)
        val_cost = ComputeCost(P_epoch_val, y_val, trained_net, lam)
        val_acc = ComputeAccuracy(P_epoch_val, y_val)

        history['val_loss'].append(val_loss)
        history['val_cost'].append(val_cost)
        history['val_acc'].append(val_acc)

        print(f"epoch {epoch+1}/{n_epochs}: "
              f"train loss = {train_loss:.6f}, train cost = {train_cost:.6f}, train acc = {train_acc:.4f}, "
              f"val loss = {val_loss:.6f}, val cost = {val_cost:.6f}, val acc = {val_acc:.4f}")

    return trained_net, history



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




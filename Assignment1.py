import numpy as np
import pickle
import copy
import matplotlib.pyplot as plt

from torch_gradient_computations import ComputeGradsWithTorch


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

def NormalizeData(data, mean, std):
    """
    Normalizes image data

    Args:
        data: d x n numpy array
    """
    return (data - mean) / std


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
    # P[y, np.arange(n)]) extracts probilities for correct class
    # apply -log
    # get mean of each -log(py)
    L = -np.mean(np.log(P[y, np.arange(n)]))
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
            X_batch = X_epoch[:, j_start:j_end].copy()      # copy for flipping
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


def VisualizeWeights(network, filename=None):
    """
    Visualizes weights for each class.

    Args:
        network: network parameters, dict with keys 'W', 'b'
        filename: name of file to save visualization image
    """
    fig, axes = plt.subplots(2, 5)
    labels = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']

    Ws = network['W'].transpose().reshape((32, 32, 3, 10), order='F')
    W_im = np.transpose(Ws, (1, 0, 2, 3))

    for i in range(10):
        w_im = W_im[:, :, :, i]
        w_im_norm = (w_im - np.min(w_im)) / (np.max(w_im) - np.min(w_im))

        ax = axes[i // 5, i % 5]
        ax.imshow(w_im_norm)
        ax.set_title(labels[i])
        ax.axis('off')

    plt.tight_layout()
    if filename is not None:
        plt.savefig(filename)
    #plt.show()


def LoadAll(dir):
    X_batches = []
    Y_batches = []
    y_batches = []

    for i in range(1,6):
        X, Y, y = LoadBatch(dir + f'data_batch_{i}')
        X_batches.append(X)     # X is (d, n)
        Y_batches.append(Y)     # Y is (K, n)
        y_batches.append(y)     # y is (n,)
        #print(f'batch {i} has {X.shape[1]} samples')
    
    X_all = np.concatenate(X_batches, axis=1) # (d, 5n)
    Y_all = np.concatenate(Y_batches, axis=1) # (K, 5n)
    y_all = np.concatenate(y_batches, axis=0)     # (5n,)

    #print(X_all.shape, Y_all.shape, y_all.shape)

    return X_all, Y_all, y_all

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


# ---- 1: Load data -------

cifar_dir = './Datasets/cifar-10-batches-py/'

trainX, trainY, trainy = LoadBatch(cifar_dir +  'data_batch_1')
validX, validY, validy = LoadBatch(cifar_dir +  'data_batch_2')

"""
# improvement 2.1a: use all training batches
X, Y, y = LoadAll(cifar_dir)
trainX = X[:, 1000:]
trainY = Y[:, 1000:]
trainy = y[1000:]
validX = X[:, :1000]
validY = Y[:, :1000]
validy = y[:1000]
"""

testX, testY, testy = LoadBatch(cifar_dir +  'test_batch')

#print(trainy[0:10])
#print(trainY[:, 0:10])

d = trainX.shape[0]
n = trainX.shape[1]
K = trainY.shape[0]

# ---- 2: Normalize data -------

mean_X = np.mean(trainX, axis=1).reshape(d, 1)
std_X = np.std(trainX, axis=1).reshape(d, 1)

trainX = NormalizeData(trainX, mean_X, std_X)
validX = NormalizeData(validX, mean_X, std_X)
testX = NormalizeData(testX, mean_X, std_X)


# ---- 3: Initialize parameters -------

# creat random generator object
rng = np.random.default_rng()

# get the BitGenerator used by default_rng
BitGen = type(rng.bit_generator)
# use a seed and save it
# makes initilization repeatable
seed = 42
# use the state from a fresh bit generator
rng.bit_generator.state = BitGen(seed).state

# network represented by dict to hold parameters: keys 'W', 'b'
init_net = {}
# W is (K, d): one weight per class and input feature
# Initialize W randomly normally distributed
init_net['W'] = .01*rng.standard_normal(size = (K, d))  # (K, d)
init_net['W'] = init_net['W'].astype(np.float32)
# b is (K, 1): one bias per class
# initialize b to zero
init_net['b'] = np.zeros((K, 1), dtype=np.float32)                        # (K, 1)



# ------------------------
# bonus points

#lam = 0.1
#eta = 0.001

"""
# improvement 2.1c: grid search
lambda_grid = [0, 1e-4, 1e-3, 1e-2]
eta_grid = [5e-4, 1e-3]
batch_grid = [50, 100]

results = []

for lam in lambda_grid:
    for eta in eta_grid:
        for n_batch in batch_grid:
            # ----- 8: Mini batch gradient descent -----

            print("----------------------------------------")
            print(f"testing parameters: lam={lam:.6f}, eta={eta:.6f}, n_batch={n_batch}")
            print("----------------------------------------")

            GDparams = {'n_batch': n_batch, 'eta': eta, 'n_epochs': 40}

            # train network
            trained_net, history = MiniBatchGD(trainX, trainY, trainy,
                                            validX, validY, validy,
                                            GDparams, init_net, lam, seed=42, flip=True)

            # training accuracy
            P_train = ApplyNetwork(trainX, trained_net)
            train_acc = ComputeAccuracy(P_train, trainy)

            # validation accuracy
            P_val = ApplyNetwork(validX, trained_net)
            val_acc = ComputeAccuracy(P_val, validy)

            results.append({
                'lam': lam,
                'eta': eta,
                'n_batch': n_batch,
                'val_acc': val_acc,
                'net': trained_net
            })

            print(f"lam={lam}, eta={eta}, n_batch={n_batch},",
                   f"val_acc={100*val_acc:.2f}, train_acc={100*train_acc:.2f}")


best_result = max(results, key=lambda r: r['val_acc'])
best_net = best_result['net']

P_test = ApplyNetwork(testX, best_net)
test_acc = ComputeAccuracy(P_test, testy)


# training accuracy
P_train = ApplyNetwork(trainX, best_net)
train_acc = ComputeAccuracy(P_train, trainy)
print(f"Training accuracy: {100 * train_acc:.2f}%")

# validation accuracy
P_val = ApplyNetwork(validX, best_net)
val_acc = ComputeAccuracy(P_val, validy)
print(f"Validation accuracy: {100 * val_acc:.2f}%")

print("Best parameters:")
print(best_result['lam'], best_result['eta'], best_result['n_batch'])
print(f"Test accuracy: {100 * test_acc:.2f}%")
"""

# -----------------

#"""


GDparams = {'n_batch': 100, 'eta': 0.001, 'n_epochs': 20}

# train network FLIPPED
trained_net, history = MiniBatchGD(trainX, trainY, trainy,
                                validX, validY, validy,
                                GDparams, init_net, 0.01, seed=42, flip=True)

print("FLIPPED: ")

# training accuracy
P_train = ApplyNetwork(trainX, trained_net)
train_acc = ComputeAccuracy(P_train, trainy)
print(f"Training accuracy: {100 * train_acc:.2f}%")

# validation accuracy
P_val = ApplyNetwork(validX, trained_net)
val_acc = ComputeAccuracy(P_val, validy)
print(f"Validation accuracy: {100 * val_acc:.2f}%")

P_test = ApplyNetwork(testX, trained_net)
test_acc = ComputeAccuracy(P_test, testy)
print(f"Test accuracy: {100 * test_acc:.2f}%")

#"""

"""                         

GDparams = {'n_batch': 100, 'eta': 0.001, 'n_epochs':40}       

# train network NOT FLIPPED
trained_net, history = MiniBatchGD(trainX, trainY, trainy,
                                validX, validY, validy,
                                GDparams, init_net, 0.01, seed=42, flip=False)


print("NOT FLIPPED: ")

# training accuracy
P_train = ApplyNetwork(trainX, trained_net)
train_acc = ComputeAccuracy(P_train, trainy)
print(f"Training accuracy: {100 * train_acc:.2f}%")

# validation accuracy
P_val = ApplyNetwork(validX, trained_net)
val_acc = ComputeAccuracy(P_val, validy)
print(f"Validation accuracy: {100 * val_acc:.2f}%")

P_test = ApplyNetwork(testX, trained_net)
test_acc = ComputeAccuracy(P_test, testy)
print(f"Test accuracy: {100 * test_acc:.2f}%")

"""
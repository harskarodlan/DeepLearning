import numpy as np
import copy
import matplotlib.pyplot as plt
import pickle
import torch
from math import floor


def ComputeGradsWithTorch(X, y, network_params):
    
    Xt = torch.from_numpy(X)
    Xt = Xt.to(torch.float64)

    L = len(network_params['W'])

    # will be computing the gradient w.r.t. these parameters    
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

    S1 = torch.matmul(W[0], Xt) + b[0]     # (m, n)
    H = apply_relu(S1)                     # (m, n)
    scores = torch.matmul(W[1], H) + b[1]  # (K, n)

    #### END of your code ###########################            

    # apply SoftMax to each column of scores     
    P = apply_softmax(scores)
    
    # compute the loss
    n = X.shape[1]
    loss = torch.mean(-torch.log(P[y, np.arange(n)]))
    
    # compute the backward pass relative to the loss and the named parameters 
    loss.backward()

    # extract the computed gradients and make them numpy arrays 
    grads = {}
    grads['W'] = [None] * L
    grads['b'] = [None] * L
    for i in range(L):
        grads['W'][i] = W[i].grad.numpy()
        grads['b'][i] = b[i].grad.numpy()

    return grads


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
        X_batches.append(X)     # X is (d, n)
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
        network: network parameters, dict with  
                 network['W'][0] = W1, shape (m, d)
                 network['b'][0] = b1, shape (m, 1)
                 network['W'][1] = W2, shape (K, m)
                 network['b'][1] = b2, shape (K, 1)

    Returns:
        fp_data: intermediary forward-pass values, dict with keys
                 S1 - (m, n)
                 H - (m, n)
                 S - (K, n)
                 P - (K, n) : probability for each class for each image
                 
    """

    W1 = network['W'][0]
    b1 = network['b'][0]
    W2 = network['W'][1]
    b2 = network['b'][1]
  
    n = X.shape[1]

    S1 = W1 @ X + b1    # (m, n)
    H = ReLU(S1)        # (m, n)
    S = W2 @ H + b2     # (K, n)
    P = Softmax(S)      # (K, n)

    fp_data = {'S1': S1, 'H': H, 'S': S, 'P': P}

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

    # Follow procedure from lecture 4, slide 34-37:

    # step 1: G_batch = - (Y_batch - P_batch)
    G = P - Y

    # step 2: Add gradient of l wrt b2 & W2
    dJdW2 = (G @ H.T) / n + 2*lam*W2                # (K, m)
    # dJ/db2 = 1/nb * G * 1_nb
    dJdb2 = np.sum(G, axis=1, keepdims=True) / n    # (K, 1)

    # step 3: backprop gradient through 2nd layer
    G = W2.T @ G                                    # (m, n)
    # G = G*ind(H>0)
    G = G * (H > 0)                                 # (m, n)

    # step 4: Add gradient of l wrt b1 & W1
    dJdW1 = (G @ X.T) / n + 2*lam*W1                # (m, d)
    # dJ/db1 = 1/nb * G * 1_nb
    dJdb1 = np.sum(G, axis=1, keepdims=True) / n    # (m, 1)

    grads = {}
    grads['W'] = [dJdW1, dJdW2]
    grads['b'] = [dJdb1, dJdb2]

    return grads




def MiniBatchGD(X, Y, y,  X_val, Y_val, y_val, GDparams, init_net, lam, seed=None, n_rec=10):
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

            X_batch = X_epoch[:, j_start:j_end]
            Y_batch = Y_epoch[:, j_start:j_end]

            eta = CyclicEta(t, eta_min, eta_max, n_s)

            # apply mini batch
            fp_data = ApplyNetwork(X_batch, trained_net)
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

# -------- Initialization test -----------------------------------------
"""

m = 50

net = InitializeNet(d, m, K)

print(trainX.shape, trainY.shape, trainy.shape)
print(net['W'][0].shape, net['b'][0].shape)
print(net['W'][1].shape, net['b'][1].shape)

"""


# ---------- Forward pass test -----------------------------------------
"""

X_small = trainX[:, 0:5]
fp_data = ApplyNetwork(X_small, net)

print("P shape:", fp_data['P'].shape)                 # (10, 5)
print("s1 shape:", fp_data['S1'].shape)    # (50, 5)
print("h shape:", fp_data['H'].shape)      # (50, 5)
print("s shape:", fp_data['S'].shape)      # (10, 5)

"""


# --------- Backward pass test ----------------------------------------


# ------------ Test with PyTorch
"""

d_small = 5
n_small = 3
m = 6
lam = 0
small_net = InitializeNet(d_small, m, K)


X_small = trainX[0:d_small, 0:n_small]
Y_small = trainY[:, 0:n_small]
fp_data = ApplyNetwork(X_small, small_net)
my_grads = BackwardPass(X_small, Y_small, fp_data, small_net, lam)

torch_grads = ComputeGradsWithTorch(X_small, trainy[0:n_small], small_net)

# Print comparison error between analytic and PyTorch gradients
for layer in range(len(my_grads['W'])):
    abs_diff_W = np.max(np.abs(my_grads['W'][layer] - torch_grads['W'][layer]))
    abs_diff_b = np.max(np.abs(my_grads['b'][layer] - torch_grads['b'][layer]))

    rel_diff_W = abs_diff_W / np.maximum(
        1e-12,
        np.max(np.abs(my_grads['W'][layer]) + np.abs(torch_grads['W'][layer]))
    )
    rel_diff_b = abs_diff_b / np.maximum(
        1e-12,
        np.max(np.abs(my_grads['b'][layer]) + np.abs(torch_grads['b'][layer]))
    )

    print(f"Layer {layer+1}")
    print(f"  W max abs diff: {abs_diff_W:.10e}")
    print(f"  W max rel diff: {rel_diff_W:.10e}")
    print(f"  b max abs diff: {abs_diff_b:.10e}")
    print(f"  b max rel diff: {rel_diff_b:.10e}")

"""


# ---------- Test overfitting 
"""

m = 50
lam = 0
small_net = InitializeNet(d, m, K)

X_tiny = trainX[:, :100]
Y_tiny = trainY[:, :100]
y_tiny = trainy[:100]

X_tiny_val = validX[:, :100]
Y_tiny_val = validY[:, :100]
y_tiny_val = validy[:100]


GDparams = {'n_batch': 10, 'eta': 0.01, 'n_epochs': 200}

trained_net, history = MiniBatchGD(X_tiny, Y_tiny, y_tiny,
                                X_tiny_val, Y_tiny_val, y_tiny_val,
                                GDparams, small_net, lam, seed=42)

# training accuracy
P_train = ApplyNetwork(X_tiny, trained_net)['P']
train_acc = ComputeAccuracy(P_train, y_tiny)
print(f"Training accuracy: {100 * train_acc:.2f}%")

# validation accuracy
P_val = ApplyNetwork(X_tiny_val, trained_net)['P']
val_acc = ComputeAccuracy(P_val, y_tiny_val)
print(f"Validation accuracy: {100 * val_acc:.2f}%")

"""

# .................. Exercise 3: cyclic GD ------------------------
"""

m = 50
lam = 0.01

net = InitializeNet(d, m, K)

eta_min = 1e-5
eta_max = 1e-1
n_s = 500
n_cycles = 1

GDparams = {'n_batch': 100, 'eta_min': eta_min, 'eta_max': eta_max,
             'n_s': n_s, 'n_cycles': n_cycles}


trained_net, history = MiniBatchGD(
    trainX, trainY, trainy,
    validX, validY, validy,
    GDparams, net, lam, seed=42
)


# ............. Plot eta

plt.figure()
plt.plot(history['step'], history['eta'])
plt.xlabel('update step')
plt.ylabel('eta')
plt.title('Cyclic learning rate')
plt.grid(True)
plt.show()


# ............ Plot performance

PlotPerformance(history['step'], history['train_cost'], history['val_cost'],
                title='Cost plot', ylabel='cost', file_name='ex3_cost')

PlotPerformance(history['step'], history['train_loss'], history['val_loss'],
                title='Loss plot', ylabel='loss', file_name='ex3_loss')

PlotPerformance(history['step'], history['train_acc'], history['val_acc'],
                title='Accuracy plot', ylabel='accuracy', file_name='ex3_acc')

"""


# -------------------- Exercise 4 ------------------

# --------------- Proper run
"""

m = 50
lam = 0.01

net = InitializeNet(d, m, K)

eta_min = 1e-5
eta_max = 1e-1
n_s = 800
n_cycles = 3

GDparams = {
    'n_batch': 100,
    'eta_min': eta_min,
    'eta_max': eta_max,
    'n_s': n_s,
    'n_cycles': n_cycles
}

trained_net, history = MiniBatchGD(
    trainX, trainY, trainy,
    validX, validY, validy,
    GDparams, net, lam, seed=42, n_rec=9
)


PlotPerformance(history['step'], history['train_cost'], history['val_cost'],
                title='Cost plot', ylabel='cost', file_name='fig4_cost')

PlotPerformance(history['step'], history['train_loss'], history['val_loss'],
                title='Loss plot', ylabel='loss', file_name='fig4_loss')

PlotPerformance(history['step'], history['train_acc'], history['val_acc'],
                title='Accuracy plot', ylabel='accuracy', file_name='fig4_acc')

"""


# ----------------- Lambda search --------------------
"""

m = 50
n_batch = 100
n_s = 2 * floor(n / n_batch)
eta_min = 1e-5
eta_max = 1e-1
n_cycles = 3

GDparams = {
    'n_batch': n_batch,
    'eta_min': eta_min,
    'eta_max': eta_max,
    'n_s': n_s,
    'n_cycles': n_cycles
}

"""


# ----------------- Coarse 
"""

# get a uniform (log) grid of 8 lambda values
lambda_values = np.logspace(-5, -1, 8)

results = LambdaSearch(lambda_values, data, d ,m, K, GDparams)

print("\nCoarse lambda search results: ")
PrintResults(results)
SaveResults(results, "coarse_lamda_search.txt")

# Gives best lambda as:
# lambda = 0.00013895,  best_val_acc = 0.5292

"""


# ---------------- Fine
"""

# get a uniform (log) grid of 8 lambda values
lambda_values = np.logspace(-4, -2.7, 10)

results = LambdaSearch(lambda_values, data, d ,m, K, GDparams)

print("\nFine lambda search results: ")
PrintResults(results)
SaveResults(results, "fine_lamda_search.txt")

# Gives best lambda as:
# lambda=0.00199526, best_val_acc=0.532400

"""


# -------------------------- Final training ----------------------
#"""

lam_best = 0.00199526

m = 50
n_batch = 100
n_s = 2 * floor(n / n_batch)
eta_min = 1e-5
eta_max = 1e-1
n_cycles = 3

GDparams = {
    'n_batch': n_batch,
    'eta_min': eta_min,
    'eta_max': eta_max,
    'n_s': n_s,
    'n_cycles': n_cycles
}


net = InitializeNet(d, m, K)

trained_net, history = MiniBatchGD(
    trainX, trainY, trainy,
    validX, validY, validy,
    GDparams, net, lam_best, seed=42
)

P_train = ApplyNetwork(trainX, trained_net)['P']
train_acc = ComputeAccuracy(P_train, trainy)

P_val = ApplyNetwork(validX, trained_net)['P']
val_acc = ComputeAccuracy(P_val, validy)

P_test = ApplyNetwork(testX, trained_net)['P']
test_acc = ComputeAccuracy(P_test, testy)

print(f"Best lambda: {lam_best:.8f}")
print(f"Training accuracy:   {100 * train_acc:.2f}%")
print(f"Validation accuracy: {100 * val_acc:.2f}%")
print(f"Test accuracy:       {100 * test_acc:.2f}%")

PlotPerformance(history['step'], history['train_cost'], history['val_cost'],
                title='Cost plot', ylabel='cost', file_name='final_cost')

PlotPerformance(history['step'], history['train_loss'], history['val_loss'],
                title='Loss plot', ylabel='loss', file_name='final_loss')

PlotPerformance(history['step'], history['train_acc'], history['val_acc'],
                title='Accuracy plot', ylabel='accuracy', file_name='final_acc')

#"""


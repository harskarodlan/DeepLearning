import numpy as np
import pickle


def LoadBatch(filename):
    # Load a batch of training data
    with open(filename, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    
    # Extract the image data and cast to float from the dict dictionary
    # n = num images, d = image data dimension = 32*32*3
    # normalize to rgb values to range [0,1]
    X = dict[b'data'].astype(np.float64) / 255.0    # n x d = (10000, 3072)
    X = X.transpose()                               # d x n = (3072, 10000)
    #print(X.shape)

    # Extract integer (int64) image labels, y[i] is int 0-9
    y = np.array(dict[b'labels'])                   # (n,) = (10000,)
    #print(y.shape)
    #print(type(y[0]))

    # Extract one-hot encoded image labels
    K = 10              # num of labels
    n = X.shape[1]      # num of images
    Y = np.zeros((K, n), dtype=X.dtype)     # (K, n) = (10, 10000), float64
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

    S = W @ X + b @ np.ones((1, n))
    P = Softmax(S)

    return P

# ---- 1: Load data -------

cifar_dir = './Datasets/cifar-10-batches-py/'
trainX, trainY, trainy = LoadBatch(cifar_dir +  'data_batch_1')
validX, validY, validy = LoadBatch(cifar_dir +  'data_batch_2')
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
# b is (K, 1): one bias per class
# initialize b to zero
init_net['b'] = np.zeros((K, 1))                        # (K, 1)

# ----- 4: Forward pass -----

P = ApplyNetwork(trainX[:, 0:100], init_net)
print(P.shape)          # (10, 100)
print(np.sum(P[:, 0]))  # about 1.0
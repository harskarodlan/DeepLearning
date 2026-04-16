import numpy as np
import pickle
import copy



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





def LoadDebugData():
    """
    Loads debugging data.

    Returns:
        X_ims: image data (32, 32, 3, n) = (32, 32, 3, 5)
        Fs: filters (f, f, 3, nf) = (4, 4, 3, 2)
        conv_out: correct convolution result
                  (32/f, 32/f, nf, n) = (8, 8, 2, 5)
    """
    debug_file = 'debug_info.npz'
    load_data = np.load(debug_file)
    X = load_data['X']
    Fs = load_data['Fs']
    conv_out = load_data['conv_outputs']

    n = X.shape[1]

    X_ims = np.transpose(X.reshape((32, 32, 3, n), order='F'), (1, 0, 2, 3))

    return X_ims, Fs, conv_out



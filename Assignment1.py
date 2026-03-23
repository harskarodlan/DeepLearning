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
    data: d x n numpy array
    """
    return (data - mean) / std



cifar_dir = './Datasets/cifar-10-batches-py/'
trainX, trainY, trainy = LoadBatch(cifar_dir +  'data_batch_1')
#validX, validY, validy = LoadBatch(cifar_dir +  'data_batch_2')
#testX, testY, testy = LoadBatch(cifar_dir +  'data_batch_3')

#print(trainy[0:10])
#print(trainY[:, 0:10])

d = trainX.shape[0]
mean_X = np.mean(trainX, axis=1).reshape(d, 1)
std_X = np.std(trainX, axis=1).reshape(d, 1)

trainX  = NormalizeData(trainX, mean_X, std_X)


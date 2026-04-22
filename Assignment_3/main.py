import numpy as np
import time
import matplotlib.pyplot as plt
import pickle

from plotting import PlotPerformance
from ann import ComputeAccuracy
from data_handling import *
from convolution import *


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
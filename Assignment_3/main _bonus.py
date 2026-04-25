import numpy as np
import time
import matplotlib.pyplot as plt

from plotting import PlotPerformance
from ann import ComputeAccuracy
from data_handling import *
from convolution import *



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


def RunArchitectureBonus(data, test_data, f, nf, nh, lam, GDparams, 
                        flip=False, smooth=False, eps=0.1, decay=1):
        K = data['trainY'].shape[0]

        # Build validation/test MX 
        data['validMX'] = MXFromX(data['validX'], f)
        test_data['testMX'] = MXFromX(test_data['testX'], f)
        del data['validX'], test_data['testX']  # to free memory

        # Initialize net 
        init_net = InitializeCNN(f, nf, nh, K, seed=42)

        t0 = time.perf_counter()

        trained_net = MiniBatchGDConvBonus(data, test_data, GDparams, init_net, lam, f,
                                                    seed=42,flip=True, smooth=smooth, eps=eps, decay=decay)

        train_time = time.perf_counter() - t0
        print(f"Training time: {train_time:.2f} s")

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

f = 4
nf = 60
nh = 300
lam = 0.0025

GDparams = {
    'n_batch': 100,
    'eta_min': 1e-5,
    'eta_max': 1e-1,
    'step_1': 800,
    'n_cycles': 4
}

RunArchitectureBonus(data, test_data, f, nf, nh, lam, GDparams,
                     flip=True, smooth=True, eps=0.1, decay=1)

                     
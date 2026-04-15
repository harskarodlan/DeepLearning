import numpy as np
import matplotlib.pyplot as plt


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
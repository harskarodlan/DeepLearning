import matplotlib.pyplot as plt

def PlotSmoothLoss(smooth_losses, filename):

    plt.figure()
    plt.plot(smooth_losses)
    plt.xlabel("Update step")
    plt.ylabel("Smooth loss")
    plt.savefig('./images/'+filename)
    plt.show()
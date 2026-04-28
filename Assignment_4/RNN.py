import numpy as np

def InitializeRNN(K, m, seed=42):
    """
    Inititalizes RNN using normal dist for weights, zero for biases.

    Args: 
        K: dim of output/input to RNN
        m: dim of hidden state
        seed: seed for random generator used for weight initialization
    Returns:
        RNN: dict of RNN parameters. with
             RNN['b'] - (m, 1)
             RNN['c'] - (K, 1)
             RNN['U'] - (m, K)
             RNN['W'] - (m, m)
             RNN['V'] - (K, m)
    """
    rng = np.random.default_rng(seed)

    RNN = {}

    RNN['b'] = np.zeros((m,1))
    RNN['c'] = np.zeros((K,1))

    RNN['U'] = (1/np.sqrt(2*K))*rng.standard_normal(size = (m, K))
    RNN['W'] = (1/np.sqrt(2*m))*rng.standard_normal(size = (m, m))
    RNN['V'] = (1/np.sqrt(m))*rng.standard_normal(size = (K, m))

    return RNN




def Softmax(ot):
    """
    Applies softmax.

    Args:
        ot: V*ht + c
    Return:
        pt: probabilities of each label
    """
    exp_ot = np.exp(ot)
    return exp_ot/np.sum(exp_ot, axis=0)
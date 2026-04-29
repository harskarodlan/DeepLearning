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




def SoftMax(ot):
    """
    Applies softmax.

    Args:
        ot: V*ht + c
    Return:
        pt: probabilities of each label
    """
    exp_ot = np.exp(ot)
    return exp_ot/np.sum(exp_ot, axis=0)



def Synthesize(RNN, h0, x0, n, rng):
    """
    Synthesize text from initial input.

    Args:
        RNN: dict of weights/biases of RNN
        h0: initial hidden state, (m, 1)
        x0: initial input, one hot encoded char index, (K, 1)
        n: length of synthesized text
        rng:  random number generator for label sampling
    Returns:
        Y: synthesized sequence as one-hot encoded indices (K, n)
    """

    K = x0.shape[0]

    Y = np.zeros((K, n))

    h = h0
    x = x0

    for t in range(n):
        # eq. 1: a_t = W*h_{t-1} + U*x_t  + b
        a = RNN['W'] @ h + RNN['U'] @ x + RNN['b']
        # eq. 2: h_t = tanh(a_t)
        h = np.tanh(a)
        # eq. 3: o_t = V*h_t + c
        o = RNN['V'] @ h + RNN['c']
        # eq. 4: p_t = SoftMax(o_t)
        p = SoftMax(o)

        # Sample label from probabilities
        cp = np.cumsum(p, axis=0)
        a = rng.uniform(size=1)
        ii = np.argmax(cp - a > 0)

        # predicted x_{t+1} as one-hot encoded
        xnext = np.zeros((K, 1))
        xnext[ii] = 1

        # save prediction at step t
        Y[:, t] = xnext[:, 0]  

        x = xnext

    return Y


def YtoString(Y, ind_to_char):
    """
    Convert one-hot encoded char index sequence to string.

    Args:
        Y: sequence as one-hot encoded char indices (K, n)
    Returns: 
        text: Y converted into string
    """
    n = Y.shape[0]

    text = ""

    for i in range(n):
        y = Y[:, i]
        idx = np.argmax(y)
        c = ind_to_char[idx]
        text = text + c
    
    return text








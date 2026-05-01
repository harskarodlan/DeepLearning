import numpy as np
from optimizer import InitAdam, AdamStep
from data_handling import StrToOneHot, OneHotToStr

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



def ForwardPass(X, Y, RNN, h0):
    """
    Forward pass for RNN.

    Args:
        X: input one-hot encoded char index sequence, (K, seq_length)
        Y: target one-hot encoded char index sequence, (K, seq_length)
        RNN: dict of RNN params
        h0: initial hidden state, m x 1
    Returns:
        loss: avg cross entropy loss
        fp: dict of intermediate values for backprop
    """

    K, seq_length = X.shape
    m = h0.shape[0]

    # matrices for intermediete values
    A = np.zeros((m, seq_length))
    H = np.zeros((m, seq_length + 1))
    O = np.zeros((K, seq_length))
    P = np.zeros((K, seq_length))

    # using H[:, 0:1] instead of H[:, 0] to get
    # shape (m, 1) instead of (m,)
    H[:, 0:1] = h0
    
    loss = 0

    for t in range(seq_length):
        x = X[:, t:t+1]
        y = Y[:, t:t+1]
        h = H[:, t:t+1]

        # eq. 1: a_t = W*h_{t-1} + U*x_t  + b
        a = RNN['W'] @ h + RNN['U'] @ x + RNN['b']
        # eq. 2: h_t = tanh(a_t)
        h = np.tanh(a)
        # eq. 3: o_t = V*h_t + c
        o = RNN['V'] @ h + RNN['c']
        # eq. 4: p_t = SoftMax(o_t)
        p = SoftMax(o)

        # save intermediate values
        A[:, t:t+1] = a
        H[:, t+1:t+2] = h
        O[:, t:t+1] = o
        P[:, t:t+1] = p

        # add cross entropy loss for step t
        loss += -np.log(np.sum(y*p))

    # average loss
    loss = loss / seq_length

    fp = {}
    fp['A'] = A
    fp['H'] = H
    fp['O'] = O
    fp['P'] = P

    return loss, fp


def BackwardPass(X, Y, RNN, fp):
    """
    Backward pass for vanilla RNN.

    Args:
        RNN: dict of RNN params
        fp: dict of intermediate values from forward pass
    Returns:
        grads: dict of gradients
    """

    H = fp['H']
    P = fp['P']

    K, seq_length = X.shape
    m = H.shape[0]

    grads = {}
    grads['b'] = np.zeros_like(RNN['b'])    # (m,1)
    grads['c'] = np.zeros_like(RNN['c'])    # (K,1)
    grads['U'] = np.zeros_like(RNN['U'])    # (m,K)
    grads['W'] = np.zeros_like(RNN['W'])    # (m,m)
    grads['V'] = np.zeros_like(RNN['V'])    # (K,m)

    grad_anext = np.zeros((m, 1)) 

    #  for t = seq_length - 1, ..., 0
    for t in reversed(range(seq_length)):
        x = X[:, t:t+1]     # (K,1)
        y = Y[:, t:t+1]     # (K,1)
        p = P[:, t:t+1]     # (K,1)

        h = H[:, t+1:t+2]   # (m,1)
        hprev = H[:, t:t+1] # (m,1)

        # Lec 8 slide 32: dL/dot = -(yt - pt)^T
        # then avg over sequence
        grad_o = (p -y)/seq_length  # (K,1)

        # Lec 8 slide 33: dL/dV = sum_t gt^T*ht^T
        # with gt = grad_o
        grads['V'] += grad_o @ h.T  # (K,m)
        
        grads['c'] += grad_o        # (K,1)

        # Lec 8 slide 37: dL/dht = dL/dot * V + dL/a_{t+1} * W
        # (m,K)*(K,1) + (m,m)*(m,1)
        grad_h = RNN['V'].T @ grad_o + RNN['W'].T @ grad_anext # (m,1)
        # Lec 8 slide 37: dL/dat = dL/dht * diag(1-tanh²(at))
        grad_a = grad_h * (1-h**2)                 # (m,1)

        grads['b'] += grad_a

        # Lec 8 slide 33: dL/dW = sum_t gt^T*h_{t-1}^T
        # with gt = grad_a
        grads['W'] += grad_a @ hprev.T
    
        # Lec 8 slide 39: dL/dU = sum_t gt^T*xt^T
        # with gt = grad_a
        grads['U'] += grad_a @ x.T

        # dL/da_{t+1}
        grad_anext = grad_a
    
    # clip to avoid exploding gradients
    for kk in grads.keys():
        grads[kk] = np.clip(grads[kk], -5, 5)

    return grads



def TrainRNN(book_data, char_to_ind, ind_to_char, RNN, eta, seq_length, n_updates, seed=42):
    """
    Train RNN using Adam optimizer.

    Args:
        book_data: full training text
        char_to_ind: dict mapping character -> index
        ind_to_char: dict mapping index -> character
        RNN: dict of RNN params
        eta: learning rate
        seq_length: sequence length for each update step
        n_updates: number of update steps
    Returns:
        RNN: trained RNN params
        smooth_losses: list of smooth loss values
    """

    K = len(char_to_ind)
    m = RNN['W'].shape[0]

    m_adam, v_adam = InitAdam()

    hprev = np.zeros((m,1))
    smooth_loss = None
    smooth_losses = []
    e = 0

    rng = np.random.default_rng(seed)

    for t in range(1, n_updates+1):
        # if finished 1 epoch = 1 run through whole book_data
        if e + seq_length + 1 >= len(book_data):
            e = 0   # reset cursor
            hprev = np.zeros((m,1)) # reset input hidden state
        
        X_chars = book_data[0:seq_length]
        Y_chars = book_data[1:seq_length+1]

        X = StrToOneHot(X_chars, char_to_ind, K)
        Y = StrToOneHot(Y_chars, char_to_ind, K)


        loss, fp = ForwardPass(X, Y, RNN, hprev)
        grads = BackwardPass(X, Y, RNN, fp)

        RNN, m_adam, v_adam = AdamStep(RNN, grads, m_adam, v_adam, t, eta,)

        hprev = fp['H'][:,-1:]

        if smooth_loss is None:
            smooth_loss = loss
        else:
            smooth_loss = .999* smooth_loss + .001 * loss
        smooth_losses.append(smooth_loss)

        e += seq_length

        if t % 100 == 0:
            print("update:", t, "smooth loss:", smooth_loss)

        if t % 10000 == 0:
            x0 = X[:, 0:1]
            Y_sample = Synthesize(RNN, hprev, x0, 200, rng)

            print("synthesized text at update", t, ": ")
            print(OneHotToStr(Y_sample, ind_to_char))
            print("------------------------------------------")

    return RNN, smooth_losses
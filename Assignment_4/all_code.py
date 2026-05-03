import numpy as np
import copy
import matplotlib.pyplot as plt


def LoadBookData(book_dir="./"):
    """
    Loads book text from text file.

    Args:
        book_dir: directory of book text file
    Returns:
        book_data: string of complete book text
        unique_chars: list of unique characters
        char_to_ind: dict mapping character -> index
        ind_to_char: dict mapping index -> character
    """

    book_fname = book_dir + 'goblet_book.txt'
    fid = open(book_fname, "r")
    book_data = fid.read()
    fid.close()

    unique_chars = list(set(book_data))

    K = len(unique_chars)

    char_to_ind = {}
    ind_to_char = {}

    for i in range(K):
        c = unique_chars[i]
        char_to_ind[c] = i
        ind_to_char[i] = c
    
    return book_data, unique_chars, char_to_ind, ind_to_char 



def OneHotToStr(X, ind_to_char):
    """
    Convert one-hot encoded char index sequence to string.

    Args:
        X: sequence as one-hot encoded char indices (K, n)
        ind_to_char: dict mapping index -> character
    Returns: 
        text: X converted into string
    """
    n = X.shape[1]

    text = ""

    for i in range(n):
        x = X[:, i]
        idx = np.argmax(x)
        c = ind_to_char[idx]
        text = text + c
    
    return text



def StrToOneHot(text, char_to_ind, K):
    """
    Convert string to one-hot encoded char index sequence.

    Args:
        text: string of character sequence
        char_to_ind: dict mapping character -> index
        K: num of unique characters
    Returns: 
        X: text as one-hot encoded char indices (K, n)
    """
    n =  len(text)

    X = np.zeros((K, n))

    for i in range(n):
        c = text[i]
        idx = char_to_ind[c]
        X[idx, i] = 1
    
    return X


def InitAdam(RNN):
    """
    Initializes Adam variables.

    Args:
        RNN: dict of RNN params
    Returns:
        m: mean estimate
        v: variance estimate
    """

    m = {}
    v = {}

    for kk in RNN.keys():
        m[kk] = np.zeros_like(RNN[kk])
        v[kk] = np.zeros_like(RNN[kk])

    return m, v


def AdamStep(RNN, grads, m, v, t, eta, beta1=0.9,beta2=0.999, eps=1e-8):
    """
    Updates RNN using Adam optimizer.

    Args:
        RNN: dict of RNN params
        grads: dict of gradients
        m: mean estimates
        v: variance estimates
        t: update step
        eta: learning rate
        beta1, beta2, eps: adam params
    Returns:
        RNN: updated RNN params
        m: updates mean estimates
        v: updated variance estimates
    """

    for kk in grads.keys():
        m[kk] = beta1*m[kk] + (1-beta1)*grads[kk]
        v[kk] = beta2*v[kk] + (1-beta2)*(grads[kk]**2)

        m_hat = m[kk]/(1-beta1**t)
        v_hat = v[kk]/(1-beta2**t)

        RNN[kk] = RNN[kk] - eta*m_hat/(np.sqrt(v_hat)+eps)
    
    return RNN, m, v





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

    m_adam, v_adam = InitAdam(RNN)

    hprev = np.zeros((m,1))
    smooth_loss = None
    smooth_losses = []
    e = 0

    best_loss = np.inf
    best_RNN = copy.deepcopy(RNN)

    rng = np.random.default_rng(seed)

    synth_file = open("synth_evol.txt", "w")

    for t in range(1, n_updates+1):

        # if finished 1 epoch = 1 run through whole book_data
        if e + seq_length + 1 >= len(book_data):
            e = 0   # reset cursor
            hprev = np.zeros((m,1)) # reset input hidden state
        
        X_chars = book_data[e:e+seq_length]
        Y_chars = book_data[e+1:e+seq_length+1]

        X = StrToOneHot(X_chars, char_to_ind, K)
        Y = StrToOneHot(Y_chars, char_to_ind, K)

        # print and save synthesized text regularly
        if t == 1 or t % 10000 == 0:
            x0 = X[:, 0:1]
            Y_sample = Synthesize(RNN, hprev, x0, 200, rng)
            synth_txt = OneHotToStr(Y_sample, ind_to_char)

            if smooth_loss is None:
                loss_text = "None"
            else:
                loss_text = str(smooth_loss)

            synth_file.write("Before update " + str(t) + "\n")
            synth_file.write("Smooth loss: " + loss_text + "\n")
            synth_file.write(synth_txt + "\n")
            synth_file.write("------------------------------------------\n\n")

            print("synthesized text before update", t, ": ")
            print(synth_txt)
            print("------------------------------------------")


        loss, fp = ForwardPass(X, Y, RNN, hprev)
        grads = BackwardPass(X, Y, RNN, fp)

        RNN, m_adam, v_adam = AdamStep(RNN, grads, m_adam, v_adam, t, eta,)

        hprev = fp['H'][:,-1:]

        if smooth_loss is None:
            smooth_loss = loss
        else:
            smooth_loss = .999* smooth_loss + .001 * loss
        smooth_losses.append(smooth_loss)

        if smooth_loss < best_loss:
            best_loss = smooth_loss
            best_RNN = copy.deepcopy(RNN)

        e += seq_length

        if t % 100 == 0:
            print("update:", t, "smooth loss:", smooth_loss)

    synth_file.close()
    return RNN, best_RNN, smooth_losses, best_loss



def PlotSmoothLoss(smooth_losses, filename):

    plt.figure()
    plt.plot(smooth_losses)
    plt.xlabel("Update step")
    plt.ylabel("Smooth loss")
    plt.savefig('./images/'+filename)
    plt.show()

################### DEBUGGING PART ##############################################

""""
import torch

# assumes X has size d x tau, h0 has size m x 1, etc
def ComputeGradsWithTorch(X, y, h0, RNN):

    tau = X.shape[1]

    Xt = torch.from_numpy(X)
    ht = torch.from_numpy(h0)

    torch_network = {}
    for kk in RNN.keys():
        torch_network[kk] = torch.tensor(RNN[kk], requires_grad=True)


    ## give informative names to these torch classes        
    apply_tanh = torch.nn.Tanh()
    apply_softmax = torch.nn.Softmax(dim=0) 
    
    # create an empty tensor to store the hidden vector at each timestep
    Hs = torch.empty(h0.shape[0], X.shape[1], dtype=torch.float64)
    
    hprev = ht
    for t in range(tau):

        #### BEGIN your code ######

        # Code to apply the RNN to hprev and Xt[:, t:t+1] to compute the hidden scores "Hs" at timestep t
        # (ie equations (1,2) in the assignment instructions)
        # Store results in Hs

        # Don't forget to update hprev!

        a = (torch.matmul(torch_network['W'], hprev) + 
             torch.matmul(torch_network['U'], Xt[:, t:t+1]) + torch_network['b'])
        h = apply_tanh(a)
        Hs[:, t:t+1] = h
        hprev = h
        
        #### END of your code ######            

    Os = torch.matmul(torch_network['V'], Hs) + torch_network['c']        
    P = apply_softmax(Os)    
    
    # compute the loss
    
    loss = torch.mean(-torch.log(P[y, np.arange(tau)]))
    
    # compute the backward pass relative to the loss and the named parameters 
    loss.backward()

    # extract the computed gradients and make them numpy arrays
    grads = {}
    for kk in RNN.keys():
        grads[kk] = torch_network[kk].grad.numpy()

    return grads


book_data, unique_chars, char_to_ind, ind_to_char = LoadBookData()

K = len(unique_chars)
m = 10
eta = 0.001
seq_length = 25

RNN = InitializeRNN(K, m)

X_chars = book_data[0:seq_length]
Y_chars = book_data[1:seq_length+1]

X = StrToOneHot(X_chars, char_to_ind, K)
Y = StrToOneHot(Y_chars, char_to_ind, K)

h0 = np.zeros((m,1))

loss, fp = ForwardPass(X, Y, RNN, h0)
grads = BackwardPass(X, Y, RNN, fp)

# Y = one hot labels ==> y = integer labels
y = np.argmax(Y, axis=0)

torch_grads = ComputeGradsWithTorch(X, y, h0, RNN)

for kk in grads.keys():
    diff = np.max(np.abs(grads[kk] - torch_grads[kk]))
    print(kk)
    print("max abs diff:", diff)

"""

############ END OF DEBUGGING PART ###################################

# ---------- Exercise 0.1: Read in the data ---------------------------------

book_data, unique_chars, char_to_ind, ind_to_char = LoadBookData()


# ---------- Exercise 0.2:  Hyper-parameters & Initilization ----------------

K = len(unique_chars)
m = 100
eta = 0.001
seq_length = 25

RNN = InitializeRNN(K, m)

# --------------- Exercise 0.3:  Synthesize text ------------------------
"""
rng = np.random.default_rng(42)

n = 200

h0 = np.zeros((m, 1))
x0 = np.zeros((K, 1))
x0[char_to_ind['.']] = 1

Y = Synthesize(RNN, h0, x0, n, rng)

print("Generated text: ")
print(OneHotToStr(Y, ind_to_char))
print("------------------")
"""

# --------------- Exercise 0.4: Forward & backward pass ------------------------

"""
X_chars = book_data[0:seq_length]
Y_chars = book_data[1:seq_length+1]

X = StrToOneHot(X_chars, char_to_ind, K)
Y = StrToOneHot(Y_chars, char_to_ind, K)

h0 = np.zeros((m,1))

loss, fp = ForwardPass(X, Y, RNN, h0)
grads = BackwardPass(X, Y, RNN, fp)

print("loss: ", loss)
"""

# --------------- Exercise 0.5: Train ------------------------

n_updates = 100000

updates_per_epoch = (len(book_data) - 1) // seq_length
epochs = n_updates / updates_per_epoch

print("updates per epoch:", updates_per_epoch)
print("epochs:", epochs)

RNN, best_RNN, smooth_losses, best_loss = TrainRNN(
    book_data,
    char_to_ind,
    ind_to_char,
    RNN,
    eta,
    seq_length,
    n_updates)


PlotSmoothLoss(smooth_losses, 'smooth_loss.png')

# --------------- Synthesize text from best model -----------------
 
rng = np.random.default_rng(seed=42)

h0 = np.zeros((m,1))

x0 = np.zeros((K,1))
x0[char_to_ind['.']] = 1

Y = Synthesize(best_RNN, h0, x0, 1000, rng)
best_text = OneHotToStr(Y, ind_to_char)

# print synthesized text
print("synthesized text from best model: ")
print(best_text)
print("------------------------------------------")
print("Best smooth loss:", best_loss)

# save synthesized text to file
synth_file = open("best_synth.txt", "w")
synth_file.write("Best smooth loss: " + str(best_loss) + "\n\n")
synth_file.write(best_text)
synth_file.close()
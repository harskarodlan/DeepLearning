import numpy as np


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


def AdamStep(RNN, grads, m, v, t, eta, beta1=0.1,beta2=0.999, eps=1e-8):
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



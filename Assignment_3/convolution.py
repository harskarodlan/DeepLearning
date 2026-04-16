import numpy as np


def SlowConv(X_ims, Fs):
    """
    Performs slow convolution (for-loops).

    Args:
        X_ims: image data (32, 32, 3, n) = (32, 32, 3, 5)
        Fs: filters (f, f, 3, nf) = (4, 4, 3, 2)
    Returns:
        conv_out: convolution result 
                  (32/f, 32/f, nf, n) = (8, 8, 2, 5)
    """

    h, w, _, n = X_ims.shape
    f, _, _, nf = Fs.shape

    out_h = h // f
    out_w = w // f
    conv_out = np.zeros((out_h, out_w, nf, n), dtype=X_ims.dtype)

    # for each image
    for i in range(n):
        # for each filter
        for k in range(nf):
            # for each patch row
            for r in range(out_h):
                # for each patch column
                for c in range(out_w):
                    # idx of beginning of patch
                    r0 = r * f
                    c0 = c * f

                    patch = X_ims[r0: r0 + f, c0: c0+f, :, i]

                    # apply k:th filter to patch
                    conv_out[r, c, k, i] = np.sum(patch * Fs[:, :, :, k])
    return conv_out


def BuildMX(X_ims, f):
    """
    Builds matrix MX for convolution of image data.

    Args:
        X_ims: image data (32, 32, 3, n)
        f: filter width
    Returns:
        MX: (n_p, f*f*3, n)
            n_p = 64
    """

    h, w, _, n = X_ims.shape
    out_h = h // f
    out_w = w // f
    n_p = out_h*out_w # num of patches

    # allocate space for MX
    MX = np.zeros((n_p, f*f*3, n), dtype=X_ims.dtype)

    # for each image
    for i in range(n):
        # for each patch, flatten it into a row in MX
        l = 0   # row num in MX
        for r in range(out_h): # patch rows
            for c in range(out_w): # patch columns
                r0 = r*f
                c0 = c*f
                X_patch = X_ims[r0:r0+f, c0:c0+f,:,i]

                MX[l, :, i] = X_patch.reshape((1, f*f*3), order='C')
                
                l += 1

    return MX



def FlattenFilters(Fs):
    """
    Args:
        Fs: unflattened filters (f, f, 3, nf)
    Returns:
        Fs_flat: flattened filters (f*f*3, nf)
    """
    f, _, _, nf = Fs.shape
    Fs_flat = Fs.reshape((f*f*3, nf), order='C')    
    return Fs_flat



def Conv(MX, Fs_flat):
    """
    Performs fast (matrix) convolution.

    Args:
        MX: (n_p, f*f*3, n)
        Fs_flat: filters flattened (f*f*3, nf)
    Returns:
        conv_outputs_mat: convolution result
                          (n_p, nf, n)
    """
    conv_outputs_mat = np.einsum('ijn, jl ->iln', MX, Fs_flat, optimize=True)
    
    return conv_outputs_mat

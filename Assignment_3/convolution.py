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
    conv_out = np.zeros((out_h, out_w, nf, n), dtype=np.float64)

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
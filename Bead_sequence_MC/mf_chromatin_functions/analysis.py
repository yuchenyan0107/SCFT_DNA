import numpy as np

def diagonal_decay_profile(A, agg="mean", normalize=True):
    """
    A: square matrix (N x N), ideally symmetric
    agg: "mean", "median", or a callable reducing 1D arrays -> scalar
    normalize: if True, divide by the k=0 value
    Returns:
        profile: length-N array, average intensity vs. distance k from diagonal
        horiz:   length-N array, average of super-diagonals  (offset +k)
        vert:    length-N array, average of sub-diagonals    (offset -k)
    """
    A = np.asarray(A)
    assert A.ndim == 2 and A.shape[0] == A.shape[1], "A must be square"
    N = A.shape[0]

    if agg == "mean":
        reducer = np.nanmean
    elif agg == "median":
        reducer = np.nanmedian
    elif callable(agg):
        reducer = agg
    else:
        raise ValueError("agg must be 'mean', 'median', or a callable")

    # horizontal: values k to the right of each diagonal element → super-diagonals
    horiz = np.array([reducer(np.diagonal(A, offset=k)) for k in range(N)])
    # vertical: values k below each diagonal element → sub-diagonals
    vert  = np.array([reducer(np.diagonal(A, offset=-k)) for k in range(N)])

    # Combine horizontal+vertical for each k (k=0 is just the main diagonal once)
    profile = horiz.copy()
    # For k>0, average super and sub diagonals together
    profile[1:] = 0.5 * (horiz[1:] + vert[1:])

    if normalize and profile[0] != 0:
        norm = profile / profile[0]
    else:
        norm = profile

    return norm, horiz, vert
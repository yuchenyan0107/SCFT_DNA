import numpy as np

use_cupy = True
if use_cupy:

    try:
        import cupy as cp
        xp = cp
        print("Using cupy")
    except ImportError:
        cp = None
        xp = np
        print("Using numpy")
else:
    cp = None
    xp = np
    print("Using numpy")

def to_numpy(arr):
    if cp is not None and hasattr(cp, "asnumpy") and isinstance(arr, cp.ndarray):
        return cp.asnumpy(arr)
    return np.asarray(arr)

DTYPE = xp.float32
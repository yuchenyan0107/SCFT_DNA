import numpy as np
from numba import njit, prange

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
'''
from scipy.stats import pearsonr

def pearson_corr_upper(A, B, exclude_diagonal=True):
    """
    Pearson correlation between two symmetric matrices using
    only the upper triangle.
    """
    A = np.asarray(A, dtype=float)
    B = np.asarray(B, dtype=float)
    assert A.shape == B.shape, "A and B must have the same shape"
    assert A.ndim == 2 and A.shape[0] == A.shape[1], "A and B must be square"

    k = 1 if exclude_diagonal else 0
    iu, ju = np.triu_indices_from(A, k=k)

    x = A[iu, ju]
    y = B[iu, ju]

    mask = ~np.isnan(x) & ~np.isnan(y)
    x = x[mask]
    y = y[mask]

    r, p = pearsonr(x, y)
    return r, p

def distance_corrected_residuals(M):
    """
    Subtract mean per genomic distance (per diagonal |i-j|) from a square matrix M.
    """
    n = M.shape[0]
    assert M.shape[0] == M.shape[1], "M must be square"

    R = M.copy()

    for d in prange(1, n):  # usually skip the main diagonal (d=0)
        i = np.arange(n - d)
        j = i + d

        vals = R[i, j]
        m = np.nanmean(vals)

        # subtract from upper and lower diagonals to keep symmetry
        R[i, j] -= m
        R[j, i] -= m

    return R

def pearson_corr_distance_corrected(A, B):
    """
    Distance-corrected Pearson correlation between A and B.
    """
    A_res = distance_corrected_residuals(A)
    B_res = distance_corrected_residuals(B)
    r, p = pearson_corr_upper(A_res, B_res, exclude_diagonal=True)
    return r, p
'''

# faster version:

import numpy as np
from scipy.stats import t as t_dist

# ----------------------------
# Small helpers (p-value, etc.)
# ----------------------------
def _p_from_r(r: float, n: int) -> float:
    """Two-sided p-value for Pearson r with n samples."""
    if n < 3 or (not np.isfinite(r)):
        return np.nan
    if r >= 1.0:
        return 0.0
    if r <= -1.0:
        return 0.0
    tt = r * np.sqrt((n - 2) / (1.0 - r * r))
    return 2.0 * t_dist.sf(np.abs(tt), df=n - 2)


# =========================================
# FAST PATH: Numba (recommended)
# =========================================
try:
    from numba import njit, prange
    _HAS_NUMBA = True
except Exception:
    _HAS_NUMBA = False


if _HAS_NUMBA:
    @njit(parallel=True, fastmath=True, cache=True)
    def _diag_means_upper(M: np.ndarray) -> np.ndarray:
        """
        means[d] = nanmean of upper diagonal (i, i+d), for d=1..n-1
        means[0] = 0.0 (unused if excluding diagonal).
        """
        n = M.shape[0]
        means = np.empty(n, dtype=np.float64)
        means[0] = 0.0
        for d in prange(1, n):
            s = 0.0
            c = 0
            for i in range(n - d):
                v = M[i, i + d]
                if not np.isnan(v):
                    s += v
                    c += 1
            means[d] = s / c if c > 0 else np.nan
        return means


    @njit(fastmath=True, cache=True)
    def _pearson_upper_basic_numba(A: np.ndarray, B: np.ndarray, exclude_diagonal: bool) -> tuple:
        """
        Pearson r over upper triangle (optionally excluding diagonal), skipping NaN pairs.
        Returns (r, n_pairs_used).
        """
        n = A.shape[0]
        start_d = 1 if exclude_diagonal else 0

        meanx = 0.0
        meany = 0.0
        Sxx = 0.0
        Syy = 0.0
        Sxy = 0.0
        k = 0

        for d in range(start_d, n):
            for i in range(n - d):
                j = i + d
                x = A[i, j]
                y = B[i, j]
                if np.isnan(x) or np.isnan(y):
                    continue

                k += 1
                dx = x - meanx
                meanx += dx / k
                dy = y - meany
                meany += dy / k

                Sxx += dx * (x - meanx)
                Syy += dy * (y - meany)
                Sxy += dx * (y - meany)

        denom = np.sqrt(Sxx * Syy)
        if k < 2 or denom == 0.0:
            return (np.nan, k)
        return (Sxy / denom, k)


    @njit(fastmath=True, cache=True)
    def _pearson_upper_distance_corrected_numba(
        A: np.ndarray, B: np.ndarray, meanA: np.ndarray, meanB: np.ndarray
    ) -> tuple:
        """
        Pearson r over upper triangle excluding diagonal, using distance-corrected residuals:
            x = A[i,j] - meanA[d], y = B[i,j] - meanB[d], d=j-i
        Skips NaN pairs.
        Returns (r, n_pairs_used).
        """
        n = A.shape[0]

        meanx = 0.0
        meany = 0.0
        Sxx = 0.0
        Syy = 0.0
        Sxy = 0.0
        k = 0

        for d in range(1, n):
            ma = meanA[d]
            mb = meanB[d]
            # If a diagonal has no finite values, ma/mb may be NaN; skip cheaply.
            if np.isnan(ma) or np.isnan(mb):
                continue

            for i in range(n - d):
                j = i + d
                x0 = A[i, j]
                y0 = B[i, j]
                if np.isnan(x0) or np.isnan(y0):
                    continue

                x = x0 - ma
                y = y0 - mb

                k += 1
                dx = x - meanx
                meanx += dx / k
                dy = y - meany
                meany += dy / k

                Sxx += dx * (x - meanx)
                Syy += dy * (y - meany)
                Sxy += dx * (y - meany)

        denom = np.sqrt(Sxx * Syy)
        if k < 2 or denom == 0.0:
            return (np.nan, k)
        return (Sxy / denom, k)


def pearson_corr_upper_fast(A, B, exclude_diagonal=True, return_p=True):
    """
    Faster Pearson correlation on upper triangle (NaN-safe).
    Uses Numba if available; otherwise falls back to NumPy.
    """
    A = np.asarray(A, dtype=np.float64)
    B = np.asarray(B, dtype=np.float64)
    if A.shape != B.shape or A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError("A and B must be square matrices of the same shape.")

    if _HAS_NUMBA:
        r, n_used = _pearson_upper_basic_numba(A, B, exclude_diagonal)
    else:
        # NumPy fallback (still faster than pearsonr in many cases)
        k = 1 if exclude_diagonal else 0
        iu, ju = np.triu_indices(A.shape[0], k=k)
        x = A[iu, ju]
        y = B[iu, ju]
        m = (~np.isnan(x)) & (~np.isnan(y))
        x = x[m]
        y = y[m]
        n_used = x.size
        if n_used < 2:
            r = np.nan
        else:
            x = x - x.mean()
            y = y - y.mean()
            denom = np.sqrt(np.dot(x, x) * np.dot(y, y))
            r = (np.dot(x, y) / denom) if denom != 0 else np.nan

    if not return_p:
        return r
    return r, _p_from_r(r, int(n_used))


def distance_corrected_residuals_fast(M):
    """
    Faster distance correction using diagonal *views* (NumPy).
    This is much faster than indexing with np.arange for each diagonal.
    """
    M = np.asarray(M, dtype=np.float64)
    if M.ndim != 2 or M.shape[0] != M.shape[1]:
        raise ValueError("M must be square.")

    R = M.copy()
    n = R.shape[0]
    for d in range(1, n):
        diag_u = R.diagonal(d)     # view (upper diagonal)
        m = np.nanmean(diag_u)
        diag_u -= m
        R.diagonal(-d)[:] -= m     # view (lower diagonal)
    return R


def pearson_corr_distance_corrected(A, B, return_p=True):
    """
    Fast distance-corrected Pearson correlation:
      1) compute per-distance means on upper diagonals
      2) compute Pearson r on residuals WITHOUT building residual matrices (Numba path)
    """
    A = np.asarray(A, dtype=np.float64)
    B = np.asarray(B, dtype=np.float64)
    if A.shape != B.shape or A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError("A and B must be square matrices of the same shape.")

    if _HAS_NUMBA:
        meanA = _diag_means_upper(A)
        meanB = _diag_means_upper(B)
        r, n_used = _pearson_upper_distance_corrected_numba(A, B, meanA, meanB)
        if not return_p:
            return r
        return r, _p_from_r(r, int(n_used))

    # Fallback: build residuals using the faster diagonal-view method
    A_res = distance_corrected_residuals_fast(A)
    B_res = distance_corrected_residuals_fast(B)
    return pearson_corr_upper_fast(A_res, B_res, exclude_diagonal=True, return_p=return_p)


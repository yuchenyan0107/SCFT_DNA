import numpy as np
from typing import Tuple, Literal, Optional

from typing import Dict
from scipy.ndimage import gaussian_filter  # same as original
from scipy import ndimage as ndi

def distance_index(n: int) -> np.ndarray:
    """Return an n×n array with |i-j| (fine-grained genomic distance in bins)."""
    idx = np.arange(n, dtype=np.int32)
    return np.abs(idx[:, None] - idx[None, :])

def distance_profiles(
    M: np.ndarray,
    low_pp: float,
    high_pp: float
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    For each diagonal k = 0..N-1 (distance k), compute:
      - mean (ignoring zeros and NaNs),
      - low percentile (low_pp),
      - high percentile (high_pp).

    Mirrors the behavior of funinv.profPlot for mom=['pp', p]:
    it filters out zeros on each diagonal before computing stats.
    """
    assert M.ndim == 2 and M.shape[0] == M.shape[1], "M must be square."
    n = M.shape[0]
    means  = np.zeros(n, dtype=np.float64)
    p_low  = np.zeros(n, dtype=np.float64)
    p_high = np.zeros(n, dtype=np.float64)

    for k in range(n):
        d = np.diagonal(M, offset=k).astype(np.float64, copy=False)
        # Drop zeros (as in the original) and NaNs
        d = d[(d != 0) & np.isfinite(d)]
        if d.size == 0:
            means[k]  = 0.0
            p_low[k]  = 0.0
            p_high[k] = 0.0
        else:
            means[k]  = np.nanmean(d)
            p_low[k]  = np.nanpercentile(d, low_pp)
            p_high[k] = np.nanpercentile(d, high_pp)
    return means, p_low, p_high

def preprocess_contact_matrix(
    C: np.ndarray,
    *,
    sig: Optional[float] = None,                   # Gaussian sigma; None/0 → no filter
    Cnanflag: Literal['nanlow', 'nanhigh', 'zero'] = 'nanlow',
    Clowpp: float = 5.0,
    Chighpp: float = 95.0,
    cfmax: Optional[float] = None,                 # e.g., 0.10 caps at 10% of max; None → no cap
    normalize: bool = True,
    zero_diagonal: bool = True,
    return_info: bool = True
) -> Tuple[np.ndarray, Dict]:
    """
    Pre-process the experimental contact matrix to mirror the original pipeline:

    1) Optional Gaussian filter (if sig>0).
       - If filtering created NaNs at positions that were finite before, restore originals.
    2) Identify beads with entire-NaN rows/cols; *do not* fill NaNs touching those beads.
    3) Fill remaining NaNs distance-wise:
       - 'nanlow'  → use high percentile of CCf at that distance (mom=['pp', Clowpp] → upper band in original)
       - 'nanhigh' → use low  percentile of CCf at that distance (mom=['pp', Chighpp] → lower band in original)
       - 'zero'    → set to 0
       Any leftover NaNs (in valid bead pairs) are set to 0.
    4) Saturate very large values: X > cfmax*max → cfmax*max   (if cfmax is given)
    5) Normalize by max (if normalize = True)
    6) Zero main diagonal (if zero_diagonal = True)

    Returns the processed matrix and an info dict with masks and profiles.
    """
    if C.ndim != 2 or C.shape[0] != C.shape[1]:
        raise ValueError("C must be a square 2D array.")

    C = C.astype(np.float64, copy=True)  # work in float64 for stable stats
    n = C.shape[0]

    # 1) Gaussian filter (optional)
    if sig is not None and sig != 0:
        Cf = gaussian_filter(C, sigma=float(sig))
        # If filtered created NaNs where original had numbers, restore originals (parity with original code)
        mask_new_nan = np.isnan(Cf) & np.isfinite(C)
        if mask_new_nan.any():
            Cf[mask_new_nan] = C[mask_new_nan]
    else:
        Cf = C.copy()

    # 2) Beads with entire-NaN rows/cols
    nan_row = np.isnan(C).all(axis=1)  # original checks rows of the *original* CC
    # mask that allows filling only when BOTH beads are not entirely-NaN rows/cols
    valid_pair_mask = ~(nan_row[:, None] | nan_row[None, :])

    # 3) Fill isolated NaNs distance-wise
    sijN = distance_index(n)
    means, p_low, p_high = distance_profiles(Cf, low_pp=Clowpp, high_pp=Chighpp)

    fill_targets = np.isnan(Cf) & valid_pair_mask
    if fill_targets.any():
        if Cnanflag == 'nanlow':
            # original uses mom=['pp', Clowpp] and picks the *upper* band for fill
            fill_vals = p_high[sijN[fill_targets]]
            Cf[fill_targets] = fill_vals
        elif Cnanflag == 'nanhigh':
            # original uses mom=['pp', Chighpp] and picks the *lower* band for fill
            fill_vals = p_low[sijN[fill_targets]]
            Cf[fill_targets] = fill_vals
        else:  # 'zero'
            Cf[fill_targets] = 0.0

    # Any residual NaNs in valid pairs → zero
    Cf = np.where(np.isnan(Cf) & valid_pair_mask, 0.0, Cf)

    # 4) Cap very large values at cfmax * max
    Cd = Cf.copy()
    if cfmax is not None and np.isfinite(cfmax) and cfmax > 0:
        m = np.nanmax(Cd)
        if m > 0:
            upper = cfmax * m
            Cd = np.minimum(Cd, upper)

    # 5) Normalize by max
    if normalize:
        m = np.nanmax(Cd)
        if m > 0:
            Cd = Cd / m

    # 6) Zero main diagonal
    if zero_diagonal:
        np.fill_diagonal(Cd, 0.0)

    info = {
        "valid_beads": ~nan_row,                # True where bead row/col is not entirely NaN
        "valid_pair_mask": valid_pair_mask,     # where distance-based filling was allowed
        "distance_profiles": {
            "mean": means,
            "p_low": p_low,
            "p_high": p_high,
        },
        "params": {
            "sig": sig,
            "Cnanflag": Cnanflag,
            "Clowpp": Clowpp,
            "Chighpp": Chighpp,
            "cfmax": cfmax,
            "normalize": normalize,
            "zero_diagonal": zero_diagonal,
        }
    }

    return (Cd, info) if return_info else (Cd, {})

def fill_nearest(a: np.ndarray) -> np.ndarray:
    """
    Replace -inf/NaN in an array with the value from the nearest finite cell
    (Euclidean distance). Works for N-D arrays.
    """
    a = np.asarray(a, dtype=float)
    out = a.copy()
    bad = ~np.isfinite(out)  # True where -inf or NaN

    if not bad.any():
        return out
    if bad.all():
        raise ValueError("All values are non-finite; nothing to copy from.")

    # We want the nearest *good* cells. distance_transform_edt returns indices
    # of the nearest zero; zeros occur where `bad` is False (i.e., good).
    inds = ndi.distance_transform_edt(bad, return_distances=False, return_indices=True)

    # Gather coordinates of nearest good cell for every bad location
    coords = tuple(inds[d][bad] for d in range(out.ndim))
    out[bad] = out[coords]
    return out
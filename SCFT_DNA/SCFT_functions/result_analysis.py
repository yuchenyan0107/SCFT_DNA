import pandas as pd
import cooler
from hicrep import hicrepSCC
import numpy as np

def dense_to_cool(path, M, *, chrom="chr1", bin_size=1, upper_triangle_only=True):
    """
    Write a dense NxN matrix to a .cool file.
    bin_size sets coordinate units; if you don't have bp, set bin_size=1 and treat dBPMax in bins.
    """
    M = np.asarray(M)
    n = M.shape[0]
    if M.shape != (n, n):
        raise ValueError("Matrix must be square")

    bins = pd.DataFrame({
        "chrom": [chrom] * n,
        "start": np.arange(n) * bin_size,
        "end": (np.arange(n) + 1) * bin_size
    })

    if upper_triangle_only:
        i, j = np.triu_indices(n, k=0)
    else:
        i, j = np.indices((n, n))
        i, j = i.ravel(), j.ravel()

    counts = M[i, j].ravel()
    nz = counts != 0
    pixels = pd.DataFrame({
        "bin1_id": i[nz],
        "bin2_id": j[nz],
        "count": counts[nz].astype(np.int64, copy=False)
    })

    cooler.create_cooler(path, bins=bins, pixels=pixels, ordered=True)

def hicrep_scc_from_dense(A, B, *, h=1, dBPMax=200, bin_size=1):
    """
    Compute HiCRep SCC using official hicrep package by writing temporary coolers.
    dBPMax is in the same units as bin_size; if bin_size=1, dBPMax is in bins.
    """
    dense_to_cool("A_tmp.cool", A, bin_size=bin_size)
    dense_to_cool("B_tmp.cool", B, bin_size=bin_size)

    coolA = cooler.Cooler("A_tmp.cool")
    coolB = cooler.Cooler("B_tmp.cool")

    scc_per_chrom = hicrepSCC(coolA, coolB, h, dBPMax, False)
    # single chrom -> single value in array-like output
    return float(np.asarray(scc_per_chrom).ravel()[0])
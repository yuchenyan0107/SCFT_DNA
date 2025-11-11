import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def zoom_nearest(y: np.ndarray, new_len: int) -> np.ndarray:
    """
    Resample a 1D array `y` to `new_len` points using nearest‐neighbor interpolation.

    Parameters
    ----------
    y : np.ndarray
        Original 1D data array.
    new_len : int
        Desired length of output array.

    Returns
    -------
    np.ndarray
        Resampled array of length `new_len`.
    """
    old_len = y.shape[0]
    if new_len <= 0:
        raise ValueError("new_len must be positive")
    # generate new sample positions in [0, old_len-1]
    new_positions = np.linspace(0, old_len - 1, new_len)
    # pick nearest integer index for each
    nearest_idx = np.round(new_positions).astype(int)
    # ensure within bounds
    nearest_idx = np.clip(nearest_idx, 0, old_len - 1)
    return y[nearest_idx]

def IMR90_data(ns):
    df = pd.read_csv('IMR90/polymer_IMR90.bed', sep='\t', header=None, names=['pos', 'state'])

    # Identify unique states and sort them
    states = sorted(df['state'].unique())

    # Compute contiguous runs for each state
    state_runs = {}
    for state in states:
        runs = []
        in_run = False
        start = None
        for idx, s in enumerate(df['state']):
            if s == state and not in_run:
                in_run = True
                start = idx
            elif s != state and in_run:
                runs.append((start, idx - start))
                in_run = False
        if in_run:
            runs.append((start, len(df) - start))
        state_runs[state] = runs

    # Prepare the figure
    states = sorted(df['state'].unique())  # e.g. ['A','B',…,'H']
    n_states = len(states)
    n_positions = len(df)
    arr = np.zeros((n_states, n_positions), dtype=int)
    for i, st in enumerate(states):
        arr[i, df['state'] == st] = 1

    chain_interaction_binary = []
    for i in range(arr.shape[0]):
        zoomed_arr = zoom_nearest(arr[i, :], ns + 1)
        chain_interaction_binary.append(zoomed_arr)

    # --------------------------------------------
    chain_interaction_binary.append(zoom_nearest(arr[0, :], ns + 1))
    chain_interaction_binary[0][-ns//2:-1] = 0
    chain_interaction_binary[-1][0:ns//2] = 0
    # --------------------------------------------

    chain_interaction_binary = np.array(chain_interaction_binary)

    n_classes, n_indices = chain_interaction_binary.shape[0], chain_interaction_binary.shape[1]
    y = chain_interaction_binary  # replace with your array
    # --------------------------------------------

    fig, ax = plt.subplots(figsize=(10, 9))

    # vertical gap between successive curves.
    # pick something that looks good for your data range.
    offset = (y.max() - y.min()) * 1.1

    for i in range(n_classes):
        baseline = i * offset
        ax.plot(np.arange(n_indices),  # x-axis (indices)
                y[i] + baseline,  # vertically shifted data
                linewidth=1.5)

        # thin horizontal “baseline” for each class
        ax.axhline(baseline, linewidth=.6, alpha=.4)

    # cosmetic touches ------------------------------------------------------------
    ax.set_yticks([i * offset for i in range(n_classes)])
    ax.set_yticklabels([chr(65 + i) for i in range(n_classes)])  # A, B, C, …
    ax.set_xlabel("Index")
    ax.set_title("Classes of binding sites on DNA Across Indices", pad=15)
    ax.margins(x=0)  # no extra white space left/right
    plt.tight_layout()
    plt.show()

    return chain_interaction_binary
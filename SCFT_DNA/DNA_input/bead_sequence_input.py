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

def IMR90_data(ns, split_class_zero = True, plotting_style_bead = True, clip = None):
    df = pd.read_csv('inferred_sequence/polymer_IMR90.bed', sep='\t', header=None, names=['pos', 'state'])

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
    if split_class_zero == True:
        chain_interaction_binary.append(zoom_nearest(arr[0, :], ns + 1))
        chain_interaction_binary[0][-ns//2:-1] = 0
        chain_interaction_binary[-1][0:ns//2] = 0
    # --------------------------------------------

    chain_interaction_binary = np.array(chain_interaction_binary)
    print(chain_interaction_binary.shape)

    if clip is not None:
        chain_interaction_binary = chain_interaction_binary[:, clip[0]:clip[1]]

    n_classes, n_indices = chain_interaction_binary.shape[0], chain_interaction_binary.shape[1]
    y = chain_interaction_binary  # replace with your array
    # --------------------------------------------
    if plotting_style_bead:
        bead_sequences = onehot_to_color_labels(chain_interaction_binary)
        visualize_config(bead_sequences, chain_interaction_binary.shape[0])
    else:

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

def visualize_config(config, num_classes):
    """Visualize binding configuration and resulting contact map."""

    fig = plt.figure(figsize=(16, 10), dpi = 300)
    gs = fig.add_gridspec(3, 2, height_ratios=[1, 3, 1], hspace=0.3)

    # 1. Binding site configuration
    ax1 = fig.add_subplot(gs[0, :])
    colors = plt.cm.tab10(np.linspace(0, 1, num_classes))

    # Show subset if too many beads
    display_length = min(1000, len(config))
    config_subset = config[:display_length]

    for class_type in range(num_classes):
        mask = config_subset == class_type
        if np.any(mask):
            ax1.scatter(np.where(mask)[0], np.ones(np.sum(mask)) * class_type,
                        c=[colors[class_type]], s=5, alpha=0.8)

    ax1.set_xlabel('Bead Position')
    ax1.set_ylabel('Binding Site Type')
    # ax1.set_title(f'{title} - Binding Site Configuration (first {display_length} beads)')
    ax1.grid(True, alpha=0.3)

def onehot_to_color_labels(arr):
    arr = np.asarray(arr)
    if arr.ndim != 2:
        raise ValueError("Expected a 2D array of shape (m, nx).")

    m, nx = arr.shape
    col_sum = arr.sum(axis=0)

    labels0 = arr.argmax(axis=0)  # 0..m-1
    # optional extra check: winning entry really is 1
    if not np.all(arr[labels0, np.arange(nx)] == 1):
        raise ValueError("Invalid one-hot encoding detected.")

    return labels0

import numpy as np

def colors_to_onehot(colors, m=None):
    """
    Convert a 1D array of color indices (1..m) to a one-hot matrix of shape (m, nx).

    Parameters
    ----------
    colors : array-like, shape (nx,)
        Color index for each bead (1-based: 1..m).
    m : int, optional
        Number of colors. If None, inferred as colors.max().

    Returns
    -------
    onehot : ndarray, shape (m, nx)
        onehot[i, j] = 1 if colors[j] == i+1 else 0
    """
    colors = np.asarray(colors)

    nx = colors.shape[0]
    if m is None:
        m = int(colors.max())

    onehot = np.zeros((m, nx), dtype=int)
    # convert to 0-based indices for rows
    row_idx = colors - 1
    col_idx = np.arange(nx)
    onehot[row_idx, col_idx] = 1

    return onehot

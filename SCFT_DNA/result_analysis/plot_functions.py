import numpy as np
import matplotlib.pyplot as plt

def _centers_to_edges(c):
    """Convert 1D bin centers to bin edges (length N -> N+1)."""
    c = np.asarray(c)
    dc = np.diff(c)
    edges = np.empty(c.size + 1, dtype=c.dtype)
    edges[1:-1] = (c[:-1] + c[1:]) / 2
    edges[0] = c[0] - dc[0] / 2
    edges[-1] = c[-1] + dc[-1] / 2
    return edges

def plot_heatmap(Z, x, y, *, ax=None, cmap="viridis", xlabel="ChiNpp", ylabel="ChiNps", title=None):
    Z = np.asarray(Z)
    x = np.asarray(x)
    y = np.asarray(y)
    assert Z.shape == (x.size, y.size), f"Expected Z.shape == ({x.size}, {y.size}), got {Z.shape}"

    # pcolormesh wants bin edges
    xe = _centers_to_edges(x)
    ye = _centers_to_edges(y)

    if ax is None:
        fig, ax = plt.subplots()

    m = ax.pcolormesh(xe, ye, Z.T, shading="auto", cmap=cmap)  # transpose to match (x,y)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)
    plt.colorbar(m, ax=ax, label="value")
    return ax


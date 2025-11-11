import numpy as np
import matplotlib.pyplot as plt

def plot_phi_blocks_periodic(phi_blocks,
                             extent=None,
                             percentile=90,
                             base_alpha=0.4,
                             shift_x=None,
                             shift_y=None):
    """
    Visualise the top-density regions of every component in a periodically
    wrapped system.

    Parameters
    ----------
    phi_blocks : ndarray, shape (M, Nx, Ny)
        Density fields from `get_phi()`.
    extent : tuple or None
        (xmin, xmax, ymin, ymax) for imshow if you want physical axes.
    percentile : float
        Voxels below this density percentile are masked out.
    base_alpha : float
        Transparency of each overlay.
    shift_x, shift_y : int or None
        Grid steps to roll the data (+x => left→right, +y => bottom→top).
        If either is None, the code finds the global density maximum and
        chooses a shift that puts that maximum at the image centre.
    """

    # ---------- handle shifts (automatic or user-supplied) ----------
    M, Nx, Ny = phi_blocks.shape
    total = phi_blocks.sum(axis=0)               # total density
    if shift_x is None or shift_y is None:
        iy_max, ix_max = np.unravel_index(total.argmax(), total.shape)
        # move the global max to the centre of the frame
        cx, cy = Nx // 2, Ny // 2
        if shift_x is None:
            shift_x = cx - ix_max
        if shift_y is None:
            shift_y = cy - iy_max

    # roll every component
    rolled = np.roll(phi_blocks, shift=(0, shift_y, shift_x), axis=(0, 1, 2))

    # ---------- plotting ----------
    cmaps = ['Reds', 'Greens', 'Blues', 'Purples', 'Oranges', 'Greys']
    fig, ax = plt.subplots(figsize=(6, 5), dpi = 250)

    for m, field in enumerate(rolled):
        thresh = np.percentile(field, percentile)
        masked = np.ma.masked_less_equal(field, thresh)
        ax.imshow(masked,
                  cmap=cmaps[m % len(cmaps)],
                  alpha=base_alpha,
                  origin='lower',
                  extent=extent)

    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title("Spatial distribution of DNA segments")
    plt.tight_layout()
    plt.show()
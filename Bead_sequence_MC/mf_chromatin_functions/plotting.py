import numpy as np
import matplotlib.pyplot as plt


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

def sort_classes_by_com(arr):
    """
    arr: 1D array-like of ints.
    Returns:
        new_arr: array with class labels permuted
        mapping: dict {old_label -> new_label}
    """
    a = np.asarray(arr)
    idx = np.arange(a.size)

    # All classes except 0
    classes = np.unique(a)
    classes = classes[classes != 0]

    if classes.size == 0:  # only zeros
        return a.copy(), {0: 0}

    # Center of mass for each class (using 0-based positions)
    com = {c: idx[a == c].mean() for c in classes}

    # Classes ordered from left to right by their center of mass
    classes_by_com = sorted(classes, key=lambda c: com[c])

    # Numeric order we want (small index on the left, big on the right)
    target_labels = sorted(classes)

    # Build mapping: leftmost class gets smallest label, etc.
    mapping = {0: 0}
    for old, new in zip(classes_by_com, target_labels):
        mapping[old] = new

    # Apply mapping (using the original array for masking to avoid conflicts)
    out = a.copy()
    for old, new in mapping.items():
        if old != new:
            out[a == old] = new

    return out
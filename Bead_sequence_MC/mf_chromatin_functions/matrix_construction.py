import numpy as np
from scipy import ndimage as ndi

def gaussian_then_stride(a, factor, sigma=None, mode='reflect'):
    # Simple Gaussian low-pass then pick every factor-th pixel.
    # Fast, but not exact local-average like #2 and needs integer factor.
    if np.isscalar(factor):
        fy = fx = int(factor)
    else:
        fy, fx = map(int, factor)
    if sigma is None:
        sigma = 0.5 * max(fy, fx)
    a_blur = ndi.gaussian_filter(a, sigma=sigma, mode=mode)
    return a_blur[::fy, ::fx]


def contstruct_contact_matrix(
        alpha: float,
        alpscal: float,
        polymer_config: np.ndarray,
        coil_profile,
        globule_profile: np.ndarray,
        distance_matrix
):
    # distance_matrix = np.array(distance_index(len(polymer_config)) / alpscal, dtype=np.int32)

    distance_matrix = np.array(distance_matrix / alpscal, dtype=np.int32)

    matrix_same_class = polymer_config[:, None] == polymer_config[None, :]

    contact_matrix = np.zeros((len(polymer_config), len(polymer_config)), dtype=np.float64)

    contact_matrix[matrix_same_class] = alpha * globule_profile[distance_matrix[matrix_same_class]] + (1 - alpha) * \
                                        coil_profile[distance_matrix[matrix_same_class]]
    contact_matrix[~matrix_same_class] = coil_profile[distance_matrix[~matrix_same_class]]

    zero_class_matrix = (polymer_config[:, None] == 0) | (polymer_config[None, :] == 0)

    contact_matrix[zero_class_matrix] = coil_profile[distance_matrix[zero_class_matrix]]

    return contact_matrix

import numpy as np
from scipy.ndimage import zoom

def resize_square(A, out_size, order=1, mode="reflect"):
    """
    Resize a 2D square array to (out_size, out_size) using scipy.ndimage.zoom.
    """
    A = np.asarray(A)
    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError(f"Expected a square 2D array, got {A.shape}")

    z = out_size / A.shape[0]
    B = zoom(A, (z, z), order=order, mode=mode)

    # Force exact shape in case of rounding differences
    return B[:out_size, :out_size]

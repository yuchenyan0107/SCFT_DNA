import numpy as np

def matrix_rmse(matrix_a: np.ndarray, matrix_b: np.ndarray) -> float:

    mse = np.mean((matrix_a - matrix_b) ** 2)
    rmse = np.sqrt(mse)

    return rmse
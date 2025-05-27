import numpy as np

"""
--- LINEAR REGRESSION: SCALAR LEAST-SQUARES Normal Equation
"""


def lin_regressor(x: np.ndarray, y: np.ndarray) -> list:
    # assuming x and y being 1-dim np.arrays of length N (number of training data samples)

    # number of training samples
    N = x.shape[0]

    # normal form (see lecturing slides)
    # A = [N, sum(x); sum(x), sum(x^2)]
    # b = [sum(y); sum(y*x)]
    A = np.array([[N, np.sum(x)], [np.sum(x), np.sum(x ** 2)]])
    b = np.expand_dims(np.array([np.sum(y), np.sum(y * x)]), axis=-1)

    # solve Ax = b
    theta = np.linalg.solve(A, b).flatten()
    print(f'Linear Regressor: theta0 = {theta[0]:.4f}, theta1 = {theta[1]:.4f}')

    return [theta[0], theta[1]]
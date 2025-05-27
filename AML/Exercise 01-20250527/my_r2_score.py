import numpy as np


def r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    # check types first
    if type(y_true) is not np.ndarray:
        raise TypeError(f"A numpy ndarray is expected for y_true, but got {type(y_true)}")

    if type(y_pred) is not np.ndarray:
        raise TypeError(f"A numpy ndarray is expected for y_pred, but got {type(y_pred)}")

    # catch dimensionality errors
    if y_true.shape != y_pred.shape:
        raise ValueError(f"Shape mismatch: y_true has shape {y_true.shape}, but y_pred has shape {y_pred.shape}")

    # finally compute r2 score
    numerator = np.sum((y_true - y_pred) ** 2)
    denominator = np.sum((y_true - np.mean(y_true)) ** 2)

    if denominator == 0:
        return 0.0
    else:
        return 1 - numerator / denominator

import numpy as np

def _sigmoid(z: np.ndarray) -> np.ndarray:
    """
    Returns elementwise sigmoid values.
    """
    return np.where(z >= 0, 1/(1+np.exp(-z)), np.exp(z)/(1+np.exp(z)))

def train_logistic_regression(X: np.ndarray, y: np.ndarray, lr: float = 0.1, steps: int = 1000) -> tuple[np.ndarray, float]:
    """
    Returns the trained weights and bias as (w, b).
    """
    #Write code here
    n = y.size
    w = np.zeros(X.shape[1])
    b = 0

    for step in range(steps):
        p = _sigmoid(np.matmul(X, w) + b)
        L = -(1 / n) * np.sum(y * np.log(p) + (1 - y) * np.log(1 - p))

        error = p - y
        grad_w = (1 / n) * np.matmul(np.transpose(X), error)
        grad_b = (1 / n) * np.sum(error)

        w = w - lr * grad_w
        b = b - lr * grad_b

    return w, b
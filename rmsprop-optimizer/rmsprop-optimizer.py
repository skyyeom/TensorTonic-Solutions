import numpy as np

def rmsprop_step(
    w: list,
    g: list,
    s: list,
    lr: float = 0.001,
    beta: float = 0.9,
    eps: float = 1e-8,
) -> tuple[list, list]:
    """
    Returns (new_w, new_s) with the same shapes as the inputs.
    """
    # Write code here
    w = np.asarray(w, dtype = float)
    g = np.asarray(g, dtype = float)
    s = np.asarray(s, dtype = float)

    s_update = beta*s + (1 - beta)*(g**2)
    w_update = w - (lr/(np.sqrt(s_update + eps)))*g

    s_update = tuple(s_update)
    w_update = tuple(w_update)
    
    return w_update, s_update
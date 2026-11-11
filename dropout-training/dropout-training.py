import numpy as np

def dropout(
    x: list,
    p: float = 0.5,
    rng: np.random.Generator = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Returns (output, dropout_pattern) as NumPy arrays matching the shape of x.
    """
    # Write code here
    x = np.asarray(x)
    mask = rng.random(x.shape)
    is_mask_greater_than_p = mask <= p
    print(is_mask_greater_than_p)
    scaled_element = np.where(is_mask_greater_than_p,0, x/(1-p))
    dropout_pattern = scaled_element/x
    print(scaled_element, dropout_pattern)
    return (scaled_element, dropout_pattern)
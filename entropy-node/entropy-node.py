import numpy as np

def entropy_node(y: list[int]) -> float:
    """
    Returns the Shannon entropy as a Python float.
    """
    # Write code here
    if len(y) == 0:
        return 0.0
        
    _, counts = np.unique(y,return_counts = True)
    p = counts / len(y) 
    return -np.sum(p * np.log2(p))
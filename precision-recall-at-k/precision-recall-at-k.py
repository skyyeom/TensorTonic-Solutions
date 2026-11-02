def precision_recall_at_k(recommended: list, relevant: list, k: int) -> list[float]:
    """
    Returns [precision, recall] as a list of two floats.
    """
    # Write code here
    relevant_set = set(relevant)
    relevantk = sum(item in relevant_set for item in recommended[:k])
    
    precisionk = abs(relevantk)/k
    recallk = abs(relevantk)/len(relevant)

    return [precisionk, recallk]
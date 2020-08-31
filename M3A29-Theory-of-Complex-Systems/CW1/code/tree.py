def tree(prob, n):
    """Function to simulate active sites.
    Input: Probability (p), Number of generations (n).
    Returns: Number of active sites at generation n (active).
    Variable active is the same as sigma in the coursework notes.
    """
    
    active = 1  # We start with just 1 active site at the beginning
    
    i = 1
    while active > 0 and i < n:  # Main loop for each generation
        
        # Generating branching outcomes from probabilities:
        prob_array = np.random.binomial(1, prob, active)
        
        # Counting number of active sites at next generation:
        active = 2*sum(prob_array)
        i += 1
        
    return active
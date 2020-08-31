def avalanche(prob, n):
    """Function to simulate avalanche.
    Input: Probability (p),  Number of generations (n).
    Output: Avalanches: Total number of active sites in the tree (aval).
    """
    active = 1  # We start with just 1 active site at the beginning
    aval = 1 # Avalanche at the start is simply the initial active site.
    
    i = 1  # Counter
    while active > 0 and i < n:  # Main loop for each generation
        
        # Generating branching outcomes from probabilities:
        prob_array = np.random.binomial(1, prob, active)
        
        # Counting number of active sites at next generation:
        active = 2*sum(prob_array)
        
        # Keeping track of total active sites, which is by def. the avalanche.
        aval += active
        i += 1
    
    return aval
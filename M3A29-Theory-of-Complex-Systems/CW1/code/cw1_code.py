""" Tudor Trita Trita
MSci Mathematics Year 3
M3A29 Theory of Complex Systems
Coursework 1: Code

Note: Code written in Python 3.7 and requires numpy and matplotlib  to 
function correctly.

"""

# cd "C:\Users\Tudor Trita\OneDrive - Imperial College London\Documents\Documents Academic\Imperial College London\MSci Mathematics\Year 3 2018-2019\6. M3A29 Theory of Complex Systems\4. Coursework"
import matplotlib.pyplot as plt
import numpy as np


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


if __name__=="__main__":
    """Main program starts here:"""
    
    # Code for part f
    
    #Initial parameters
    n = 10
    N = 2**(n+1) - 1
    t = 3000
    times = range(t)
    parray1 = [0.1] # Starting probability 1
    parray2 = [0.6] # Starting Probability 2

    # Main loop for generating probabilities at each timestep
    for i in range(t-1):
        sigma1 = tree(parray1[i], n)
        sigma2 = tree(parray2[i], n)    
        
        # Using formulas in page 4072 of paper:
        flt1 = parray1[i] + (1 - sigma1)/N  
        flt2 = parray2[i] + (1 - sigma2)/N
        
        parray1.append(flt1)
        parray2.append(flt2)
    
    plt.figure(figsize=(12,8))
    plt.plot(times, parray1, 'b,',
             times, parray2, 'k,')
    plt.xlabel('t')
    plt.ylabel('p(t)')
    plt.legend(('Blue dots: Initial p = 0.1', 'Black dots: Initial p = 0.6')) 
    plt.title('Name: Tudor Trita Trita, CID:01199397 \n Plot of p(t) against t, like in Fig. 2 of paper (Page 4073)')
    plt.grid(True)
    #plt.savefig("partf.png")
    #plt.show()
    
    
    # Code for part g
    
    # For this part, we can use the average to show how it 
    # We can achieve this by taking sigma = (2*p)**n

    #Initial parameters
    narray = [10,12,14,16,18,20,22]
    t = 10000000
    times = []
    gradarray = []
    # Main loop for generating probabilities at each timestep
    
    for k in range(len(narray)):
        n = narray[k]
        N = 2**(n+1) - 1
        parray = [0.1]  # Starting probability 1
        
        for j in range(t-1):
            sigma1 = (2*parray[j])**n
            
            # Using formulas in page 4072 of paper:
            flt1 = parray[j] + (1 - sigma1)/N  
            
            parray.append(flt1)
            if abs(parray[j]-0.5)<0.01:
                break
        gradarray.append((parray[4]-0.1)/5)
        times.append(j)
    
    
    plt.figure(figsize=(12,8))
    plt.plot(narray, times)
    plt.xlabel('n')
    plt.ylabel('t*')
    plt.title('Name: Tudor Trita Trita, CID:01199397 \n Plot of n against t*, value at which p(t*) approx. 0.5')
    plt.grid(True)
    #plt.savefig("partg.png")
    #plt.show()
    
    print(gradarray)
    
    
    # Code for part h

    # S is number of active sites
    
    # Want to simulate process and see the distribution of active sites.
    
    # TO DO: CHANGE PLOTS
    
    narray = [16,20,24,28]
    p = 0.5
    t = 10000000
    colours = ['b,', 'g,', 'y,', 'k,']
    plt.figure(figsize=(10,6))
    
    for i in range(len(narray)):
        dict = {}
        
        for j in range(t):
            
            avl = avalanche(p, narray[i])
            
            if avl in dict.keys():
                dict[avl] += 1
            else:
                dict[avl] = 1
        
        vals = []
        
        for k in dict.keys():
            vals.append(dict[k])
        
        vals = np.asarray(vals)/sum(vals)
    
        plt.loglog(dict.keys(), vals, colours[i])
    
	s = np.linspace(1,75,75)
	plt.loglog(s, s**(-3/2))
    
    
    plt.xlabel('s')
    plt.ylabel('D(s)')
    plt.legend(('Blue dots: n=16', 'Green dots: n=20', 'Yellow dots: n=24', 'Black dots: n=28'))
    plt.title('Name: Tudor Trita Trita, CID:01199397 \n Plot of s against D(s) with tau = 3/2 line.')
    plt.grid(True)
    plt.savefig('parth.png')
    plt.show()
    
    # Finished
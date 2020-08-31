# Code for part g
    
# For this part, I am using the average of sigma.
# We can achieve this by taking sigma = (2*p)**n,
# as shown in the paper that this formula is correct for 
# the average of sigma.

#Initial parameters
narray = [10,12,14,16,18,20,22]
t = 10000000
times = []
gradarray = []
# Main loop for generating probabilities at each timestep

for k in range(len(narray)):
	n = narray[k]
	N = 2**(n+1) - 1
	parray = [0.1]  # Starting probability equal to 0.1
	
	for j in range(t-1):
		sigma1 = (2*parray[j])**n  # Getting average of sigma for each probability.
		
		# Using formulas in page 4072 of paper:
		flt1 = parray[j] + (1 - sigma1)/N  
		
		parray.append(flt1)
		
		# Checking if current probability has reached 0.5, ie. the critical value.
		if abs(parray[j]-0.5)<0.01: 
			break
		
	times.append(j)  # Storing location at which condition above holds.
	gradarray.append((parray[4]-0.1)/5) # Estimating gradient at p(0).
	


plt.figure(figsize=(12,8))
plt.plot(narray, times)
plt.xlabel('n')
plt.ylabel('t*')
plt.title('Name: Tudor Trita Trita, CID:01199397 \n Plot of n against t*, value at which p(t*) approx. 0.5')
plt.grid(True)
plt.show()
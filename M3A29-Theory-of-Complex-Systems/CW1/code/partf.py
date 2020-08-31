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
plt.show()
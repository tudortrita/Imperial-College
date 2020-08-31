# Code for part h

# S is number of active sites

# Want to simulate process and see the distribution of active sites.

narray = [8,10,12,14]
p = 0.5  # Probability fixed at 0.5
t = 10000000  # Max no. of repetions
colours = ['b,', 'g,', 'y,', 'k,']  # Colours for plot later
plt.figure(figsize=(10,6))

for i in range(len(narray)):
	dict = {}
	
	for j in range(t):
		
		avl = avalanche(p, narray[i])
		
		# To see the distribution of the avalanches, I store them in a dictionary:
		if avl in dict.keys():
			dict[avl] += 1
		else:
			dict[avl] = 1
	
	vals = []
	
	for k in dict.keys():
		vals.append(dict[k])
	
	vals = np.asarray(vals)/sum(vals)

	plt.loglog(dict.keys(), vals, colours[i])

s = np.linspace(1,30,30)
plt.loglog(s, s**(-3/2))


plt.xlabel('s')
plt.ylabel('D(s)')
plt.legend(('Blue dots: n=8', 'Green dots: n=10', 'Yellow dots: n=12', 'Black dots: n=14'))
plt.title('Name: Tudor Trita Trita, CID:01199397 \n Plot of s against D(s) with tau = 3/2 line.')
plt.grid(True)
plt.savefig('parth2.png')
plt.show()
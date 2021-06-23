import matplotlib.pyplot as plt

# load file
#individualsFile = "allIndividuals.csv"
individualsFile = "/Users/elia/Downloads/GoogleGraph-N69501-E9168660-population.csv"
influenceValues = []
nodesValues = []

with open(individualsFile, "r") as fp :
	lines = fp.readlines()
	# pop the header
	header = lines.pop(0)
	
	for line in lines :
		tokens = line.rstrip().split(',')
		print(tokens)
		influenceValues.append( float(tokens[0]) )
		nodesValues.append(float(tokens[2]) )
		

fig = plt.figure()
ax = fig.add_subplot(111)
ax.set_title("NSGA-II")
ax.set_xlabel("influence")
ax.set_ylabel("nodes in the seed set")
ax.plot(influenceValues,nodesValues, 'bo')
plt.savefig(individualsFile[:-4] + ".png")
plt.show()
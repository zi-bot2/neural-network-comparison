"""DQNN Generalisation"""

from dqnn_functions import *

# Data specifications
rangeSizeQuantumData = list(range(2, 21))
numQubits = 1
qnnArch = [numQubits, 2, numQubits]

# Training parameters
learningRate = 0.1
numEpochs = 1000
sizeTestData = 10

pointsX = list(range(1,5))
for n in pointsX:
    subsetNetwork121 = randomNetwork(qnnArch, n) # qnnArch, numTrainingPairs

start = time() #Optional


pointsAverageCost = [subsetTrainingAvg(subsetNetwork121[0], 
                                       subsetNetwork121[1], 
                                       subsetNetwork121[2], 
                                       1.5, 0.1, 1000, 20, 
                                       n, alertIt=20) for n in pointsX]
# qnnArch, initialUnitaries, trainingData, lda, ep, trainingRounds, iterations, n, alertIt=0

print(time() - start) #Optional

plt.plot(pointsX, pointsAverageCost, 'bo')
plt.show()
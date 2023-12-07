"""# DQNN Generalisation"""

from dqnn_functions import *
import pandas as pd

"""## Kerstin's version"""

# subsetNetwork22 = randomNetwork([1,2,1], 10)
  
# start = time() #Optional

# pointsX = list(range(1,5))
# # pointsBoundRand = [boundRand(4, 10, n) for n in pointsX]
# pointsAverageCost = [subsetTrainingAvg(subsetNetwork22[0], subsetNetwork22[1], subsetNetwork22[2], 1.5, 0.1, 1000, 20, n, alertIt=20) for n in pointsX]

# print(time() - start) #Optional

# # plt.plot(pointsX, pointsBoundRand, 'co')
# plt.plot(pointsX, pointsAverageCost, 'bo')
# plt.show()


"""## My version"""

# Data specifications
rangeSizeQuantumData = list(range(2, 21))
numQubits = 1
qnnArch = [numQubits, 2, numQubits]

# Training parameters
epsilon = 0.1
numEpochs = 10
sizeTestData = 10

# Making DataFrame to store values
train_dict = {}
test_dict = {}
numTrials = 5

# for sizeQuantumData in rangeSizeQuantumData:
#   train_dict[f'{sizeQuantumData}'] = []
#   test_dict[f'{sizeQuantumData}'] = []

# for sizeQuantumData in rangeSizeQuantumData:
#   fidelities = []
#   testFidelities = []

#   for i in range(numTrials):
#     dqnn = randomNetwork([1,2,1], sizeQuantumData + sizeTestData)
#     fidelity, testFidelity = trainOnSubset(dqnn[0], dqnn[1], dqnn[2], sizeQuantumData, sizeTestData, 1.5, learningRate, numEpochs)
#     fidelities.append(fidelity)
#     testFidelities.append(testFidelity)
#     print(f'Trial ({sizeQuantumData}, {i}) done.')

#   train_dict[f'{sizeQuantumData}'] = fidelities
#   test_dict[f'{sizeQuantumData}'] = testFidelities

# train_df = pd.DataFrame(train_dict)
# test_df = pd.DataFrame(test_dict)

# train_df.head()

# import os  
# train_df.to_csv(f'/home/zchua/thesis_code/dqnn_train_df.csv')
# test_df.to_csv(f'/home/zchua/thesis_code/dqnn_test_df.csv')


sizeQuantumData = 2
dqnn = randomNetwork(qnnArch, sizeQuantumData + sizeTestData)
# trainOnSubset(dqnn[0], dqnn[1], dqnn[2], sizeQuantumData, sizeTestData, 1.5, learningRate, numEpochs)

plotlist, testPlotlist, currentUnitaries = qnnTrainingTesting(dqnn[0], dqnn[1], dqnn[2], sizeQuantumData, sizeTestData, 1, epsilon, numEpochs)

plotlist[1]
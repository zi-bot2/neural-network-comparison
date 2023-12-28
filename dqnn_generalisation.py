"""# DQNN Generalisation"""

from dqnn_functions import *
import pandas as pd

"""## My version"""

# Data specifications
rangeSizeTrainingData = list(range(1, 21))
qnnArch = [1, 2, 1]

# Training parameters
lambda_ = 1
epsilon = 0.1
numEpochs = 1000
sizeTestData = 10

# Making DataFrame to store values
train_dict = {}
test_dict = {}
numTrials = 30

for sizeQuantumData in rangeSizeTrainingData:
  train_dict[f'{sizeQuantumData}'] = []
  test_dict[f'{sizeQuantumData}'] = []

for sizeQuantumData in rangeSizeTrainingData:
  fidelities = []
  testFidelities = []

  for i in range(numTrials):
    dqnn = randomNetwork([1,2,1], sizeQuantumData + sizeTestData)
    fidelity, testFidelity = trainOnSubset(dqnn[0], dqnn[1], dqnn[2], sizeQuantumData, sizeTestData, lambda_, epsilon, numEpochs)
    fidelities.append(fidelity)
    testFidelities.append(testFidelity)
    print(f'Trial ({sizeQuantumData}, {i}) done.')

  train_dict[f'{sizeQuantumData}'] = fidelities
  test_dict[f'{sizeQuantumData}'] = testFidelities

train_df = pd.DataFrame(train_dict) 
test_df = pd.DataFrame(test_dict)

# train_df.head()

import os  
train_df.to_csv(f'/home/zchua/thesis_code/csvs/dqnn_train_df_12_2023.csv')
test_df.to_csv(f'/home/zchua/thesis_code/csvs/dqnn_test_df_12_2023.csv')

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
"""# DQNN Generalisation"""

from dqnn_functions import *
import pandas as pd
import os

"""## My version"""

# Data specifications
rangeSizeTrainingData = list(range(1, 21))
qnnArch = [3, 4, 3]

# Training parameters
lambda_ = 1
epsilon = 0.05
numEpochs = 1000
sizeTestData = 10

# Generalisation specs
numTrials = 2

directory = '/home/zchua/thesis_code/csvs/dqnn/343/lr_pointzerofive'
os.mkdir(directory)
directory = '/home/zchua/thesis_code/csvs/dqnn/343/lr_pointzerofive/test'
os.mkdir(directory)
# directory = '/home/zchua/thesis_code/csvs/dqnn/23432/lr_pointzerothree'
# os.mkdir(directory)

make_dqnn_generalisation_csvs(qnnArch, rangeSizeTrainingData, 
                                  lambda_, epsilon, numEpochs, 
                                  sizeTestData, numTrials, directory)


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
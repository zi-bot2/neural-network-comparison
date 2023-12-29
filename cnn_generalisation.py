"""# CNN Generalisation"""

from cnn_functions import *

"""We want to plot mean/media value of the fidelity vs sizeQuantumData

0. Instantiate neural network
1. Generate N quantum data training pairs
2. Train the CNN on those pairs
3. Keep final training and testing fidelity values after 1000 epochs
4. Do Steps 0-3 30 times
6. Plot the training fidelity median or mean and the test fidelity median or mean with uncertainty bars for each N.

"""

# Data specifications
rangeSizeQuantumData = list(range(1, 21))
numQubits = 1
qnnArch = [numQubits, 2, numQubits]

# Training parameters
learningRate = 0.1
numEpochs = 1000
sizeTestData = 10
# loss_fn = nn.MSELoss()
loss_fns = [lossFidelityInverseSquared, lossFidelityInverseSquaredPhysInformed, 
            lossMSEPhysInformed]
loss_fn = loss_fns[0]
print(f'Loss function: {loss_fn}')

# Making DataFrame to store values
train_dict = {}
test_dict = {}
numTrials = 30

for sizeQuantumData in rangeSizeQuantumData:
  train_dict[f'{sizeQuantumData}'] = []
  test_dict[f'{sizeQuantumData}'] = []

for sizeQuantumData in rangeSizeQuantumData:
  fidelities = []
  testFidelities = []

  for i in range(numTrials):
    cnn121 = NeuralNetwork121()
    cnn121 = NeuralNetwork121().to(device)
    fidelity, testFidelity = trainModel(cnn121, learningRate, loss_fn, numEpochs, sizeQuantumData, sizeTestData, qnnArch)
    fidelities.append(fidelity.cpu().detach().numpy())
    testFidelities.append(testFidelity.cpu().detach().numpy())
    print(f'Trial ({sizeQuantumData}, {i}) done.')

  train_dict[f'{sizeQuantumData}'] = fidelities
  test_dict[f'{sizeQuantumData}'] = testFidelities

train_df = pd.DataFrame(train_dict)
test_df = pd.DataFrame(test_dict)

import os  
# train_df.to_csv(f'/home/zchua/thesis_code/csvs/MSELoss_train_df_12_2023.csv')
# test_df.to_csv(f'/home/zchua/thesis_code/csvs/MSELoss_test_df_12_2023.csv')
train_df.to_csv(f'/home/zchua/thesis_code/csvs/{loss_fn.__name__}_train_df_12_2023.csv')
test_df.to_csv(f'/home/zchua/thesis_code/csvs/{loss_fn.__name__}_test_df_12_2023.csv')
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

# Quantum data and DQNN, CNN specifications
rangeSizeQuantumData = list(range(1, 21))
qnnArch = [2, 3, 4, 3, 2]
model = NeuralNetwork23432
model_name = '23432'
directory = f'/home/zchua/thesis_code/csvs/{model_name}'

# Training and testing specs
learningRate = 0.1
numEpochs = 1000
sizeTestData = 10
numTrials = 30
loss_fns = [lossFidelityInverseSquared, 
            lossFidelityInverseSquaredPhysInformed, 
            nn.MSELoss(),
            lossMSEPhysInformed]

for loss_fn in loss_fns:
  make_cnn_generalisation_csvs(model, numTrials, 
                               learningRate, loss_fn, 
                               numEpochs, rangeSizeQuantumData, 
                               sizeTestData, qnnArch, 
                               directory)
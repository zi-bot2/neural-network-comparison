"""# CNN Generalisation"""

from cnn_functions import *

# Quantum data and DQNN, CNN specifications
rangeSizeQuantumData = list(range(1, 21))
qnnArch = [1, 2, 1]
model = NeuralNetwork_4_8_4_Linear
model_name = 'NeuralNetwork_4_8_4_Linear'
lr = 'lr_pointzerofive'
print(model_name)
print(lr)

# Training and testing specs
learningRate = 0.05
numEpochs = 1000
sizeTestData = 10
numTrials = 30
loss_fns = [lossFidelityInverseSquared, 
            lossFidelityInverseSquaredPhysInformed, 
            nn.MSELoss(),
            lossMSEPhysInformed]

directory = f'/home/zchua/thesis_code/csvs/linear_cnn/{model_name}/{lr}'
os.mkdir(directory)

for loss_fn in loss_fns:
  make_cnn_generalisation_csvs(model, numTrials, 
                               learningRate, loss_fn, 
                               numEpochs, rangeSizeQuantumData, 
                               sizeTestData, qnnArch, 
                               directory)
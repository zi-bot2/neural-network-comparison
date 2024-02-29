from cnn_functions import *

rangeSizeQuantumData = list(range(1, 21))
numQubits = 2
qnnArch = [numQubits, 3, numQubits]

learningRate = 0.1
numEpochs = 1000
sizeTestData = 10
numTrials = 30
loss_fns = [lossFidelityInverseSquared, 
            lossFidelityInverseSquaredPhysInformed, 
            nn.MSELoss(),
            lossMSEPhysInformed]

model = NeuralNetwork232
model_name = '232CNN'

directory = f'/home/zchua/thesis_code/csvs/{model_name}'
os.mkdir(directory)
for loss_fn in loss_fns:
  make_cnn_generalisation_csvs(model, numTrials, learningRate, loss_fn, numEpochs, rangeSizeQuantumData, sizeTestData, qnnArch, directory)

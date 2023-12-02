from dqnn_functions import *
# from sklearn.model_selection import train_test_split
import torch
from torch import nn
import pandas as pd
import seaborn as sns

sns.set()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
device

"""## Fidelity function for pure states"""

def fidelityPureStates(pred, target): # pred = prediction of the NN
  kTerm = 0
  lTerm = 0
  norm = 0
  
  for k in range(len(target) // 2):
    kTerm += ( (target[2*k] - target[2*k + 1]*1j) * (pred[2*k] + pred[2*k + 1]*1j) )
  for l in range(len(target) // 2):
    lTerm += ( (pred[2*l] - pred[2*l + 1]*1j) * (target[2*l] + target[2*l + 1]*1j) )
  for m in range(len(target) // 2):
    norm += ( (pred[2*m] - pred[2*m + 1]*1j) * (pred[2*m] + pred[2*m + 1]*1j) )

  return torch.real(kTerm * lTerm / norm)

def fidelityAverage(preds, targets):
  average = 0
  for i in range(len(targets)):
    average += fidelityPureStates(preds[i], targets[i])
  average = average / len(targets)

  return average

"""Old non-differentiable fidelity function"""

# def fidelityPureStates(trainingOutput, predictedOutput):
#   trainingOutput = torch.view_as_complex(torch.tensor([[trainingOutput[0], trainingOutput[1]], [trainingOutput[2], trainingOutput[3]]]))
#   predictedOutput = torch.view_as_complex(torch.tensor([[predictedOutput[0], predictedOutput[1]], [predictedOutput[2], predictedOutput[3]]]))
#   fidelity = torch.norm(torch.inner(trainingOutput, predictedOutput))**2
#   fidelity.requires_grad = True

#   return fidelity

"""## Generating random quantum data"""

def makeQuantumData(sizeQuantumData, sizeTestingData, qnnArch): # Each element of trainingInputs is a 4x1 tensor, this is how I represent a 2x1 column vector with complex-valued entries.
  dqnn = randomNetwork(qnnArch, sizeQuantumData + sizeTestingData)

  quantumData = dqnn[2]
  inputs, outputs = np.array([pair[0].full() for pair in quantumData]), np.array([pair[1].full() for pair in quantumData])
  inputs, outputs = torch.view_as_real(torch.from_numpy(inputs).type(torch.cfloat).squeeze()).flatten(start_dim = 1), torch.view_as_real(torch.from_numpy(outputs).type(torch.cfloat).squeeze()).flatten(start_dim = 1)
  inputs, outputs = inputs.to(device), outputs.to(device)

  trainingInputs, trainingOutputs = inputs[:sizeQuantumData], outputs[:sizeQuantumData]
  testingInputs, testingOutputs = inputs[sizeQuantumData : sizeQuantumData + sizeTestingData], outputs[sizeQuantumData : sizeQuantumData + sizeTestingData]
  # inputsTrain, inputsTest, outputsTrain, outputsTest = train_test_split(inputs, outputs, test_size = 0.2)

  return trainingInputs, testingInputs, trainingOutputs, testingOutputs

"""!!!Not done!!!"""
# Want to make density matrices quantum data and eventually mixed states
# def makeRandomPureDensityOps(sizeQuantumData, sizeTestingData, qnnArch):
#   dqnn = randomNetwork(qnnArch, sizeQuantumData + sizeTestingData)

#   quantumData = dqnn[2]
#   inputs, outputs = np.array([torch.outer(pair[0].full(), pair[0].full()) for pair in quantumData]), np.array([torch.outer(pair[1].full(), pair[1].full()) for pair in quantumData])
#   inputs, outputs = inputs.to(device), outputs.to(device)

#   inputsTrain, outputsTrain = inputs[:sizeQuantumData], outputs[:sizeQuantumData]
#   inputsTest, outputsTest = inputs[sizeQuantumData : sizeQuantumData + sizeTestingData], outputs[sizeQuantumData : sizeQuantumData + sizeTestingData]

#   return inputsTrain, inputsTest, outputsTrain, outputsTest

"""## Defining the 1-2-1 classical neural network class"""

class NeuralNetwork121(nn.Module):
  def __init__(self):
    super().__init__()
    self.layer_1 = nn.Linear(2*(2**1), 2*(2**2)) # 2^{numQubits} is the complex vector, then represent real and imaginary parts separately as a 2^{numQubits} x 2 matrix, then flatten that into a (2^{numQubits} * 2) x 1 vector
    self.layer_2 = nn.Linear(2*(2**2), 2*(2**1))
    self.relu = nn.ReLU()

  def forward(self, x):
    return self.layer_2(self.relu(self.layer_1(x)))

"""## Training and testing"""

def trainTestLoop(model, loss_fn, optimizer, trainingInputs, testingInputs, trainingOutputs, testingOutputs):
  model.train()

  predictedOutputs = model(trainingInputs)
  loss = loss_fn(predictedOutputs, trainingOutputs)
  fidelity = fidelityAverage(predictedOutputs, trainingOutputs)

  optimizer.zero_grad()
  loss.backward()
  optimizer.step()

  model.eval()

  with torch.no_grad():
    testPreds = model(testingInputs)
    testLoss = loss_fn(testPreds, testingOutputs)
    testFidelity = fidelityAverage(testPreds, testingOutputs)

  return loss, testLoss, fidelity, testFidelity

def trainModel(model, learningRate, loss_fn, numEpochs, sizeQuantumData, sizeTestData, qnnArch):
  trainingInputs, testingInputs, trainingOutputs, testingOutputs = makeQuantumData(sizeQuantumData, sizeTestData, qnnArch)

  optimizer = torch.optim.SGD(model.parameters(), lr = learningRate)

  for i in range(numEpochs):
      loss, testLoss, fidelity, testFidelity = trainTestLoop(model, loss_fn, optimizer, trainingInputs, testingInputs, trainingOutputs, testingOutputs)

  return fidelity, testFidelity

"""## Defining the various loss functions"""

"""
Loss(predicted output, training/testing output)
1. MSELoss
2. (1 - fidelity)^2 OR (1 - fidelity)?
3. MSELoss + (1 - norm(predicted output))^2
4. (1 - fidelity) + (1 - norm(predicted output))^2 OR (1 - fidelity)^2
"""

def norm(state):
  norm = 0
  for i in range(len(state)):
    norm += state[i]**2

  return norm

def lossFidelityInverse_sub(pred, target):
  return (1 - fidelityPureStates(pred, target))

def lossPhysInformed_sub(pred, target):
  loss = nn.MSELoss()
  return 0.9*loss(pred, target) + 0.1*(norm(pred) - 1)**2

def lossPhysInformed2_sub(pred, target):
  fidelity = fidelityPureStates(pred, target)
  loss = 0.9*((1 - fidelity)) + 0.1*((1 - norm(pred))**2)
  
  return loss

def lossAverage(loss_fn, preds, targets):
  average = 0
  for i in range(len(targets)):
    average += loss_fn(preds[i], targets[i])
  average = average / len(targets)

  return average

def lossFidelityInverse(preds, targets):
  return lossAverage(lossFidelityInverse_sub, preds, targets)

def lossPhysInformed(preds, targets):
  return lossAverage(lossPhysInformed_sub, preds, targets)

def lossPhysInformed2(preds, targets):
  return lossAverage(lossPhysInformed2_sub, preds, targets)

"""to-do
* mixed states
* larger NNs
* try density ops instead of states
"""
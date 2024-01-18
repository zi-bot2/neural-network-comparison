from dqnn_functions import *
# from sklearn.model_selection import train_test_split
import torch
from torch import nn
import pandas as pd
import seaborn as sns
import os

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

def makeQuantumData(qnnArch, sizeQuantumData, sizeTestingData): # Each element of trainingInputs is a 4x1 tensor, this is how I represent a 2x1 column vector with complex-valued entries.
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

"""## Defining the classical neural network classes"""

class NeuralNetwork121(nn.Module):
  def __init__(self):
    super().__init__()
    self.layer_1 = nn.Linear(2*(2**1), 2*(2**2)) # 2^{numQubits} is the complex vector, then represent real and imaginary parts separately as a 2^{numQubits} x 2 matrix, then flatten that into a (2^{numQubits} * 2) x 1 vector
    self.layer_2 = nn.Linear(2*(2**2), 2*(2**1))
    self.relu = nn.ReLU()

  def forward(self, x):
    return self.layer_2(self.relu(self.layer_1(x)))
  
class NeuralNetwork232(nn.Module):
  def __init__(self):
    super().__init__()
    self.layer_1 = nn.Linear(2*(2**2), 2*(2**3)) # 2^{numQubits} is the complex vector, then represent real and imaginary parts separately as a 2^{numQubits} x 2 matrix, then flatten that into a (2^{numQubits} * 2) x 1 vector
    self.layer_2 = nn.Linear(2*(2**3), 2*(2**2))
    self.relu = nn.ReLU()

  def forward(self, x):
    return self.layer_2(self.relu(self.layer_1(x)))
  
class NeuralNetwork343(nn.Module):
  def __init__(self):
    super().__init__()
    self.layer_1 = nn.Linear(2*(2**3), 2*(2**4))
    self.layer_2 = nn.Linear(2*(2**4), 2*(2**3))
    self.relu = nn.ReLU()

  def forward(self, x):
    return self.layer_2(self.relu(self.layer_1(x)))
  
class NeuralNetwork23432(nn.Module):
  def __init__(self):
    super().__init__()
    self.layer_1 = nn.Linear(2*(2**2), 2*(2**3))
    self.layer_2 = nn.Linear(2*(2**3), 2*(2**4))
    self.layer_3 = nn.Linear(2*(2**4), 2*(2**3))
    self.layer_4 = nn.Linear(2*(2**3), 2*(2**2))
    self.relu = nn.ReLU()

  def forward(self, x):
    return self.layer_4(self.relu(self.layer_3(self.relu(self.layer_2(self.relu(self.layer_1(x)))))))

"""## Linear 121"""

class NeuralNetwork121Linear(nn.Module):
  def __init__(self):
    super().__init__()
    self.layer_1 = nn.Linear(2*(2**1), 2*(2**2))
    self.layer_2 = nn.Linear(2*(2**2), 2*(2**1))

  def forward(self, x):
    return self.layer_2((self.layer_1(x)))

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
  trainingInputs, testingInputs, trainingOutputs, testingOutputs = makeQuantumData(qnnArch, sizeQuantumData, sizeTestData)

  optimizer = torch.optim.SGD(model.parameters(), lr = learningRate)

  for i in range(numEpochs):
      loss, testLoss, fidelity, testFidelity = trainTestLoop(model, loss_fn, optimizer, trainingInputs, testingInputs, trainingOutputs, testingOutputs)

  return fidelity, testFidelity

"""## Defining the various loss functions"""

"""
Loss(predicted output, training/testing output)
1. MSELoss
2. lossFidelityInverse = (1 - fidelity)
3. lossPhysInformed = 0.9*MSELoss + 0.1*(1 - norm(predicted output))^2
4. lossPhysInformed2 = 0.9*(1 - fidelity) + 0.1*(1 - norm(predicted output))^2 OR (1 - fidelity)^2
"""

def norm(state):
  norm = 0
  for i in range(len(state)):
    norm += state[i]**2

  return norm

def lossFidelityInverse_sub(pred, target):
  return (1 - fidelityPureStates(pred, target))

def lossFidelityInversePhysInformed_sub(pred, target):
  fidelity = fidelityPureStates(pred, target)
  return 0.9*((1 - fidelity)) + 0.1*((1 - norm(pred))**2)

def lossFidelityInverseSquared_sub(pred, target):
  return (1 - fidelityPureStates(pred, target))**2

def lossFidelityInverseSquaredPhysInformed_sub(pred, target):
  fidelity = fidelityPureStates(pred, target)
  return 0.9*((1 - fidelity))**2 + 0.1*((1 - norm(pred))**2)

def lossMSEPhysInformed_sub(pred, target):
  loss = nn.MSELoss()
  return 0.9*loss(pred, target) + 0.1*(norm(pred) - 1)**2

def lossAverage(loss_fn, preds, targets):
  sum = 0
  for i in range(len(targets)):
    sum += loss_fn(preds[i], targets[i])

  return sum / len(targets)

def lossFidelityInverse(preds, targets):
  return lossAverage(lossFidelityInverse_sub, preds, targets)

def lossFidelityInversePhysInformed(preds, targets):
  return lossAverage(lossFidelityInversePhysInformed_sub, preds, targets)

def lossFidelityInverseSquared(preds, targets):
  return lossAverage(lossFidelityInverseSquared_sub, preds, targets)

def lossFidelityInverseSquaredPhysInformed(preds, targets):
  return lossAverage(lossFidelityInverseSquaredPhysInformed_sub, preds, targets)

def lossMSEPhysInformed(preds, targets):
  return lossAverage(lossMSEPhysInformed_sub, preds, targets)


"""Generalisation functions"""

def make_cnn_generalisation_csvs(model, numTrials, learningRate, loss_fn, numEpochs, rangeSizeQuantumData, sizeTestData, qnnArch, directory):
  if f'{loss_fn}' == 'MSELoss()':
    loss_fn_name = 'MSELoss'
  else:
    loss_fn_name = loss_fn.__name__
  
  print(f'Loss function: {loss_fn_name}')

  train_dict = {}
  test_dict = {}

  for sizeQuantumData in rangeSizeQuantumData:
    train_dict[f'{sizeQuantumData}'] = []
    test_dict[f'{sizeQuantumData}'] = []

  for sizeQuantumData in rangeSizeQuantumData:
    fidelities = []
    testFidelities = []

    for i in range(numTrials):
      cnn = model().to(device)
      fidelity, testFidelity = trainModel(cnn, learningRate, loss_fn, numEpochs, sizeQuantumData, sizeTestData, qnnArch)
      fidelities.append(fidelity.cpu().detach().numpy())
      testFidelities.append(testFidelity.cpu().detach().numpy())
      print(f'Trial ({sizeQuantumData}, {i}) done.')

    train_dict[f'{sizeQuantumData}'] = fidelities
    test_dict[f'{sizeQuantumData}'] = testFidelities

  train_df = pd.DataFrame(train_dict)
  test_df = pd.DataFrame(test_dict)

  train_df.to_csv(f'{directory}/{loss_fn_name}_train_df.csv')
  test_df.to_csv(f'{directory}/{loss_fn_name}_test_df.csv')

"""to-do
* mixed states
* try density ops instead of states
"""
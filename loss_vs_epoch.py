"""# Making loss vs epoch plots for the classical NN"""

from cnn_functions import *

def plotLossVsEpoch(model, loss_fn, learningRate, numEpochs, trainingInputs, testingInputs, trainingOutputs, testingOutputs):
  optimizer = torch.optim.SGD(model.parameters(), lr = learningRate)
  
  loss_dict = {'Epochs': list(range(numEpochs)), 'Training loss': [], 'Testing loss': [], 'Training fidelity': [], 'Testing fidelity': []}
  for i in range(numEpochs):
    loss, testLoss, fidelity, testFidelity = trainTestLoop(model, loss_fn, optimizer, trainingInputs, testingInputs, trainingOutputs, testingOutputs)
    loss_dict['Training fidelity'].append(fidelity.item())
    loss_dict['Testing fidelity'].append(testFidelity.item())
  
  epochs, training_fidelities, testing_fidelities = loss_dict['Epochs'], loss_dict['Training fidelity'], loss_dict['Testing fidelity']
  
  return epochs, training_fidelities, testing_fidelities

sizeTrainingData = 10
sizeTestingData = 10
qnnArch = [2, 3, 4, 3, 2]
trainingInputs, testingInputs, trainingOutputs, testingOutputs = makeQuantumData(qnnArch, sizeTrainingData, sizeTestingData)

loss_fns = [nn.MSELoss(),
            lossMSEPhysInformed,
            lossFidelityInverseSquared, 
            lossFidelityInverseSquaredPhysInformed,
            lossNormed]
# loss_fns = [nn.MSELoss()]

numEpochs = 1000
learningRate = 1
cfnn_arch = '8_41_49_41_8_Linear'
model_name = '8-41-49-41-8 Linear'
lr = 'lr_one'

for loss_fn in loss_fns:
  model = NeuralNetwork_8_41_49_41_8_Linear()
  model = model.to(device)
  
  if f'{loss_fn}' == 'MSELoss()':
    loss_fn_name = 'MSELoss'
  else:
    loss_fn_name = loss_fn.__name__

  print(f'{model_name}, lr = {learningRate}, {loss_fn_name}')
  epochs, training_fidelities, testing_fidelities = plotLossVsEpoch(model, loss_fn, learningRate, numEpochs, trainingInputs, testingInputs, trainingOutputs, testingOutputs)
  
  plt.plot(epochs, testing_fidelities, label = 'Testing')
  plt.plot(epochs, training_fidelities, label = 'Training')
  plt.title(f'{model_name} CFNN \nLoss function = {loss_fn_name}\n# training pairs = {sizeTrainingData}\n# testing pairs = {sizeTestingData}\nLearning rate = {learningRate}')
  plt.legend()
  plt.xlabel('Epoch')
  plt.ylabel('Fidelity')
  plt.ylim([0, 1.1])
  plt.show()
  plt.savefig(f'/home/zchua/thesis_code/plots/thesis/cfnn_{cfnn_arch}_{loss_fn_name}_{lr}.pdf', bbox_inches='tight', dpi=300)
  plt.close()
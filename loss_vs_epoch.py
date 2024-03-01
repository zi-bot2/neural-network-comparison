"""# Making loss vs epoch plots for the classical NN"""

from cnn_functions import *

def plotLossVsEpoch(model, model_name, loss_fn, sizeTrainingData, sizeTestingData, learningRate, numEpochs, trainingInputs, testingInputs, trainingOutputs, testingOutputs):
  optimizer = torch.optim.SGD(model.parameters(), lr = learningRate)
  
  loss_dict = {'Epochs': list(range(numEpochs)), 'Training loss': [], 'Testing loss': [], 'Training fidelity': [], 'Testing fidelity': []}
  for i in range(numEpochs):
    loss, testLoss, fidelity, testFidelity = trainTestLoop(model, loss_fn, optimizer, trainingInputs, testingInputs, trainingOutputs, testingOutputs)
    # loss_dict['Training loss'].append(loss.item())
    # loss_dict['Testing loss'].append(testLoss.item())
    loss_dict['Training fidelity'].append(fidelity.item())
    loss_dict['Testing fidelity'].append(testFidelity.item())

  if f'{loss_fn}' == 'MSELoss()':
    loss_fn_name = 'MSELoss'
  else:
    loss_fn_name = loss_fn.__name__
  
  plt.plot(loss_dict['Epochs'], loss_dict['Testing fidelity'], label = 'Testing fidelity')
  plt.plot(loss_dict['Epochs'], loss_dict['Training fidelity'], label = 'Training fidelity')
  # plt.plot(loss_dict['Epochs'], loss_dict['Testing loss'], label = 'Testing loss')
  # plt.plot(loss_dict['Epochs'], loss_dict['Training loss'], label = 'Training loss')
  plt.title(f'{model_name} <-> {qnnArch} DQNN\nLoss function = {loss_fn_name}\n# training pairs = {sizeTrainingData}\n# testing pairs = {sizeTestingData}\nLearning rate = {learningRate}')
  plt.legend()
  plt.xlabel('Epoch')
  plt.ylim([0, 1.1])
  plt.show()
  # plt.savefig(f'/home/zchua/thesis_code/plots/master_talk/{loss_fn.__name__}_performance.pdf', bbox_inches='tight', dpi=300)
  # # plt.savefig(f'/home/zchua/thesis_code/plots/master_talk/MSELoss_performance.pdf', bbox_inches='tight', dpi=300)
  # plt.close()

sizeTrainingData = 10
sizeTestingData = 10
qnnArch = [1, 2, 1]
trainingInputs, testingInputs, trainingOutputs, testingOutputs = makeQuantumData(qnnArch, sizeTrainingData, sizeTestingData)

learningRate = 0.1
numEpochs = 1000

loss_fns = [lossNormed,
            lossFidelityInverseSquared, 
            lossFidelityInverseSquaredPhysInformed,
            nn.MSELoss(),
            lossMSEPhysInformed,]

model_name = '4-8-4 Linear CNN'
for loss_fn in loss_fns:
  model = NeuralNetwork_4_8_4_Linear()
  model = model.to(device)
  plotLossVsEpoch(model, model_name, loss_fn, sizeTrainingData, sizeTestingData, learningRate, numEpochs, trainingInputs, testingInputs, trainingOutputs, testingOutputs)
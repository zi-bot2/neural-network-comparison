"""# Making loss vs epoch plots for the classical NN"""

from cnn_functions import *

sizeTrainingData = 1
sizeTestingData = 10
qnnArch = [1, 2, 1]
trainingInputs, testingInputs, trainingOutputs, testingOutputs = makeQuantumData(qnnArch, sizeTrainingData, sizeTestingData)

learningRate = 0.1
numEpochs = 1000

loss_fns = [lossFidelityInverseSquared, lossFidelityInverseSquaredPhysInformed, 
            lossMSEPhysInformed]

def plotLossVsEpoch(model, loss_fn, sizeTrainingData, sizeTestingData, learningRate, numEpochs, trainingInputs, testingInputs, trainingOutputs, testingOutputs):
  optimizer = torch.optim.SGD(model.parameters(), lr = learningRate)
  
  loss_dict = {'Epochs': list(range(numEpochs)), 'Training loss': [], 'Testing loss': [], 'Training fidelity': [], 'Testing fidelity': []}
  for i in range(numEpochs):
    loss, testLoss, fidelity, testFidelity = trainTestLoop(model, loss_fn, optimizer, trainingInputs, testingInputs, trainingOutputs, testingOutputs)
    loss_dict['Training loss'].append(loss.item())
    loss_dict['Testing loss'].append(testLoss.item())
    loss_dict['Training fidelity'].append(fidelity.item())
    loss_dict['Testing fidelity'].append(testFidelity.item())

  plt.plot(loss_dict['Epochs'], loss_dict['Testing fidelity'], label = 'Testing fidelity')
  plt.plot(loss_dict['Epochs'], loss_dict['Training fidelity'], label = 'Training fidelity')
  plt.plot(loss_dict['Epochs'], loss_dict['Testing loss'], label = 'Testing loss')
  plt.plot(loss_dict['Epochs'], loss_dict['Training loss'], label = 'Training loss')
  plt.title(f'4-8-4 NN\nLoss function = {loss_fn.__name__} \n# training pairs = {sizeTrainingData} \n# testing pairs = {sizeTestingData}')
  # plt.title(f'4-8-4 NN\nLoss function = MSELoss\n# training pairs = {sizeTrainingData} \n# testing pairs = {sizeTestingData}')
  plt.legend()
  plt.xlabel('Epoch')
  plt.show()
  plt.savefig(f'/home/zchua/thesis_code/plots/master_talk/{loss_fn.__name__}_performance.pdf', bbox_inches='tight', dpi=300)
  # # plt.savefig(f'/home/zchua/thesis_code/plots/master_talk/MSELoss_performance.pdf', bbox_inches='tight', dpi=300)
  plt.close()


for loss_fn in loss_fns:
  cnn121 = NeuralNetwork121().to(device)
  plotLossVsEpoch(cnn121, loss_fn, sizeTrainingData, sizeTestingData, learningRate, numEpochs, trainingInputs, testingInputs, trainingOutputs, testingOutputs)

# # Then modified plotLossVsEpoch bc nn.MSELoss() doesn't have a function .__name__
# cnn121 = NeuralNetwork121().to(device)
# plotLossVsEpoch(cnn121, nn.MSELoss(), sizeTrainingData, sizeTestingData, learningRate, numEpochs, trainingInputs, testingInputs, trainingOutputs, testingOutputs)
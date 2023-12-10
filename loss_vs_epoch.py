"""# Making loss vs epoch plots for the classical NN"""

from cnn_functions import *

sizeTrainingData = 5
sizeTestingData = 10
qnnArch = [1, 2, 1]
inputsTrain, inputsTest, outputsTrain, outputsTest = makeQuantumData(qnnArch, sizeTrainingData, sizeTestingData)

learningRate = 0.1
numEpochs = 10

# loss_fn = lossFidelityInverse
# loss_fn = lossFidelityInversePhysInformed
# loss_fn = lossFidelityInverseSquared
# loss_fn = lossFidelityInverseSquaredPhysInformed
# loss_fn = nn.MSELoss()
# loss_fn = lossMSEPhysInformed

loss_fns = [lossFidelityInverse, lossFidelityInversePhysInformed, 
            lossFidelityInverseSquared, lossFidelityInverseSquaredPhysInformed,
            nn.MSELoss(), lossMSEPhysInformed]


cnn121 = NeuralNetwork121().to(device)

def plotLossVsEpoch(model, loss_fn, sizeTrainingData, sizeTestingData, learningRate, numEpochs, inputsTrain, inputsTest, outputsTrain, outputsTest):
  optimizer = torch.optim.SGD(model.parameters(), lr = learningRate)
  
  loss_dict = {'Epochs': list(range(numEpochs)), 'Training loss': [], 'Testing loss': [], 'Training fidelity': [], 'Testing fidelity': []}
  for i in range(numEpochs):
    loss, testLoss, fidelity, testFidelity = trainTestLoop(model, loss_fn, optimizer, inputsTrain, outputsTrain, inputsTest, outputsTest)
    loss_dict['Training loss'].append(loss.item())
    loss_dict['Testing loss'].append(testLoss.item())
    loss_dict['Training fidelity'].append(fidelity.item())
    loss_dict['Testing fidelity'].append(testFidelity.item())

  plt.plot(loss_dict['Epochs'], loss_dict['Training loss'], label = 'Training loss')
  plt.plot(loss_dict['Epochs'], loss_dict['Testing loss'], label = 'Testing loss')
  plt.plot(loss_dict['Epochs'], loss_dict['Training fidelity'], label = 'Training fidelity')
  plt.plot(loss_dict['Epochs'], loss_dict['Testing fidelity'], label = 'Testing fidelity')
  plt.title(f'4-8-4 NN\nLoss function = {loss_fn.__name__} \n# training pairs = {sizeTrainingData} \n# testing pairs = {sizeTestingData}')
  plt.legend()
  plt.xlabel('Epoch')
  plt.show()
  # plt.savefig(f'/home/zchua/thesis_code/plots/master_talk/{loss_fn.__name__}_performance.pdf', bbox_inches='tight', dpi=300)
  # plt.close()


plotLossVsEpoch(cnn121, loss_fns[0], sizeTrainingData, sizeTestingData, learningRate, numEpochs, inputsTrain, inputsTest, outputsTrain, outputsTest)
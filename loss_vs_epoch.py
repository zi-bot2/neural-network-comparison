"""# Making loss vs epoch plots"""

from cnn_functions import *

"""## CNN performance"""

sizeQuantumData = 10
sizeTestData = 10
numQubits = 1
qnnArch = [numQubits, 2, numQubits]
inputsTrain, inputsTest, outputsTrain, outputsTest = makeQuantumData(sizeQuantumData, sizeTestData, qnnArch)

learningRate = 0.1
numEpochs = 2000
cnn121 = NeuralNetwork121()
cnn121 = NeuralNetwork121().to(device)
optimizer = torch.optim.SGD(cnn121.parameters(), lr = learningRate)
loss_fn = nn.MSELoss()
# loss_fn = lossFidelityInverse
# loss_fn = lossPhysInformed
# loss_fn = lossPhysInformed2

loss_dict = {'Epochs': list(range(numEpochs)), 'Training loss': [], 'Testing loss': [], 'Training fidelity': [], 'Testing fidelity': []}
for i in range(numEpochs):
  loss, testLoss, fidelity, testFidelity = trainTestLoop(cnn121, loss_fn, optimizer, inputsTrain, outputsTrain, inputsTest, outputsTest)
  loss_dict['Training loss'].append(loss.item())
  loss_dict['Testing loss'].append(testLoss.item())
  loss_dict['Training fidelity'].append(fidelity.item())
  loss_dict['Testing fidelity'].append(testFidelity.item())

plt.plot(loss_dict['Epochs'], loss_dict['Training loss'], label = 'Training loss')
plt.plot(loss_dict['Epochs'], loss_dict['Testing loss'], label = 'Testing loss')
plt.plot(loss_dict['Epochs'], loss_dict['Training fidelity'], label = 'Training fidelity')
plt.plot(loss_dict['Epochs'], loss_dict['Testing fidelity'], label = 'Testing fidelity')
plt.title(f'4-8-4 NN <=> {qnnArch} QNN \n Loss function = {loss_fn} \n sizeQuantumData = {sizeQuantumData} \n sizeTestData = {sizeTestData}')
plt.legend()
# plt.ylim(top=1.5)
plt.xlabel('Epoch')
plt.savefig(f'/home/zchua/thesis_code/plots/{loss_fn}_performance.pdf', bbox_inches='tight', dpi=300)
# plt.close()

"""## DQNN performance"""

# lambda_ = 1
# epsilon = 0.1

# network = randomNetwork(qnnArch, sizeQuantumData)

# plotlist121 = qnnTraining(network[0], network[1], network[2],
#                           lambda_, epsilon, numEpochs)[0]

# # for i in range(len(plotlist121[1])):
# #   if plotlist121[1][i] >= 0.95:
# #     print('Exceeds cost of 0.95 at training step ' + str(i))
# #     break

# plt.plot(plotlist121[0], plotlist121[1])
# plt.xlabel('s')
# plt.ylabel('Cost[s]')
# plt.show()
"""Loss vs epoch for DQNN"""

from dqnn_functions import *
import seaborn as sns
sns.set()

# Training only

numQubits = 1
numTrainingPairs = 10
numTestingPairs = 10
qnnArch = [numQubits, 2, numQubits]
lambda_ = 1
epsilon = 0.1
numEpochs = 500

network = randomNetwork(qnnArch, numTrainingPairs + numTestingPairs)

plotlist121 = qnnTraining(network[0], network[1], network[2],
                          lambda_, epsilon, numEpochs)[0]

for i in range(len(plotlist121[1])):
  if plotlist121[1][i] >= 0.95:
    print('Exceeds cost of 0.95 at training step ' + str(i))
    break

plt.plot(plotlist121[0], plotlist121[1])
plt.xlabel('s')
plt.ylabel('Cost[s]')
plt.show()


# Training and testing

numQubits = 1
numTrainingPairs = 10
numTestingPairs = 10
qnnArch = [numQubits, 2, numQubits]
lambda_ = 1
epsilon = 0.1
numEpochs = 1000

network = randomNetwork(qnnArch, numTrainingPairs + numTestingPairs)

plotlist121, testPlotlist121 = qnnTrainingTesting(network[0], network[1], network[2], numTrainingPairs, numTestingPairs, lambda_, epsilon, numEpochs)[0], qnnTrainingTesting(network[0], network[1], network[2], numTrainingPairs, numTestingPairs, lambda_, epsilon, numEpochs)[1]

plt.title('Fidelity during DQNN training \nsizeTrainingData = {numTrainingPairs} \n sizeTestingData = {numTestingPairs}')
plt.plot(plotlist121[0], plotlist121[1], label='Training')
plt.plot(testPlotlist121[0], testPlotlist121[1], label='Testing')
plt.legend()
plt.xlabel('s')
plt.ylabel('Fidelity[s]')
plt.show()
plt.savefig(f'/home/zchua/thesis_code/plots/dqnn_performance.pdf', bbox_inches='tight', dpi=300)
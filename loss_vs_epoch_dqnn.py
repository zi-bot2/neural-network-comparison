"""# Loss vs epoch for DQNN"""

from dqnn_functions import *
import seaborn as sns
sns.set()

numTrainingPairs = 10
numTestingPairs = 10
qnnArch = [2, 3, 4, 3, 2]
lambda_ = 1
epsilon = 0.03
numEpochs = 1000

network = randomNetwork(qnnArch, numTrainingPairs + numTestingPairs)

plotlist121, testPlotlist121, currentUnitaries = qnnTrainingTesting(network[0], network[1], network[2], numTrainingPairs, numTestingPairs, lambda_, epsilon, numEpochs)

plt.title(f'{qnnArch} DQNN\n# training pairs = {numTrainingPairs}\n# testing pairs = {numTestingPairs}\n Learning rate = {epsilon}')
plt.plot(testPlotlist121[0], testPlotlist121[1], label='Testing')
plt.plot(plotlist121[0], plotlist121[1], label='Training')
plt.legend()
plt.ylim([0, 1.1])
plt.xlabel('s')
plt.ylabel('Fidelity[s]')
plt.savefig(f'/home/zchua/thesis_code/plots/tests/dqnn_2_3_4_3_2_lr_pointzerothree.pdf', bbox_inches='tight', dpi=300)
plt.close()
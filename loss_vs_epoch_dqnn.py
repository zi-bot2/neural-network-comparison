"""# Loss vs epoch for DQNN"""

from dqnn_functions import *
import seaborn as sns
sns.set()

numTrainingPairs = 10
numTestingPairs = 10
qnnArch = [1, 2, 1]
lambda_ = 1
epsilon = 0.1
numEpochs = 1000

network = randomNetwork(qnnArch, numTrainingPairs + numTestingPairs)

plotlist121, testPlotlist121, currentUnitaries = qnnTrainingTesting(network[0], network[1], network[2], numTrainingPairs, numTestingPairs, lambda_, epsilon, numEpochs)

plt.title(f'1-2-1 DQNN\n# training pairs = {numTrainingPairs}\n# testing pairs = {numTestingPairs}')
plt.plot(testPlotlist121[0], testPlotlist121[1], label='Testing')
plt.plot(plotlist121[0], plotlist121[1], label='Training')
plt.legend()
plt.ylim([0, 1.1])
plt.xlabel('s')
plt.ylabel('Fidelity[s]')
plt.savefig(f'/home/zchua/thesis_code/plots/master_talk/dqnn_fidelity_performance_2.pdf', bbox_inches='tight', dpi=300)
plt.close()
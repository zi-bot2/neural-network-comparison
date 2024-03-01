"""# Loss vs epoch for DQNN"""

from dqnn_functions import *
import seaborn as sns
sns.set()

numTrainingPairs = 10
numTestingPairs = 10
qnnArch = [3, 4, 3]
lambda_ = 1
epsilon = 1
numEpochs = 1000
dqnn_arch = '3_4_3'
lr = 'lr_one'
print(f'dqnn_arch = {qnnArch}, learning rate = {epsilon}')

network = randomNetwork(qnnArch, numTrainingPairs + numTestingPairs)

plotlist, testPlotlist, currentUnitaries = qnnTrainingTesting(network[0], network[1], network[2], numTrainingPairs, numTestingPairs, lambda_, epsilon, numEpochs)

plt.title(f'{qnnArch} DQNN\n# training pairs = {numTrainingPairs}\n# testing pairs = {numTestingPairs}\n Learning rate = {epsilon}')
plt.plot(testPlotlist[0], testPlotlist[1], label='Testing')
plt.plot(plotlist[0], plotlist[1], label='Training')
plt.legend()
plt.ylim([0, 1.1])
plt.xlabel('s')
plt.ylabel('Fidelity[s]')
plt.savefig(f'/home/zchua/thesis_code/plots/tests/dqnn_{dqnn_arch}_{lr}.pdf', bbox_inches='tight', dpi=300)
plt.close()
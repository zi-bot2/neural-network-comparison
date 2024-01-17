"""# Plotting generalisations for diff loss functions"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
sns.set()
# os.mkdir()

trainFile = '/home/zchua/thesis_code/csvs/121LinearCNN/lossMSEPhysInformed_train_df.csv'
testFile = '/home/zchua/thesis_code/csvs/121LinearCNN/lossMSEPhysInformed_test_df.csv'

train_df = pd.read_csv(trainFile)
test_df = pd.read_csv(testFile)

loss_fn = trainFile[len('/home/zchua/thesis_code/csvs/') : trainFile.find('_train_df')]
numEpochs = 1000
# loss_fn = 'MSELoss'

training_x = list(train_df)[1:]
training_y = [train_df[f'{sizeTrainingData}'].mean() for sizeTrainingData in training_x]
training_stds = [train_df[f'{sizeTrainingData}'].std() for sizeTrainingData in training_x]

testing_x = list(test_df)[1:]
testing_y = [test_df[f'{sizeTrainingData}'].mean() for sizeTrainingData in testing_x]
testing_stds = [test_df[f'{sizeTrainingData}'].std() for sizeTrainingData in testing_x]

plt.scatter(testing_x, testing_y, label='Testing')
plt.errorbar(testing_x, testing_y, yerr=testing_stds, fmt = 'o')
plt.scatter(training_x, training_y, label='Training')
plt.errorbar(training_x, training_y, yerr=training_stds, fmt = 'o')
plt.ylim([0, 1.1])
plt.xlabel('# training pairs')
plt.ylabel('Fidelity')
plt.legend()
# plt.title(f'1-2-1 DQNN\nFidelity after {numEpochs} training epochs')
# plt.savefig(f'/home/zchua/thesis_code/plots/121_dqnn_generalisation.pdf', bbox_inches='tight', dpi=300)
plt.title(f'Fidelity for 4-8-4 CNN after {numEpochs} epochs\nLoss = {loss_fn}')

# plt.savefig(f'/home/zchua/thesis_code/plots/121_{loss_fn}_generalisation.pdf', bbox_inches='tight', dpi=300)
# plt.close()
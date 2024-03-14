"""# Plotting generalisations for diff loss functions"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from dqnn_functions import *
sns.set()
# os.mkdir()

loss_fns = ['lossFidelityInverseSquared', 'MSELoss']
loss_fn = loss_fns[1]

mode = 'qnn'
numEpochs = 1000
lr = 'lr_pointzerofive'

if mode == 'cnn':
    arch = 'NeuralNetwork_8_41_8_Linear'
    nn_name = '8-41-8 Linear'
    save_name = 'cfnn_8_41_8_Linear'
    model = f'linear_cnn/{arch}/{lr}'
    trainFile = f'/home/zchua/thesis_code/csvs/{model}/{loss_fn}_train_df.csv'
    testFile = f'/home/zchua/thesis_code/csvs/{model}/{loss_fn}_test_df.csv'
    # cnn_arch = trainFile[len('/home/zchua/thesis_code/csvs/') : trainFile.find(f'/{loss_fn}_train_df.csv')]

if mode == 'qnn':
    model = '343'
    nn_name = '3-4-3'
    save_name = 'dqnn_3_4_3'
    trainFile = f'/home/zchua/thesis_code/csvs/dqnn/{model}/{lr}/dqnn_train_df.csv'
    testFile = f'/home/zchua/thesis_code/csvs/dqnn/{model}/{lr}/dqnn_test_df.csv'

train_df = pd.read_csv(trainFile)
test_df = pd.read_csv(testFile)

training_x = list(train_df)[1:]
training_y = [train_df[f'{sizeTrainingData}'].mean() for sizeTrainingData in training_x]
training_stds = [train_df[f'{sizeTrainingData}'].std() for sizeTrainingData in training_x]
training_x = [int(i) for i in training_x]

testing_x = list(test_df)[1:]
testing_y = [test_df[f'{sizeTrainingData}'].mean() for sizeTrainingData in testing_x]
testing_stds = [test_df[f'{sizeTrainingData}'].std() for sizeTrainingData in testing_x]
testing_x = [int(i) for i in testing_x]

dim_input_space = 2**(int(model[0]))
training_set_sizes = np.arange(1, dim_input_space+1)
qnfl_values = make_qnfl_bound_plotlist(dim_input_space, training_set_sizes)

plt.scatter(testing_x, testing_y, label='Testing')
plt.errorbar(testing_x, testing_y, yerr=testing_stds, fmt = 'o')
plt.scatter(training_x, training_y, label='Training')
plt.errorbar(training_x, training_y, yerr=training_stds, fmt = 'o')
plt.scatter(training_set_sizes, qnfl_values, label='QNFL')
plt.xticks(np.arange(1, 21, step=1))
plt.ylim([0, 1.1])
plt.xlabel('S')
plt.ylabel('Fidelity')
plt.legend()
if mode == 'cnn':
    plt.title(f'{nn_name} CNN\nFidelity after {numEpochs} training epochs\nLoss function = {loss_fn}')
    plt.savefig(f'/home/zchua/thesis_code/plots/thesis/generalisation/{save_name}_{lr}_{loss_fn}_generalisation.pdf', bbox_inches='tight', dpi=300)
elif mode == 'qnn':
    plt.title(f'{nn_name} DQNN\nFidelity after {numEpochs} training epochs')
    plt.savefig(f'/home/zchua/thesis_code/plots/thesis/generalisation/{save_name}_{lr}_qnfl.pdf', bbox_inches='tight', dpi=300)
plt.show()
# plt.close()
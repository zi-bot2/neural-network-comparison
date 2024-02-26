"""# Plotting generalisations for diff loss functions"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
sns.set()
# os.mkdir()

loss_fns = ['lossFidelityInverseSquared', 
            'lossFidelityInverseSquaredPhysInformed', 
            'MSELoss',
            'lossMSEPhysInformed']
loss_fn = loss_fns[3]

mode = 'cnn'
numEpochs = 1000
lr = 'lr_pointzerofive'

if mode == 'cnn':
    arch = 'NeuralNetwork_4_10_4_Linear'
    cnn_arch = '4-10-4'
    model = f'linear_cnn/{arch}/{lr}'
    trainFile = f'/home/zchua/thesis_code/csvs/{model}/{loss_fn}_train_df.csv'
    testFile = f'/home/zchua/thesis_code/csvs/{model}/{loss_fn}_test_df.csv'
    # cnn_arch = trainFile[len('/home/zchua/thesis_code/csvs/') : trainFile.find(f'/{loss_fn}_train_df.csv')]

if mode == 'qnn':
    model = 'dqnn/23432'
    trainFile = f'/home/zchua/thesis_code/csvs/{model}/dqnn_train_df.csv'
    testFile = f'/home/zchua/thesis_code/csvs/{model}/dqnn_test_df.csv'
    # qnn_arch = trainFile[len('/home/zchua/thesis_code/csvs/') : trainFile.find(f'/dqnn_train_df.csv')]
    qnn_arch = '2-3-4-3-2'

train_df = pd.read_csv(trainFile)
test_df = pd.read_csv(testFile)

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
if mode == 'cnn':
    plt.title(f'{cnn_arch} CNN\nFidelity after {numEpochs} epochs\nLoss = {loss_fn}')
    # plt.savefig(f'/home/zchua/thesis_code/plots/121_{loss_fn}_generalisation.pdf', bbox_inches='tight', dpi=300)
if mode == 'qnn':
    plt.title(f'{qnn_arch} DQNN\nFidelity after {numEpochs} training epochs')
    # plt.savefig(f'/home/zchua/thesis_code/plots/121_dqnn_generalisation.pdf', bbox_inches='tight', dpi=300)
# plt.close()
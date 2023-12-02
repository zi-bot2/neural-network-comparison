"""# Plotting generalisations for diff loss functions"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

trainFile = '/home/zchua/thesis_code/lossFidelityInverse_train_df.csv'
testFile = '/home/zchua/thesis_code/lossFidelityInverse_test_df.csv'
# trainFile = '/home/zchua/thesis_code/MSELoss()_train_df.csv'
# testFile = '/home/zchua/thesis_code/MSELoss()_test_df.csv'
train_df = pd.read_csv(trainFile)
test_df = pd.read_csv(testFile)

loss_fn = trainFile[len('/home/zchua/thesis_code/') : trainFile.find('_train_df.csv')]
loss_fn
numEpochs = 1000

training_x = list(train_df)[1:]
training_y = [train_df[f'{sizeQuantumData}'].mean() for sizeQuantumData in x]
training_stds = [train_df[f'{sizeQuantumData}'].std() for sizeQuantumData in x]

testing_x = list(test_df)[1:]
testing_y = [test_df[f'{sizeQuantumData}'].mean() for sizeQuantumData in x]
testing_stds = [test_df[f'{sizeQuantumData}'].std() for sizeQuantumData in x]

plt.scatter(testing_x, testing_y, label='test')
plt.errorbar(testing_x, testing_y, yerr=testing_stds, fmt = 'o')
plt.scatter(training_x, training_y, label='train')
plt.errorbar(training_x, training_y, yerr=training_stds, fmt = 'o')
plt.xlabel('sizeQuantumData')
plt.ylabel('Fidelity')
plt.title(f'Fidelity for 4-8-4 CNN (1-2-1 DQNN) after {numEpochs} epochs\nLoss = {loss_fn}')
plt.legend()
plt.savefig(f'/home/zchua/thesis_code/plots/{loss_fn}_generalisation.pdf', bbox_inches='tight', dpi=300)
plt.close()
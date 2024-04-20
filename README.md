# Learning Quantum Unitaries: Classical Feedforward Neural Networks vs Quantum Dissipative Neural Networks

In this project I compare classical feedfoward neural networks (CFNN) built with PyTorch to classical simulations of small "dissipative quantum neural networks" (DQNN) simulated with the Python library QuTip, which is a quantum circuit with parametrised gates that can learn from quantum states as input.
The learning task I compare them with is that of learning an unknown unitary. This code generates loss vs epoch plots as well as generalisation* plots for each of the neural networks. These DQNNs were first developed by Kerstin Beer and others at the Quantum Information Group at Leibniz Universitaet Hannover in 2020.

*"generalisation" means I do several trials of initialising the neural network, then training it for a certain number of epochs on a training data set of a certain size, and then seeing how well it does on testing data for a training data set of that particular size. I'm interested in the average testing loss over these trials for training sets of different sizes. The idea is to see how well the network can "generalise" on average after training on limited training data.

## What each file does
The file cnn_functions.py defines all the relevant functions for instantiating, training and testing the CFNNs, including the random generation of training and testing data, the classes for the neural network architectures I wanted to study, and the loss functions.
The file cnn_generalisation.py generates CSVs of the generalisation data and stores it in the specified directory.
The file dqnn_functions.py similarly defines all the relevant functions for the DQNNs, and dqnn_generalisation.py generates the CSVs.
The file generalisation_plots.py generates generalisation plots from the generalisation CSVs and stores them in the specified directory.
The file loss_vs_epoch.py generates loss vs. epoch plots for the CFNNs (specifically quantum fidelity vs epoch plots, but the fidelity is maximised exactly when each of the loss functions I've defined are minimised) and stores them in the specified directory.
The file loss_vs_epoch_dqnn.py generates fidelity vs. epoch plots for the DQNNs and stores them in the specified directory.
The file qnfl_plotting.py generates the plot of the "quantum no free lunch" (QNFL) bound for the specified DQNN architecture. The QNFL bound is the theoretical bound on how well the DQNN can generalise for this particular learning task.

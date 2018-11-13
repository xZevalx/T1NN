from NeuralNetwork import NeuralNetwork, Sigmoid

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

import sys


# def precision(tp, fp):
#     return tp / (tp + fp)
#
#
# def recall(tp, fn):
#     return tp / (tp + fn)

def score(yes, total):
    """

    :param yes: (int) Aciertos en la clasificación
    :param total: (int) total de datos testeados
    :return: Ratio aciertos sobre el total
    """
    return yes / total


def aprox_nn_output(nn_output):
    """
    Binariza una salida de la red neuronal.
    Ejemplo: [.345, .788] -> [0 , 1]
    """
    return [0 if i < .5 else 1 for i in nn_output]


def learning_rate_experiment(training_data, validation_data, ninputs, noutputs):
	lrs = [.1, .35, .75, 1.]
	scores = []
	total = len(validation_data)

	plt.figure()

	for lr in lrs:
	    # Entrenamiento
	    nn = NeuralNetwork(layers_conf=[ninputs, 2 * ninputs, 3 * ninputs, noutputs],
	                       neuron_type=Sigmoid,
	                       learning_rate=lr)
	    errors = nn.epoch_training(training_data, epoch=80)
	    plt.plot(errors, label='LR: {}'.format(lr))

	    # Validación
	    yes = 0
	    for input_val_data, expected_output in validation_data:
	        nn_output = aprox_nn_output(nn.feed_forward(input_val_data))
	        if nn_output == expected_output:
	            yes += 1

	    scores.append(score(yes, total))

	plt.xlabel('Epoch')
	plt.ylabel('Error normalizado')
	plt.title('Curva de error normalizado durante entrenamiento.\nLayer={}'.format([ninputs, 2 * ninputs, 3 * ninputs, noutputs]))
	plt.legend()
	plt.savefig('training_error_learning_rate_experiment.png')


	plt.figure()
	plt.bar(range(1, len(lrs) + 1), scores)
	plt.xticks(range(1, len(lrs) + 1), lrs)
	plt.xlabel('Learning rate')
	plt.ylabel('Score')
	plt.title('Score vs learning rate.\n {} muestras de entrenamiento y {} muestras de validación'.format(len(leaf)-total,
	                                                                                                    total))
	plt.savefig('score_vs_lr.png')


def layers_experiment(training_data, validation_data, ninputs, noutputs):
	layers_configs = [[ninputs, 2 * ninputs, noutputs],
					  [ninputs, 3 * ninputs, noutputs],
					  [ninputs, 2 * ninputs, 2 * ninputs, noutputs],
					  [ninputs, 3 * ninputs, 3 * ninputs, noutputs]]
	scores = []
	total = len(validation_data)

	plt.figure()

	for layers_conf in layers_configs:
	    # Entrenamiento
	    nn = NeuralNetwork(layers_conf=layers_conf,
	                       neuron_type=Sigmoid,
	                       learning_rate=.1)
	    errors = nn.epoch_training(training_data, epoch=80)
	    plt.plot(errors, label='layers: {}'.format(layers_conf))

	    # Validación
	    yes = 0
	    for input_val_data, expected_output in validation_data:
	        nn_output = aprox_nn_output(nn.feed_forward(input_val_data))
	        if nn_output == expected_output:
	            yes += 1

	    scores.append(score(yes, total))

	plt.xlabel('Epoch')
	plt.ylabel('Error normalizado')
	plt.title('Curva de error normalizado durante entrenamiento. LR=0.1')
	plt.legend()
	plt.savefig('training_error_layers_experiment.png')


	plt.figure()
	plt.bar(range(1, len(layers_configs) + 1), scores)
	plt.xticks(range(1, len(layers_configs) + 1), layers_configs, rotation=45)
	plt.xlabel('Layers')
	plt.ylabel('Score')
	plt.title('Score vs configuración de capas.\n {} muestras de entrenamiento y {} muestras de validación'.format(len(leaf)-total,
	                                                                                                    total))
	plt.savefig('score_vs_layers.png')


def epochs_experiment(training_data, validation_data, ninputs, noutputs):
	epochs = [50, 100, 500, 1000]
	scores = []
	total = len(validation_data)

	plt.figure()

	for epoch in epochs:
	    # Entrenamiento
	    nn = NeuralNetwork(layers_conf=[ninputs, 2 * ninputs, 2 * ninputs, noutputs],
	                       neuron_type=Sigmoid,
	                       learning_rate=.1)
	    errors = nn.epoch_training(training_data, epoch=epoch)
	    plt.plot(errors, label='epoch: {}'.format(epoch))

	    # Validación
	    yes = 0
	    for input_val_data, expected_output in validation_data:
	        nn_output = aprox_nn_output(nn.feed_forward(input_val_data))
	        if nn_output == expected_output:
	            yes += 1

	    scores.append(score(yes, total))

	plt.xlabel('Epoch')
	plt.ylabel('Error normalizado')
	plt.title('Curva de error normalizado durante entrenamiento.\nLR=0.1. Layers={}'.format([ninputs, 2 * ninputs, 2 * ninputs, noutputs]))
	plt.legend()
	plt.savefig('training_error_epochs_experiment.png')


	plt.figure()
	plt.bar(range(1, len(epochs) + 1), scores)
	plt.xticks(range(1, len(epochs) + 1), epochs)
	plt.xlabel('Epochs')
	plt.ylabel('Score')
	plt.title('Score vs Epochs.\n {} muestras de entrenamiento y {} muestras de validación'.format(len(leaf)-total,
	                                                                                                    total))
	plt.savefig('score_vs_epochs.png')

# Header de los datos

columns = ['class',
           'specimen_number',
           'eccentricity',
           'aspect_ratio',
           'elongation',
           'solidity',
           'stochastic_convexity',
           'isoperimetric_factor',
           'maximal_indentation_depth',
           'lobedness',
           'average_intensity',
           'average_constrast',
           'smoothness',
           'third_moment',
           'uniformity',
           'entropy']

# Cargar datos

leaf = pd.read_csv('leaf.txt', names=columns).drop(columns=['specimen_number'])

classes = set(leaf['class'])

# Dictionary with formatted outputs/classes

classes_labels = {}

for enum, _class in enumerate(classes):
    classes_labels[_class] = [0 if i != enum else 1 for i in range(len(classes))]

training_data = []
validation_data = []

for _class in classes:
    _class_df = leaf.iloc[np.where(leaf['class'] == _class)[0]]
    train_portion = int(len(_class_df) * .75)
    training_data.append(_class_df.iloc[:train_portion])
    validation_data.append(_class_df.iloc[train_portion:])

training_data = pd.concat(training_data)
validation_data = pd.concat(validation_data)

# Compute arrays of encoded classes
training_data_outputs = pd.Series([classes_labels[_class] for _class in training_data['class']])
validation_data_outputs = pd.Series([classes_labels[_class] for _class in validation_data['class']])

formatted_train_data = list(zip(training_data.drop(columns=['class']).values, training_data_outputs))
formatted_validation_data = list(zip(validation_data.drop(columns=['class']).values, validation_data_outputs))

# Neural network for dataset

ninputs = len(columns) - 2
noutputs = len(classes)

# Try neural nsetwork with different learning_rates
if (int(sys.argv[1]) == 1):
	print('LR experiment')
	learning_rate_experiment(formatted_train_data, formatted_validation_data, ninputs, noutputs)

# Try different layer configuration
if (int(sys.argv[1]) == 2):
	print('Layers experiment')
	layers_experiment(formatted_train_data, formatted_validation_data, ninputs, noutputs)

# Try diffent epochs
if (int(sys.argv[1]) == 3):
	print('Epochs experiment')
	epochs_experiment(formatted_train_data, formatted_validation_data, ninputs, noutputs)






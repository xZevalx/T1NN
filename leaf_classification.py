from NeuralNetwork import NeuralNetwork, Sigmoid

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


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

# Try neural network with different learning_rates

lrs = [.1, .3, .5, .7, .8, 1.]
scores = []
total = len(formatted_validation_data)

for lr in lrs:
    # Entrenamiento
    nn = NeuralNetwork(layers_conf=[ninputs, 2 * ninputs, 3 * ninputs, noutputs],
                       neuron_type=Sigmoid,
                       learning_rate=lr)
    errors = nn.epoch_training(formatted_train_data, epoch=100)
    plt.plot(errors)
    plt.xlabel('Epoch')
    plt.ylabel('Error normalizado')
    plt.title('Curva de error normalizado durante entrenamiento. Learning rate {}'.format(lr))
    plt.savefig('training_error_lr={}.png'.format(lr))

    # Validación
    yes = 0
    for input_val_data, expected_output in formatted_validation_data:
        nn_output = aprox_nn_output(nn.feed_forward(input_val_data))
        if nn_output == expected_output:
            yes += 1

    scores.append(score(yes, total))

plt.plot(lrs, scores)
plt.xlabel('Learning rate')
plt.ylabel('Score')
plt.title('Score vs learning rate. {} muestras de entrenamiento y {} muestras de validación'.format(len(leaf)-total,
                                                                                                    total))




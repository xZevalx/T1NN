from NeuralNetwork import NeuralNetwork, Sigmoid

import pandas as pd
import numpy as np

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

nn = NeuralNetwork([ninputs, 3 * ninputs, 3 * ninputs, noutputs], neuron_type=Sigmoid)
errors = nn.epoch_training(formatted_train_data, epoch=300)

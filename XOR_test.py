from NeuralNetwork import NeuralNetwork, Perceptron, Sigmoid

import numpy as np
import matplotlib.pyplot as plt


def generate_xor_dataset(n):
    dataset = []
    for i in range(n):
        point = np.random.uniform(-1, 1, 2)
        if (point[0] < 0 and point[1] < 0) or (point[0] > 0 and point[1] > 0):
            label = 0
        else:
            label = 1

        dataset.append([point, [label]])

    return dataset


nn = NeuralNetwork([2, 10, 1], neuron_type=Sigmoid, learning_rate=0.1)  # Only input layer
XOR_dataset = [[[1, 1], [0]],
               [[1, 0], [1]],
               [[0, 1], [1]],
               [[0, 0], [0]]]

# XOR_dataset = generate_xor_dataset(50)

errors = nn.epoch_training(XOR_dataset, 1000)

print("0 0 is 0, predicted {}".format(nn.feed_forward([0, 0])))
print("0 1 is 1, predicted {}".format(nn.feed_forward([0, 1])))
print("1 0 is 1, predicted {}".format(nn.feed_forward([1, 0])))
print("1 1 is 0, predicted {}".format(nn.feed_forward([1, 1])))

plt.plot(errors)
plt.show()

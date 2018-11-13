import unittest

from Neuron import InputNeuron, Perceptron, Sigmoid
from numpy import exp


class NeuronTests(unittest.TestCase):
    
    def test_input_neuron(self):
    	# arrange
    	neuron = InputNeuron()
    	# act
    	# assert
    	self.assertEqual(neuron.ninputs, 1)
    	self.assertEqual(neuron.weights, [1])
    	self.assertEqual(neuron.guess(5), 5)

    def test_perceptron(self):
    	# arrange
    	neuron = Perceptron(ninputs=2, weights=[2, .5], learning_rate=.1)
    	# act
    	guess = neuron.guess([1, 1])
    	# assert
    	self.assertEqual(neuron.ninputs, 2)
    	self.assertEqual(len(neuron.weights), 2)
    	self.assertLessEqual(guess, 1)
    	self.assertGreaterEqual(guess, 0)
    	self.assertEqual(neuron.activation_function(45), 1)
    	self.assertEqual(neuron.activation_function(-30), 0)

    def test_sigmoid(self):
    	# arrange
    	neuron = Sigmoid(ninputs=2, weights=[2, .5], learning_rate=.1)
    	# act
    	guess = neuron.guess([1, 1])
    	# assert
    	self.assertEqual(neuron.ninputs, 2)
    	self.assertEqual(len(neuron.weights), 2)
    	self.assertLessEqual(guess, 1)
    	self.assertGreaterEqual(guess, 0)
    	self.assertAlmostEqual(neuron.activation_function(0.5), 1/(1+exp(-0.5)))

class LayerTests(unittest.TestCase):
    pass


class NetworkTests(unittest.TestCase):
    pass


if __name__=='__main__':
	unittest.main()
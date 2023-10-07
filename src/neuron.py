import numpy as np


class Neuron:
    def __init__(self, weights, bias, activation_function):
        self.weights = weights
        self.bias = bias
        self.activation_function = activation_function

    # returns activation_function(weighted sum of inputs)
    def activate(self, input):
        return self.activation_function(np.dot(input, self.weights) + self.bias)

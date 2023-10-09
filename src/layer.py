import numpy as np

from src.neuron import Neuron

DEFAULT_BIAS = 0.15


class Layer:
    def __init__(self, neuron_size, input_size, activation_function):
        self.neurons = [
            Neuron(
                weights=self.initialize_weights(input_size=input_size),
                bias=DEFAULT_BIAS,
                activation_function=activation_function,
            )
            for i in range(neuron_size)
        ]

    def initialize_weights(self, input_size):
        # Let's do a normal distribution for now
        return np.random.randn(input_size)

    def forward(self, input):
        # Note: horribly inefficient
        output_layer = np.array([])

        for neuron in self.neurons:
            output_layer = np.append(output_layer, neuron.activate(input=input))

        return output_layer

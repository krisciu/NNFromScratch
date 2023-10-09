import pytest
import numpy as np

from src.functions import relu
from src.layer import Layer

# Note: will have to change in future
TEST_BIAS = 0.15


def test_neuron_initialization():
    layer = Layer(neuron_size=3, input_size=2, activation_function=relu)
    assert len(layer.neurons) == 3
    for neuron in layer.neurons:
        assert len(neuron.weights) == 2
        assert neuron.bias == TEST_BIAS
        assert neuron.activation_function == relu


def test_forward_pass():
    class MockNeuron:
        def activate(self, input):
            return np.sum(input)

    layer = Layer(neuron_size=3, input_size=2, activation_function=MockNeuron.activate)
    layer.neurons = [MockNeuron() for i in range(3)]

    input_data = np.array([1, 2])
    output = layer.forward(input_data)

    # Since MockNeuron just sums the input, each neuron should output 3
    np.testing.assert_array_equal(output, np.array([3, 3, 3]))

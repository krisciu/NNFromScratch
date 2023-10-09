import numpy as np
import pytest
from src.functions import relu, sigmoid

from src.neuron import Neuron

import numpy as np

# test neuron class
def test_activation_sigmoid():
    neuron = Neuron(np.array([1.0, 2.0]), 1.0, sigmoid)
    assert neuron.activate(np.array([1.0, 1.0])) == pytest.approx(sigmoid(4.0), 0.01)


def test_activation_relu():
    neuron = Neuron(np.array([1.0, 2.0]), 1.0, relu)
    assert neuron.activate(np.array([1.0, 1.0])) == pytest.approx(relu(4.0), 0.01)


def test_activation_negative_weights():
    neuron = Neuron(np.array([-1.0, -2.0]), 1.0, relu)
    assert neuron.activate(np.array([1.0, 1.0])) == pytest.approx(relu(-2.0), 0.01)

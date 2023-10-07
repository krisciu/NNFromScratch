import pytest
import math
from src.functions import relu, sigmoid


# test Sigmoid
def test_sigmoid_zero():
    assert sigmoid(0) == pytest.approx(0.5, abs=1e-9)

def test_sigmoid_large_positive():
    assert sigmoid(100) == pytest.approx(1.0, abs=1e-9)

def test_sigmoid_negative():
    assert sigmoid(-100) == pytest.approx(0.0, abs=1e-9)

#Test ReLU
def test_relu_zero():
    assert relu(0) == 0

def test_relu_negative():
    assert relu(-10000.234234) == 0

def test_relu_positive():
    val = 10000.00234234
    assert relu(val) == val

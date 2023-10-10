import math
import pytest
from src.functions import binary_cross_entropy_loss, derive, relu, sigmoid
import numpy as np
from numpy.testing import assert_almost_equal
from sklearn.metrics import log_loss

LABELS = [0, 1]


# test Sigmoid
def test_sigmoid_zero():
    assert sigmoid(0) == pytest.approx(0.5, abs=1e-9)


def test_sigmoid_large_positive():
    assert sigmoid(100) == pytest.approx(1.0, abs=1e-9)


def test_sigmoid_negative():
    assert sigmoid(-100) == pytest.approx(0.0, abs=1e-9)


# Test ReLU
def test_relu_zero():
    assert relu(0) == 0


def test_relu_negative():
    assert relu(-10000.234234) == 0


def test_relu_positive():
    val = 10000.00234234
    assert relu(val) == val


#test binary CEL

def test_single_value_case_1():
    y_true = np.array([1])
    y_pred = np.array([0.9])
    assert_almost_equal(
        binary_cross_entropy_loss(y_true, y_pred),
        log_loss(y_true, y_pred, labels=LABELS),
    )


def test_single_value_case_2():
    y_true = np.array([0])
    y_pred = np.array([0.1])
    assert_almost_equal(
        binary_cross_entropy_loss(y_true, y_pred),
        log_loss(y_true, y_pred, labels=LABELS),
    )


def test_multiple_values_case_1():
    y_true = np.array([1, 0, 1, 0])
    y_pred = np.array([0.9, 0.1, 0.8, 0.2])
    assert_almost_equal(
        binary_cross_entropy_loss(y_true, y_pred),
        log_loss(y_true, y_pred, labels=LABELS),
    )


def test_all_true_labels_are_1():
    y_true = np.array([1, 1, 1, 1])
    y_pred = np.array([0.9, 0.8, 0.95, 0.85])
    assert_almost_equal(
        binary_cross_entropy_loss(y_true, y_pred),
        log_loss(y_true, y_pred, labels=LABELS),
    )


def test_all_true_labels_are_0():
    y_true = np.array([0, 0, 0, 0])
    y_pred = np.array([0.1, 0.2, 0.05, 0.15])
    assert_almost_equal(
        binary_cross_entropy_loss(y_true, y_pred),
        log_loss(y_true, y_pred, labels=LABELS),
    )


def test_edge_case_probability_0():
    y_true = np.array([0])
    y_pred = np.array([0])
    assert_almost_equal(binary_cross_entropy_loss(y_true, y_pred), 0)


def test_edge_case_probability_1():
    y_true = np.array([1])
    y_pred = np.array([1])
    assert_almost_equal(binary_cross_entropy_loss(y_true, y_pred), 0)


# Test derive

#  helpers for derive
def f1(x):
    return x**2

def f2(x):
    return math.sin(x)

def f3(x):
    return math.exp(x)

def test_derive_square():
    # Derivative of x^2 at x = 2 is 4
    assert abs(derive(f1, 2) - 4) < 1e-9

def test_derive_sin():
    # Derivative of sin(x) at x = π/2 is 0
    assert abs(derive(f2, math.pi / 2)) < 1e-9

def test_derive_exp():
    # Derivative of exp(x) at x = 1 is exp(1)
    assert abs(derive(f3, 1) - math.exp(1)) < 1e-9
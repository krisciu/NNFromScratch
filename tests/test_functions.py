import unittest
import math

from src.functions import sigmoid


def test_sigmoid_zero():
    assert sigmoid(0) == 0.5

def test_sigmoid_large_positive():
    assert sigmoid(100) == 1.0

def test_sigmoid_negative():
    assert sigmoid(-100) == 0.0

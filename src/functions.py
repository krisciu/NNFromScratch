import math


def sigmoid(x) -> float:
    return 1 / (1 + math.exp(-x))


def relu(x) -> float:
    return max(x, 0)

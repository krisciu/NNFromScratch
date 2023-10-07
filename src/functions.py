import numpy as np
import pandas as pd
import math


def sigmoid(x) -> float:
    return 1 / (1 + math.e ** (-x))

def relu(x) -> float:
    return max(x,0)
import numpy as np
import pandas as pd
import math


def sigmoid(x):
    return 1 / (1 + math.e ** (-x))

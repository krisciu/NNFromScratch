import math
import numpy as np

# to avoid impossibility
EPSILON = 1e-15

SMALL_VAL = 1e-5


def sigmoid(x) -> float:
    return 1 / (1 + math.exp(-x))


def relu(x) -> float:
    return max(x, 0)


def binary_cross_entropy_loss(expected_values, predictions) -> float:
    clipped_predictions = np.clip(predictions, EPSILON, (1 - EPSILON))

    return np.mean(
        -(
            expected_values * np.log(clipped_predictions)
            + (1 - expected_values) * np.log(1 - clipped_predictions)
        )
    )


def derive(f, x):
    return (f(x + SMALL_VAL) - f(x - SMALL_VAL)) / (2 * SMALL_VAL)

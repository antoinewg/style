import numpy as np


def roll_back(x, means):
    x = np.copy(x[0])
    x = x + means

    x = x[::-1]
    x = np.swapaxes(np.swapaxes(x, 0, 1), 1, 2)

    x = np.clip(x, 0, 255).astype("uint8")
    return x

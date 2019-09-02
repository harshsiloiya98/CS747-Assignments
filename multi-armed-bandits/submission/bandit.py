import numpy as np

def pull_arm(prob):
    if (np.random.rand() <= prob):
        return 1.
    else:
        return 0.
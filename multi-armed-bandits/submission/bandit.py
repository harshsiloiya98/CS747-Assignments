import numpy as np

# simulates a Bernoulli bandit arm
def pull_arm(prob):
    if (np.random.rand() <= prob):
        return 1.
    else:
        return 0.
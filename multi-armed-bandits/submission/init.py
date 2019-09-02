import numpy as np
import sys
from banditAlgorithms import run_algorithm

# remove before submitting
def run_as_function(banditFile, algorithm, seed, epsilon, horizon):
    np.random.seed(seed)
    mean_rewards = []
    with open(banditFile, 'r') as f:
        for line in f:
            mean_rewards.append(float(line))
    regret = run_algorithm(mean_rewards, algorithm, epsilon, horizon)
    output_string = [banditFile, algorithm, str(seed), str(epsilon), str(horizon), str(regret)]
    output_string = ", ".join(output_string)
    print(output_string)

if __name__ == "__main__":
    banditFile = sys.argv[1]
    algorithm = sys.argv[2]
    seed = int(sys.argv[3])
    epsilon = float(sys.argv[4])
    horizon = int(sys.argv[5])
    np.random.seed(seed)
    mean_rewards = []
    with open(banditFile, 'r') as f:
        for line in f:
            mean_rewards.append(float(line))
    regret = run_algorithm(mean_rewards, algorithm, epsilon, horizon)
    output_string = [banditFile, algorithm, str(seed), str(epsilon), str(horizon), str(regret)]
    output_string = ", ".join(output_string)
    print(output_string)
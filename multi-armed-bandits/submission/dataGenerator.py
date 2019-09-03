import numpy as np
from init import run_as_function

instances = ["../instances/i-1.txt", "../instances/i-2.txt", "../instances/i-3.txt"]
algorithms = ["round-robin", "epsilon-greedy", "ucb", "kl-ucb", "thompson-sampling"]
epsilons = [0.002, 0.02, 0.2]
horizons = [50, 200, 800, 3200, 12800, 51200, 204800]

for seed in range(50):
    for horizon in horizons:
        for algorithm in algorithms:
            for instance in instances:
                if (algorithm == "epsilon-greedy"):
                    for epsilon in epsilons:
                        run_as_function(instance, algorithm, seed, epsilon, horizon)
                else:
                    run_as_function(instance, algorithm, seed, 0, horizon)
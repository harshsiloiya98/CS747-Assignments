import sys
import numpy as np
from mdpAlgorithms import run_algorithm

if __name__ == "__main__":
    mdpfile = sys.argv[1]
    algorithm = sys.argv[2]
    numStates = 0
    numActions = 0
    discount = 0.0
    mdpType = ""
    rewards = []
    transition = []
    # reading from the MDP file
    with open(mdpfile, 'r') as f:
        i = 0
        for line in f:
            if (i == 0):
                numStates = int(line.strip())
            elif (i == 1):
                numActions = int(line.strip())
            elif (i >= 2 and i < numStates * numActions + 2):
                tokens = line.strip().split('\t')
                tokens = list(map(float, tokens))
                rewards.extend(tokens)
            elif (i >= numStates * numActions + 2 and i < 2 * numStates * numActions + 2):
                tokens = line.strip().split('\t')
                tokens = list(map(float, tokens))
                transition.extend(tokens)
            elif (i == 2 + 2 * numStates * numActions):
                discount = float(line.strip())
            elif (i == 3 + 2 * numStates * numActions):
                mdpType = line.strip()
            i += 1
    transition = np.reshape(transition, (numStates, numActions, numStates))
    rewards = np.reshape(rewards, (numStates, numActions, numStates))
    policy, value = run_algorithm(numStates, numActions, rewards, transition, discount, mdpType, algorithm)
    for state in range(numStates):
        print(value[state][0], policy[state][0])
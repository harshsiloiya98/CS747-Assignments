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
    rewards_temp = []
    transition_temp = []
    # reading from the MDP file
    with open(mdpfile, 'r') as f:
        i = 0
        for line in f:
            if (i == 0):
                numStates = int(line.strip())
            elif (i == 1):
                numActions = int(line.strip())
                transition = np.zeros((numStates, numActions, numStates), dtype = float)
                rewards = np.zeros((numStates, numActions, numStates), dtype = float)
            elif (i >= 2 and i < numStates * numActions + 2):
                tokens = line.strip().split('\t')
                tokens = list(map(float, tokens))
                rewards_temp.extend(tokens)
            elif (i >= numStates * numActions + 2 and i < 2 * numStates * numActions + 2):
                tokens = line.strip().split('\t')
                tokens = list(map(float, tokens))
                transition_temp.extend(tokens)
            elif (i == 2 + 2 * numStates * numActions):
                discount = float(line.strip())
            elif (i == 3 + 2 * numStates * numActions):
                mdpType = line.strip()
            i += 1
    idx = 0
    # storing the transition and reward function in a 3D array
    for i in range(numStates):
        for j in range(numActions):
            for k in range(numStates):
                transition[i][j][k] = transition_temp[idx]
                rewards[i][j][k] = rewards_temp[idx]
                idx += 1
    policy, value = run_algorithm(numStates, numActions, rewards, transition, discount, mdpType, algorithm)
    for state in range(numStates):
        print(value[state][0], policy[state][0])
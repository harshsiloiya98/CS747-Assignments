import sys
import numpy as np

if __name__ == "__main__":
    fileName = sys.argv[1]
    numStates = 0
    numActions = 0
    discount = 0.
    trajectory = []
    with open(fileName, 'r') as f:
        i = 0
        for line in f:
            if (i == 0):
                numStates = int(line.strip())
            elif (i == 1):
                numActions = int(line.strip())
            elif (i == 2):
                discount = float(line.strip())
            i += 1
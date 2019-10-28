import sys
from estimator import TD_Zero

if __name__ == "__main__":
    fileName = sys.argv[1]
    numStates = 0
    numActions = 0
    discount = 0.
    trajectory = []
    lastState = 0
    with open(fileName, 'r') as f:
        i = 0
        for line in f:
            if (i == 0):
                numStates = int(line.strip())
                i += 1
            elif (i == 1):
                numActions = int(line.strip())
                i += 1
            elif (i == 2):
                discount = float(line.strip())
                i += 1
            else:
                try:
                    [state, action, reward] = line.strip().split('\t')
                    state = int(state)
                    action = int(action)
                    reward = float(reward)
                    trajectory.append([state, action, reward])
                except ValueError:
                    lastState = int(line.strip())
    estimateV = TD_Zero(numStates, numActions, discount, trajectory, lastState)
    for v in estimateV:
       print(v)
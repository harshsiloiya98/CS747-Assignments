import numpy as np

def TD_Zero(numStates, numActions, discount, trajectory, lastState):
    V = [0] * numStates
    a = 0.015
    T = len(trajectory)
    for t in range(T):
        currentState = trajectory[t][0]
        reward = trajectory[t][2]
        if (t == T - 1):
            nextState = lastState
        else:
            nextState = trajectory[t + 1][0]
        V[currentState] += a * (reward + discount * V[nextState] - V[currentState])
    return V

def TD_Lambda(numStates, numActions, discount, trajectory, lastState):
    V = np.zeros((numStates, ))
    eT = np.zeros((numStates, ))
    l = 0.96
    a = 0.02
    T = len(trajectory)
    for t in range(T):
        currentState = trajectory[t][0]
        reward = trajectory[t][2]
        if (t == T - 1):
            nextState = lastState
        else:
            nextState = trajectory[t + 1][0]
        eT = discount * l * eT
        # every-visit MC
        eT[currentState] += 1.0
        V += a * (reward + discount * V[nextState] - V[currentState]) * eT
    return V
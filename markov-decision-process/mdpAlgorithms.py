from pulp import LpMinimize, LpProblem, lpSum, LpVariable
import numpy as np

# called from init.py, runs an algorithm according to the 'algorithm' parameter provided
def run_algorithm(numStates, numActions, rewards, transition, discount, mdpType, algorithm):
    if (algorithm == "lp"):
        return LinearProgramming(numStates, numActions, rewards, transition, discount, mdpType)
    elif (algorithm == "hpi"):
        return PolicyIteration(numStates, numActions, rewards, transition, discount, mdpType)
    else:
        print("Enter valid algorithm!\n")
        exit()

# finds optimal value function using linear programming
def LinearProgramming(numStates, numActions, rewards, transition, discount, mdpType):
    mdpProblem = LpProblem("MDP", LpMinimize)
    dictVar = LpVariable.dict("value_function", range(numStates))
    # adding objective function
    mdpProblem += lpSum([dictVar[i] for i in range(numStates)])
    # adding constraints
    V = np.zeros((numStates, 1), dtype = LpVariable)
    for i in range(numStates):
        V[i][0] = dictVar[i]
    for state in range(numStates):
        for action in range(numActions):
            lowerBound = Bellman(transition[state][action], rewards[state][action], V, discount)
            mdpProblem += dictVar[state] >= lowerBound
    # additional constraint for episodic MDPs
    if (mdpType == "episodic"):
        mdpProblem += (dictVar[numStates - 1] == 0)
    # solve the linear programming problem
    mdpProblem.solve()
    V = np.zeros((numStates, 1))
    for i in range(numStates):
        V[i][0] = dictVar[i].varValue
    # getting the optimum policy
    P = np.zeros((numStates, 1), dtype = int)
    tmp = np.zeros((numActions, ))
    for state in range(numStates):
        for action in range(numActions):
            tmp[action] = Bellman(transition[state][action], rewards[state][action], V, discount)
        P[state][0] = np.argmax(tmp)
    return P, V

# finds optimal value function using policy iteration
def PolicyIteration(numStates, numActions, rewards, transition, discount, mdpType):
    P = np.zeros((numStates, 1), dtype = int)          # Policy
    V = np.zeros((numStates, 1))                       # Value
    tmp = np.zeros((numActions, ))
    epsilon = 1e-10
    # random initialization of policy
    for i in range(numStates):
        P[i][0] = np.random.choice(range(numActions))
    while True:
        delta = 0
        old_V = np.copy(V)
        old_P = np.copy(P)
        # policy evaluation
        for state in range(numStates):
            if (mdpType == "episodic" and state == numStates - 1):
                V[state][0] = 0
            else:
                p = P[state][0]
                V[state][0] = Bellman(transition[state][p], rewards[state][p], V, discount)
            delta = max(delta, abs(V[state][0] - old_V[state][0]))
        if (delta < epsilon):
            # policy improvement
            isPolicyStable = True
            for state in range(numStates):
                for action in range(numActions):
                    tmp[action] = Bellman(transition[state][action], rewards[state][action], V, discount)
                P[state][0] = np.argmax(tmp)
                if (P[state][0] != old_P[state][0]):
                    isPolicyStable = False
                    break
            if (isPolicyStable):
                break
    return P, V

def Bellman(T, R, V, gamma):
    numStates = np.size(T)
    T = np.reshape(T, (numStates, 1))
    R = np.reshape(R, (numStates, 1))
    R = R + gamma * V
    result = np.matmul(np.transpose(T), R)
    return result[0][0]
import pulp
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
    mdpProblem = pulp.LpProblem("MDP", pulp.LpMinimize)
    dictVar = pulp.LpVariable.dict("value_function", range(numStates))
    # adding objective function
    mdpProblem += pulp.lpSum([dictVar[i] for i in range(numStates)])
    # adding constraints
    for state1 in range(numStates):
        for action in range(numActions):
            lowerBound = 0
            for state2 in range(numStates):
                lowerBound += transition[state1][action][state2] * (rewards[state1][action][state2] + discount * dictVar[state2])
            mdpProblem += dictVar[state1] >= lowerBound
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
            t = np.reshape(transition[state][action], (numStates, 1))
            r = np.reshape(rewards[state][action], (numStates, 1))
            r = r + discount * V
            result = np.matmul(np.transpose(t), r)
            tmp[action] = result[0][0]
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
                t = np.reshape(transition[state][p], (numStates, 1))
                r = np.reshape(rewards[state][p], (numStates, 1))
                r = r + discount * V
                result = np.matmul(np.transpose(t), r)
                V[state][0] = result[0][0]
            delta = max(delta, abs(V[state][0] - old_V[state][0]))
        if (delta < epsilon):
            # policy improvement
            isPolicyStable = True
            for state in range(numStates):
                for action in range(numActions):
                    t = np.reshape(transition[state][action], (numStates, 1))
                    r = np.reshape(rewards[state][action], (numStates, 1))
                    r = r + discount * V
                    result = np.matmul(np.transpose(t), r)
                    tmp[action] = result[0][0]
                P[state][0] = np.argmax(tmp)
                if (P[state][0] != old_P[state][0]):
                    isPolicyStable = False
                    break
            if (isPolicyStable):
                break
    return P, V
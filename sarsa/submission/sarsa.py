import numpy as np

def actionProbabilities(Q, epsilon, state, numActions):
    # getting probabilities for each action using epsilon-greedy method
    P = np.ones(numActions) * epsilon / numActions       # exploration
    bestAc = np.argmax(Q[state])
    P[bestAc] += 1.0 - epsilon                           # exploitation
    return P

def SARSA(transitions, numStates, numActions, discount, start, end):
    Q = np.zeros((numStates, numActions))
    alpha = 0.5
    epsilon = 0.1
    numEpisodes = 200
    numSteps = 10000
    x = np.arange(numEpisodes)
    y = np.zeros(numEpisodes)
    for episode in range(numEpisodes):
        state = start
        P = actionProbabilities(Q, epsilon, state, numActions)
        action = np.random.choice(range(numActions), p = P)
        for step in range(numSteps):
            epsilon_t = 1 / (step + 1)
            nextState = transitions[state][action][0]
            P = actionProbabilities(Q, epsilon_t, nextState, numActions)
            nextAction = np.random.choice(range(numActions), p = P)
            reward = transitions[state][action][1]
            Q[state][action] += alpha * (reward + discount * Q[nextState][nextAction] - Q[state][action])
            if (state == end):
                y[episode] = step
                break
            state = nextState
            action = nextAction
    return x, y
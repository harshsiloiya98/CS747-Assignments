import numpy as np

def selectAction(Q, epsilon, state, numActions):
    # choosing an action using epsilon-greedy method
    decision = np.random.choice([1, 2], p = [epsilon, 1 - epsilon])
    if (decision == 1):
        ac = np.random.choice(range(numActions))
    else:
        ac = np.argmax(Q[state])
    return ac

def SARSA(seedval, transitions, numStates, numActions, discount, start, end, numEpisodes = 100):
    np.random.seed(seedval)
    Q = np.zeros((numStates, numActions))
    alpha = 0.5
    epsilon = 0.1
    maxSteps = 10000
    x = np.array(range(numEpisodes))
    y = np.zeros((numEpisodes, ))
    for episode in range(numEpisodes):
        state = start
        action = selectAction(Q, epsilon, state, numActions)
        step = 1
        while (step <= maxSteps):
            nextState = transitions[state][action][0]
            if (not (type(nextState) is int or type(nextState) is np.int64)):
                probs = np.ones(len(nextState)) *  1.0 / len(nextState)
                nextState = np.random.choice(nextState, p = probs)
            nextAction = selectAction(Q, epsilon, nextState, numActions)
            reward = transitions[state][action][1]
            if (nextState == end):
                Q[state][action] += alpha * (reward - Q[state][action])
                if (episode > 0):
                    y[episode] = step + y[episode - 1]
                else:
                    y[episode] = step
                break
            Q[state][action] += alpha * (reward + discount * Q[nextState][nextAction] - Q[state][action])
            state = nextState
            action = nextAction
            step += 1
    return x, y
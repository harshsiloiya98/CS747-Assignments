import sys
import numpy as np
import matplotlib.pyplot as plt
from windygridworld import WindyGridworld, WindyGridworldK, WindyGridworldS
from sarsa import SARSA

if __name__ == "__main__":
    graphTitle = ""
    try:
        arg = sys.argv[1]
    except IndexError:
        print("Input format: ./run.sh <argument>")
        exit()
    if (arg == "default"):
        wG = WindyGridworld()
        graphTitle = "SARSA for regular Windy Gridworld"
    elif (arg == "kings"):
        wG = WindyGridworldK()
        graphTitle = "SARSA for Windy Gridworld with King's moves"
    elif (arg == "stochastic"):
        wG = WindyGridworldS()
        graphTitle = "SARSA for Windy Gridworld with Stochasticity"
    numStates = wG.getNumStates()
    numActions = wG.getNumActions()
    discount = wG.getDiscount()
    transitions = wG.getTransition()
    start = wG.getStartState()
    end = wG.getEndState()
    numEpisodes = 200
    yMean = np.zeros((numEpisodes, ))
    seedvals = [30, 46, 73, 92, 29, 65, 8, 50, 11, 81]
    for seedval in seedvals:
        x, y = SARSA(seedval, transitions, numStates, numActions, discount, start, end, numEpisodes)
        yMean += y
    yMean /= len(seedvals)
    plt.plot(yMean, x)
    plt.xlabel("Time Steps")
    plt.ylabel("Episodes")
    plt.title(graphTitle)
    plt.show()
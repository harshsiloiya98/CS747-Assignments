import sys
import numpy as np
from os.path import exists
import matplotlib.pyplot as plt
from windygridworld import WindyGridworld
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
    x, y = SARSA(transitions, numStates, numActions, discount, start, end)
    plt.plot(x, y)
    plt.xlabel("Time Steps")
    plt.ylabel("Episodes")
    plt.title(graphTitle)
    plt.show()
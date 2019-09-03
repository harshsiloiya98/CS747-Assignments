import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


df = pd.read_csv("output.txt", names = ['instance', 'algorithm', 'randomSeed', 'epsilon', 'horizon', 'REG'], skipinitialspace=True)
a = df.groupby(["instance","algorithm","horizon","epsilon"]).mean()
horizon = np.log(np.array([50,200,800,3200,12800,51200,204800]))
for instance in ["../instances/i-1.txt","../instances/i-2.txt","../instances/i-3.txt"]:
# for instance in ["../instances/i-3.txt"]:
	for algorithm in ["round-robin", "epsilon-greedy", "ucb", "kl-ucb", "thompson-sampling"]:
		if algorithm == "epsilon-greedy":
			for epsilon in [0.002, 0.02, 0.2]:
				reg_array = a.query("instance ==" + "'" + instance + "'" + " & algorithm == " + "'"+ algorithm + "'" + " & epsilon == " + str(epsilon))["REG"].tolist()
				plt.plot(horizon, reg_array)
		else:
			# print(instance, algorithm)
			reg_array = a.query("instance ==" + "'" + instance + "'" + " & algorithm == " + "'"  + algorithm + "'")["REG"].tolist()
			# print()
			plt.plot(horizon, reg_array)
	plt.legend(['round-robin', 'epsilon-greedy with epsilon=0.002', 'epsilon-greedy with epsilon=0.02', 'epsilon-greedy with epsilon=0.2','ucb','kl-ucb','thompson-sampling'], loc='upper left')
	plt.xlabel("log of horizon")
	plt.ylabel("Regret")
	plt.title("Instance : " + instance)
	plt.show()

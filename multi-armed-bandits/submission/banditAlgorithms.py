import numpy as np
from bandit import pull_arm

# calculates the KL divergence between 2 Bernoulli random variables
def KLDivergence(x, y):
    if (x == 0):
        return np.log(1 / (1 - y))
    elif (x == 1):
        return np.log(1 / y)
    else:
        return x * np.log(x / y) + (1 - x) * np.log((1 - x) / (1 - y))

# condition used in KL-UCB algorithm
def find_max_q(p, u, t):
    q = 0
    RHS = (np.log(t) + 3 * np.log(np.log(t))) / u
    i = 0.1
    while (i < 1):
        LHS = KLDivergence(p, i)
        if (LHS <= RHS):
            q = i
        i += 0.05
    return q

# called from init.py, runs an algorithm according to the 'algorithm' parameter provided
def run_algorithm(mean_rewards, algorithm, epsilon, horizon):
    if (algorithm == "round-robin"):
        return round_robin(mean_rewards, horizon)
    elif (algorithm == "epsilon-greedy"):
        return epsilon_greedy(mean_rewards, epsilon, horizon)
    elif (algorithm == "ucb"):
        return ucb(mean_rewards, horizon)
    elif (algorithm == "kl-ucb"):
        return kl_ucb(mean_rewards, horizon)
    elif (algorithm == "thompson-sampling"):
        return thompson_sampling(mean_rewards, horizon)
    else:
        print("Enter valid algorithm!\n")
        exit()

def round_robin(mean_rewards, horizon):
    # pulls all arms in a round-robin manner
    num_arms = len(mean_rewards)
    expected_reward = 0
    for i in range(horizon):
        arm_idx = i % num_arms
        reward = pull_arm(mean_rewards[arm_idx])
        expected_reward += reward
    ideal_reward = max(mean_rewards) * horizon
    regret = round(ideal_reward - expected_reward, 3)
    return regret

def epsilon_greedy(mean_rewards, epsilon, horizon):
    num_arms = len(mean_rewards)
    calculated_means = [0] * num_arms
    num_pulls = [0] * num_arms
    expected_reward = 0
    for i in range(horizon):
        arm_idx = 0
        # makes a decision for which arm to pull
        decision = np.random.choice([1, 2], p = [epsilon, 1 - epsilon])
        if (decision == 1):
            # randomly pick an arm to pull
            arm_idx = mean_rewards.index(np.random.choice(mean_rewards))
        else:
            # pick arm with max mean
            arm_idx = np.argmax(calculated_means)
        reward = pull_arm(mean_rewards[arm_idx])
        expected_reward += reward
        # calculates the new mean for the pulled arm
        calculated_means[arm_idx] = (num_pulls[arm_idx] * calculated_means[arm_idx] + reward) / (num_pulls[arm_idx] + 1)
        num_pulls[arm_idx] += 1
    ideal_reward = max(mean_rewards) * horizon
    regret = round(ideal_reward - expected_reward, 3)
    return regret

def ucb(mean_rewards, horizon):
    num_arms = len(mean_rewards)
    calculated_means = [0] * num_arms
    num_pulls = [0] * num_arms
    ucb_list = [0] * num_arms
    expected_reward = 0
    # sampling each arm once
    for arm in range(num_arms):
        reward = pull_arm(mean_rewards[arm])
        num_pulls[arm] += 1
        calculated_means[arm] = reward
    for i in range(num_arms, horizon):
        # calculating the UCBs for each arm
        for arm in range(num_arms):
            p = calculated_means[arm]
            u = num_pulls[arm]
            ucb_list[arm] = p + np.sqrt(2 * np.log(i) / u)
        # picking the arm with maximum UCB
        arm_idx = np.argmax(ucb_list)
        reward = pull_arm(mean_rewards[arm_idx])
        expected_reward += reward
        calculated_means[arm_idx] = (num_pulls[arm_idx] * calculated_means[arm_idx] + reward) / (num_pulls[arm_idx] + 1)
        num_pulls[arm_idx] += 1
    ideal_reward = max(mean_rewards) * horizon
    regret = round(ideal_reward - expected_reward, 3)
    return regret

def kl_ucb(mean_rewards, horizon):
    num_arms = len(mean_rewards)
    calculated_means = [0] * num_arms
    num_pulls = [0] * num_arms
    ucb_list = [0] * num_arms
    expected_reward = 0
    # sampling each arm once
    for arm in range(num_arms):
        reward = pull_arm(mean_rewards[arm])
        num_pulls[arm] += 1
        calculated_means[arm] = reward
    for i in range(num_arms, horizon):
        # calculating the KL-UCBs of the arms
        for arm in range(num_arms):
            p = calculated_means[arm]
            u = num_pulls[arm]
            ucb_list[arm] = find_max_q(calculated_means[arm], num_pulls[arm], i)
        # picking the arm with maximum KL-UCB
        arm_idx = np.argmax(ucb_list)
        reward = pull_arm(mean_rewards[arm_idx])
        expected_reward += reward
        calculated_means[arm_idx] = (num_pulls[arm_idx] * calculated_means[arm_idx] + reward) / (num_pulls[arm_idx] + 1)
        num_pulls[arm_idx] += 1
    ideal_reward = max(mean_rewards) * horizon
    regret = round(ideal_reward - expected_reward, 3)
    return regret

def thompson_sampling(mean_rewards, horizon):
    num_arms = len(mean_rewards)
    expected_reward = 0
    successes = [0] * num_arms
    failures = [0] * num_arms
    betas = [0] * num_arms
    for i in range(horizon):
        for arm in range(num_arms):
           arm_success = successes[arm]
           arm_failure = failures[arm]
           # picks a number from the Beta distribution with 
           # alpha = arm_success + 1, beta = arm_failure + 1
           betas[arm] = np.random.beta(arm_success + 1, arm_failure + 1)
        # picks an arm with the maximum Beta value
        arm_idx = np.argmax(betas)
        reward = pull_arm(mean_rewards[arm_idx])
        if (reward == 0):
            # failure occurs with reward 0
            failures[arm_idx] += 1
        else:
            # success occurs with reward 1
            successes[arm_idx] += 1
        expected_reward += reward
    ideal_reward = max(mean_rewards) * horizon
    regret = round(ideal_reward - expected_reward, 3)
    return regret
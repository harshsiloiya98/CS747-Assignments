import numpy as np
from bandit import pull_arm

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
        decision = np.random.choice([1, 2], p = [epsilon, 1 - epsilon])
        if (decision == 1):
            # randomly pick an arm to pull
            arm_idx = mean_rewards.index(np.random.choice(mean_rewards))
        else:
            # pick arm with max mean
            arm_idx = np.argmax(calculated_means)
        reward = pull_arm(mean_rewards[arm_idx])
        expected_reward += reward
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
        calculated_means[arm] = reward
    for i in range(num_arms, horizon):
        for arm in range(num_arms):
            p = calculated_means[arm]
            u = num_pulls[arm]
            ucb_list[arm] = p + np.sqrt(2 * np.log2(i) / u)
        arm_idx = np.argmax(ucb_list)
        reward = pull_arm(mean_rewards[arm_idx])
        expected_reward += reward
        calculated_means[arm_idx] = (num_pulls[arm_idx] * calculated_means[arm_idx] + reward) / (num_pulls[arm_idx] + 1)
        num_pulls[arm_idx] += 1
    ideal_reward = max(mean_rewards) * horizon
    regret = round(ideal_reward - expected_reward, 3)
    return regret

def kl_ucb(mean_rewards, horizon):
    return 0

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
           betas[arm] = np.random.beta(arm_success + 1, arm_failure + 1)
        arm_idx = np.argmax(betas)
        reward = pull_arm(mean_rewards[arm_idx])
        if (reward == 0):
            failures[arm_idx] += 1
        else:
            successes[arm_idx] += 1
        expected_reward += reward
    ideal_reward = max(mean_rewards) * horizon
    regret = round(ideal_reward - expected_reward, 3)
    return regret
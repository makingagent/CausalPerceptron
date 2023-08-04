import gym
import random
import numpy as np
from scipy.stats import norm
from perceptron import Perceptron

# Environment setup
env = gym.make("CartPole-v1")
observation, info = env.reset(seed=42)

# Network initialization
network = [[Perceptron([]) for _ in range(4)]]
network.append([Perceptron(network[0]) for _ in range(25)])

# Variables initialization
action = env.action_space.sample()
round_cnt, total_len, pre, max_len = 1, 0, 0, 0

# Training loop
for iteration in range(100000000):
    observation, reward, terminated, truncated, info = env.step(action)

    if terminated or truncated:
        # Update the network if the episode ends
        for _ in range(15):
            for i in range(4):
                network[0][i].action = observation[i]
            for i in range(25):
                network[1][i].update()
                network[1][i].cal()
            observation = [200 * random.random() - 100 for _ in range(4)]

        # Reset the environment and the network
        observation, info = env.reset()
        for layer in network:
            for neuron in layer:
                neuron.reset()

        # Track the episode length
        total_len += iteration - pre
        episode_len = iteration - pre
        if episode_len > max_len:
            max_len = episode_len
        print(f'round: {round_cnt} avg len: {total_len/round_cnt} max len: {max_len}')
        pre = iteration
        round_cnt += 1

    # Update the network based on the observation
    for i in range(4):
        network[0][i].action = observation[i]
    for i in range(25):
        network[1][i].update()
        network[1][i].cal()

    # Decide the action based on the network output
    total_action_value = sum(neuron.action for neuron in network[1])
    action = 1 if total_action_value > 12.5 else 0

env.close()

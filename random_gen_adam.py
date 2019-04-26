import gym
import random
import numpy as np
import tflearn
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression
from statistics import mean, median
from collections import Counter

# Learning rate
LR = 1e-3

# Build the environment
env = gym.make('CartPole-v0')
env.reset()
# Number of goal steps - every frame where we're within bounds is a +1
goal_steps = 500
score_requirement = 100
# If you make it too big, you brute force every answer
initial_games = 100000

def some_random_games_first():
	for episode in range(5):
		env.reset()
		for t in range(goal_steps):
			# renders environment every step
			# comment out if you want this to go faster
			env.render()
			# Generates a random action in the environment
			action = env.action_space.sample()
			# observation - pole position, cart position, etc
			# reward - 1 or a 0
			# done - game is over
			observation, reward, done, info = env.step(action)
			if done:
				break

some_random_games_first()
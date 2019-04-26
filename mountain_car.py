# Most neural net research focuses on reinforcement learning or training neural nets with less data more quickly
# However, if you can simulate the environment you can generate large amounts of data, which neural nets like!
# We are using a multi-layer perceptron, feedforward model

# Pip install tflearn
# Pip install gym

# Neural nets produce signals - move left/move right
# game - environment
# agent

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
env = gym.make('MountainCar-v0')
env.reset()
# Number of goal steps - every frame where we're within bounds is a +1
goal_steps = 50
score_requirement = -16.2736044
# If you make it too big, you brute force every answer
initial_games = 10000
print("Action Space:", env.action_space)

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

# some_random_games_first()

# Here we generate our training data
def initial_population():
	# Create variable placeholders
	training_data = []
	scores = []
	accepted_scores = []

	# Go through all of our initial games (currently 10,000)
	for _ in range(initial_games):
		score = 0
		game_memory = []
		prev_observation = []
		for _ in range(goal_steps):
			action = random.randrange(0,2)
			observation, reward, done, info = env.step(action)
			print("Reward: ", reward)

			# If we had a winning game, we save it to game_memory
			if len(prev_observation) > 0:
				# Append your observation and the action you took
				game_memory.append([prev_observation, action])

			# Step to next operation
			prev_observation = observation
			score += reward
			print("Score: ", score)


			if done:
				break

		if score >= score_requirement:
			accepted_scores.append(score)
			for data in game_memory:
				if data[1] == [1]:
					output = [0,1]
				elif data[1] == [0]:
					output = [1,0]
				training_data.append([data[0], output])

		env.reset()
		scores.append(score)

	#We've run through all the games, so we're going to save the training data as np array
	training_data_save = np.array(training_data)
	np.save('saved.npy', training_data_save)

	# Print the mean
	print('Average accepted score: ', mean(accepted_scores))
	# Print the median
	print('Median accepted scores', median(accepted_scores))
	# Print the count
	print(Counter(accepted_scores))

	return training_data

initial_population()

# Now we will build neural network model
def neural_network_model(input_size):
	network = input_data(shape=[None, input_size, 1], name='input')

	# 128, 256, 528 nodes on layers, 5 layers
	# Relu = rectified linear
	network = fully_connected(network, 128, activation = 'relu')
	# 0.8 is the keep rate
	network = dropout(network, 0.8)

	network = fully_connected(network, 256, activation = 'relu')
	network = dropout(network, 0.8)

	network = fully_connected(network, 512, activation = 'relu')
	network = dropout(network, 0.8)

	network = fully_connected(network, 256, activation = 'relu')
	network = dropout(network, 0.8)

	network = fully_connected(network, 128, activation = 'relu')
	network = dropout(network, 0.8)

	# Output layer
	# Takes 2 outputs - this will change depending on what game we're playing
	network = fully_connected(network, 2, activation = 'softmax')
	network = regression(network, optimizer = 'adam', learning_rate = LR, loss = 'categorical_crossentropy', name = 'targets')
	model = tflearn.DNN(network, tensorboard_dir='log')

	return model

# Now we will train the model!

def train_model(training_data, model=False):
	# Training data contains observations and output
	# Need to save and reshape the data
	X = np.array([i[0] for i in training_data]).reshape(-1, len(training_data[0][0]), 1)
	y = [i[1] for i in training_data]

	if not model:
		model = neural_network_model(input_size = len(X[0]))

	# If you use too many epochs, you will overfit
	model.fit({'input':X}, {'targets': y}, n_epoch=3, snapshot_step=500, show_metric=True, run_id='openaistuff')

	return model

# Pretty poor accuracy at this point (~0.6), but that's okay
training_data = initial_population()
model = train_model(training_data)

# Now we play a game!
scores = []
choices = []

for each_game in range(10):
	score = 0
	game_memory = []
	prev_obs = []
	env.reset()
	for _ in range(goal_steps):
		env.render()
		# If we don't see a frame
		if len(prev_obs) == 0:
			action = [random.randrange(0,2)]
		# If we see a frame
		else:
			# We take the zeroeth index
			action = [np.argmax(model.predict(prev_obs.reshape(-1, len(prev_obs), 1))[0])]
		# append the choice so we can make sure it isn't always doing one thing
		choices.append(action)

		new_observation, reward, done, info = env.step(action)
		prev_obs = new_observation
		# We can use this to implement reinforcement learning
		game_memory.append([new_observation, action])
		score += reward
		if done:
			break
	scores.append(score)

print('Average Score', sum(scores)/len(scores))
print('Choice 1: {}, Choice 0: {}'.format(choices.count(1)/len(choices), choices.count(0)/len(choices)))
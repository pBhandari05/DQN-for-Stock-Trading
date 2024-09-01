import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque
import os
import sqlalchemy
import pymysql
import math
import pandas as pd
from wakepy import keep
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt



path = "/Users/parthbhandari/Downloads/DQN/nasdaq_large.csv"
df = pd.read_csv(path)
df = df[['Date', 'symbolid', 'Open', 'High', 'Low', 'Close', 'AdjClose', 'Volume']]
df = df[df.AdjClose != 0]

df['mav20'] = df.AdjClose.rolling(20).mean()
df['mav50'] = df.AdjClose.rolling(50).mean()
df['mav100'] = df.AdjClose.rolling(100).mean()

df.loc[df['symbolid'] != df['symbolid'].shift(20), 'mav20'] = np.nan
df.loc[df['symbolid'] != df['symbolid'].shift(50), 'mav50'] = np.nan
df.loc[df['symbolid'] != df['symbolid'].shift(100), 'mav100'] = np.nan

df.dropna(inplace=True)

df['P2MAV20'] = df.AdjClose / df.mav20
df['P2MAV50'] = df.AdjClose / df.mav50
df['P2MAV100'] = df.AdjClose / df.mav100


fulldf = df[['Date', 'symbolid', 'Open', 'High', 'Low', 'Close', 'AdjClose', 'Volume', 'P2MAV20', 'P2MAV50', 'P2MAV100']]



with keep.running():
	device = torch.device('mps')
	class StockTradingEnv(gym.Env):
		def __init__(self, df):
			super(StockTradingEnv, self).__init__()
			
			# Data (Pandas DataFrame with stock prices)
			self.df = df.reset_index(drop=True)
			self.current_step = 0
			self.balance = 10000  # Initial balance in dollars
			self.shares_held = 0
			self.cost_basis = 0
			self.total_profit = 0

			# Define the action space: 0 = hold, 1 = buy, 2 = sell
			self.action_space = spaces.Discrete(3)

			# Define the observation space
			self.observation_space = spaces.Box(
				low=0, high=np.inf, shape=(3,), dtype=np.float32  # changed from 5 to 3 here due to 3 features only
			)

		def _next_observation(self):
			# Get the last five features of the dataframe
			obs = np.array([
				self.df.loc[self.current_step, 'P2MAV20'],
				self.df.loc[self.current_step, 'P2MAV50'],
				self.df.loc[self.current_step, 'P2MAV100'],
	#            self.df.loc[self.current_step, 'Close'],
	#            self.df.loc[self.current_step, 'Volume'],
			])
			return obs

		def _take_action(self, action):
			
	# change the current price according to whether we are buying or selling. Buy high and sell low
			if action == 1:  # Buy
				current_price = self.df.loc[self.current_step, 'High']	
				if self.balance >= current_price:
					# Buy for 10 percent of value rounded off or for the remaining amount
					shares_add = math.floor(self.balance * 0.1 / current_price)
					self.shares_held += shares_add
					self.balance -= shares_add * current_price
					self.cost_basis = current_price

			elif action == 2:  # Sell
				current_price = self.df.loc[self.current_step, 'Low']	
				if self.shares_held > 0:
					# Sell all shares
					self.balance += current_price * self.shares_held
					self.total_profit += (current_price - self.cost_basis) * self.shares_held
					self.shares_held = 0

		def step(self, action):
			# Execute one time step within the environment
			self._take_action(action)
			self.current_step += 1

			total_value = self.balance + self.shares_held * self.df.loc[self.current_step, 'Low']
			if self.current_step >= len(self.df) - 1:
				done = True
			else:
				done = False

			reward = self.balance + self.shares_held * self.df.loc[self.current_step, 'Close'] - 10000
			obs = self._next_observation()

			return obs, reward, done, {}

		def reset(self):
			# Reset the environment to an initial state
			self.balance = 10000
			self.shares_held = 0
			self.cost_basis = 0
			self.total_profit = 0
			self.current_step = 0

			return self._next_observation()

		def render(self, mode='human'):
			# Render the environment to the screen
			profit = self.balance + self.shares_held * self.df.loc[self.current_step, 'Close'] - 10000
			print(f'Step: {self.current_step}')
			print(f'Balance: {self.balance}')
			print(f'Shares held: {self.shares_held}')
			print(f'Total Profit: {self.total_profit}')
			print(f'Current Profit: {profit}')

	class DQNetwork(nn.Module):
		def __init__(self, input_dim, output_dim):
			super(DQNetwork, self).__init__()
			self.fc1 = nn.Linear(input_dim, 128)
			self.fc2 = nn.Linear(128, 128)
			self.fc3 = nn.Linear(128, output_dim)

		def forward(self, x):
			x = torch.relu(self.fc1(x))
			x = torch.relu(self.fc2(x))
			x = self.fc3(x)
			return x

	class DQNAgent:
		def __init__(self, input_dim, output_dim, lr=0.001, gamma=0.99, epsilon=1.0, epsilon_decay=0.995, epsilon_min=0.01, buffer_size=100000, batch_size=64):
			self.input_dim = input_dim
			self.output_dim = output_dim
			self.gamma = gamma
			self.epsilon = epsilon
			self.epsilon_decay = epsilon_decay
			self.epsilon_min = epsilon_min
			self.batch_size = batch_size
			
			self.memory = deque(maxlen=buffer_size)
			self.q_network = DQNetwork(input_dim, output_dim)
			self.target_network = DQNetwork(input_dim, output_dim)
			self.optimizer = optim.Adam(self.q_network.parameters(), lr=lr)
			
			# Synchronize target network
			self.update_target_network()
			
		def update_target_network(self):
			self.target_network.load_state_dict(self.q_network.state_dict())
		
		def remember(self, state, action, reward, next_state, done):
			self.memory.append((state, action, reward, next_state, done))
		
		def act(self, state):
			if np.random.rand() <= self.epsilon:
				return random.choice(range(self.output_dim))
			state = torch.FloatTensor(state).unsqueeze(0)
			q_values = self.q_network(state)
			return torch.argmax(q_values).item()
		
		def replay(self):
			if len(self.memory) < self.batch_size:
				return
			
			batch = random.sample(self.memory, self.batch_size)
			for state, action, reward, next_state, done in batch:
				state = torch.FloatTensor(state).unsqueeze(0)
				next_state = torch.FloatTensor(next_state).unsqueeze(0)
				
				target = self.q_network(state)
				with torch.no_grad():
					if done:
						target_value = reward
					else:
						target_value = reward + self.gamma * torch.max(self.target_network(next_state))
						
				target[0][action] = target_value
				
				# Perform a gradient descent step
				self.optimizer.zero_grad()
				q_values = self.q_network(state)
				loss = nn.MSELoss()(q_values, target)
				loss.backward()
				self.optimizer.step()

			# Reduce epsilon (exploration rate)
			if self.epsilon > self.epsilon_min:
				self.epsilon *= self.epsilon_decay

	rewards = []
	allsymbols = fulldf.symbolid.unique()

	for symbols in allsymbols:

	# Now subset the data for testing
		test_df = fulldf[fulldf.symbolid == symbols].copy()
		test_df[['Open', 'High', 'Low', 'Close', 'AdjClose', 'Volume']] = test_df[['Open', 'High', 'Low', 'Close', 'AdjClose', 'Volume']].shift(-1)
		test_df.dropna(inplace=True)
		test_df.index = test_df.Date

		# Create a new environment for testing
		env = StockTradingEnv(test_df)

		input_dim = env.observation_space.shape[0]
		output_dim = env.action_space.n

		agent = DQNAgent(input_dim, output_dim)

		agent.q_network.load_state_dict(torch.load('dqn_model.pth'))
		agent.q_network.eval()  # Set to evaluation mode


		# Test loop
		balance_history = []
		stock_value_history = []

		state = env.reset()
		done = False
		total_reward = 0

		while not done:
			action = agent.act(state)  # Use the trained model to predict actions
			next_state, reward, done, _ = env.step(action)
			state = next_state
			total_reward += reward

			# Record balance and stock value for plotting
			current_price = env.df.loc[env.current_step, 'Close']
			stock_value = env.shares_held * current_price
			balance_history.append(env.balance)
			stock_value_history.append(stock_value)

			# Render the environment (optional)
			env.render()

		print(f'Total Reward on test data: {total_reward}')
		rewards.append(total_reward)

		# Plot the balance and stock value over time
		'''plt.figure(figsize=(12, 6))
		plt.plot(balance_history, label='Balance', color='blue')
		plt.plot(stock_value_history, label='Stock Value', color='green')
		plt.plot(np.array(balance_history) + np.array(stock_value_history), label='Total Value', color='orange')
		plt.title('Balance and Stock Value Over Time (Testing)')
		plt.xlabel('Steps')
		plt.ylabel('Value ($)')
		plt.legend()
		plt.grid(True)
		#plt.show()
		plt.savefig('/Users/parthbhandari/Downloads/DQN/Results/' + symbols + '.jpg')
		plt.close()
		print(symbols + ' completed')'''
	
	rewards_df = pd.DataFrame({'Symbol': allsymbols, 'Reward': rewards})
	rewards_df.to_csv('/Users/parthbhandari/Downloads/DQN/Results/rewards.csv')
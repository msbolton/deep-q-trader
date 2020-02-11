from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import models
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import StandardScaler
from datetime import datetime

import pandas
import numpy
import pickle
import os
import re
import argparse
import itertools


# Utils
def make_directory(directory):
  if not os.path.exists(directory):
    os.makedirs(directory)  


def get_data():
  # returns a T x 3 list of stock prices
  # each row is a different stock
  dataframe = pandas.read_csv('aapl_msi_sbux.csv')
  return dataframe.values


def multilayer_perceptron(input_dim, n_action, n_hidden_layers=1, hidden_dim=32):
  # input layer
  input_layer = Input(shape=(input_dim,))
  layers = input_layer

  # hidden layers
  for _ in range(n_hidden_layers):
    layers = Dense(hidden_dim, activation='relu')(layers)
  
  # final layer
  layers = Dense(n_action)(layers)

  # make model
  model = Model(input_layer, layers)

  model.compile(loss='mse', optimizer='adam')
  print((model.summary()))
  return model




class Environment:
  """
  Action: 
    - 0 = sell
    - 1 = hold
    - 2 = buy
  State: 
    - n-shares of stock
    - n-share prices of stock
    - cash value
  """
  def __init__(self, data, initial_investment):
    self.stock_price_history = data
    self.n_step, self.n_stock = self.stock_price_history.shape

    self.initial_investment = initial_investment
    self.current_step = None
    self.stock_owned = None
    self.stock_price = None
    self.cash_value = None

    self.action_space = numpy.arange(3**self.n_stock)

    # action permutations
    # returns nested list e.g.
    # [n...,0]
    # [n...,1]
    # [n...,2]
    # [n...,1,0]
    # [n...,1,1]
    # e.g.
    # 0 = sell
    # 1 = hold
    # 2 = buy
    self.action_list = list(map(list, itertools.product([0, 1, 2], repeat=self.n_stock)))

    # calculate state size
    self.state_dim = self.n_stock * 2 + 1

    self.reset()


  def reset(self):
    self.current_step = 0
    self.stock_owned = numpy.zeros(self.n_stock)
    self.stock_price = self.stock_price_history[self.current_step]
    self.cash_value = self.initial_investment
    return self._get_observation()


  def step(self, action):
    assert action in self.action_space

    # get current value before performing the action
    previous_value = self._get_value()

    # update price, i.e. go to the next day
    self.current_step += 1
    self.stock_price = self.stock_price_history[self.current_step]

    # make the trade
    self._trade(action)

    # get the new value after taking the action
    current_value = self._get_value()

    # reward is the increase in porfolio value
    reward = current_value - previous_value

    # done if we have run out of data
    done = self.current_step == self.n_step - 1

    # store the current value of the portfolio here
    info = {'current_value': current_value}

    # conform to the Gym API
    return self._get_observation(), reward, done, info


  def _get_observation(self):
    observation = numpy.empty(self.state_dim)
    observation[:self.n_stock] = self.stock_owned
    observation[self.n_stock:2*self.n_stock] = self.stock_price
    observation[-1] = self.cash_value
    return observation
    


  def _get_value(self):
    return self.stock_owned.dot(self.stock_price) + self.cash_value


  def _trade(self, action):
    # index the action we want to perform
    # 0 = sell
    # 1 = hold
    # 2 = buy
    # e.g. [0,1,...n] means:
    # sell first stock
    # hold second stock
    # buy nth stock
    action_vector = self.action_list[action]

    # determine which stocks to buy or sell
    buy_index = [] # stores index of stocks we want to buy
    sell_index = [] # stores index of stocks we want to sell
    for index, action in enumerate(action_vector):
      if action == 0:
        sell_index.append(index)
      elif action == 2:
        buy_index.append(index)

    # sell any stocks we want to sell
    # then buy any stocks we want to buy
    if sell_index:
      # to simplify the problem, when we sell, we will sell all shares of that stock
      for index in sell_index:
        self.cash_value += self.stock_price[index] * self.stock_owned[index]
        self.stock_owned[index] = 0
    if buy_index:
        # loop through stock we want to buy, 
        # purchase one at a time until we run out of money
      buyable = True
      while buyable:
        for index in buy_index:
          if self.cash_value > self.stock_price[index]:
            self.stock_owned[index] += 1 # buy one share
            self.cash_value -= self.stock_price[index]
          else:
            buyable = False




class ExperienceReplayBuffer:
  def __init__(self, observation_dim, action_dim, size):
        self.observation_buffer = numpy.zeros([size, observation_dim], dtype=numpy.float32)
        self.next_observation_buffer = numpy.zeros([size, observation_dim], dtype=numpy.float32)
        self.actions_buffer = numpy.zeros(size, dtype=numpy.uint8)
        self.rewards_buffer = numpy.zeros(size, dtype=numpy.float32)
        self.done_buffer = numpy.zeros(size, dtype=numpy.uint8)
        self.pointer, self.size, self.max_size = 0, 0, size

  def store(self, observation, action, reward, next_observation, done):
    self.observation_buffer[self.pointer] = observation
    self.next_observation_buffer[self.pointer] = next_observation
    self.actions_buffer[self.pointer] = action
    self.rewards_buffer[self.pointer] = reward
    self.done_buffer[self.pointer] = done
    self.pointer = (self.pointer+1) % self.max_size
    self.size = min(self.size+1, self.max_size)

  def sample_batch(self, batch_size=32):
    idxs = numpy.random.randint(0, self.size, size=batch_size)
    return dict(states=self.observation_buffer[idxs],
                next_states=self.next_observation_buffer[idxs],
                actions=self.actions_buffer[idxs],
                rewards=self.rewards_buffer[idxs],
                done=self.done_buffer[idxs])



class Agent(object):
  def __init__(self, state_size, action_size):
    self.state_size = state_size
    self.action_size = action_size
    self.memory = ExperienceReplayBuffer(state_size, action_size, size=500)
    self.gamma = 0.95  # discount rate
    self.epsilon = 1.0  # exploration rate
    self.epsilon_min = 0.01
    self.epsilon_decay = 0.995
    self.model = multilayer_perceptron(state_size, action_size)

  def load(self, name):
    self.model.load_weights(name)

  def save(self, name):
    self.model.save_weights(name)    

  def act(self, state):
    if numpy.random.rand() <= self.epsilon:
      return numpy.random.choice(self.action_size)
    act_values = self.model.predict(state)
    return numpy.argmax(act_values[0])  # action

  def replay(self, batch_size=32):
    # first check if replay buffer contains enough data
    if self.memory.size < batch_size:
      return

    # sample a batch of data from the replay memory buffer
    sample = self.memory.sample_batch(batch_size)
    states = sample['states']
    actions = sample['actions']
    rewards = sample['rewards']
    next_states = sample['next_states']
    done = sample['done']

    # Calculate the target Q(s',a)
    target = rewards + self.gamma * numpy.amax(self.model.predict(next_states), axis=1)

    # The value of terminal states is zero
    # so set the target to be the reward only
    target[done] = rewards[done]

    # We only need to update the network for actions actually taken
    # If we set the target equal to predictions, we can change the targets for the actions taken Q(s,a)
    target_full = self.model.predict(states)
    target_full[numpy.arange(batch_size), actions] = target

    # Run one step
    self.model.train_on_batch(states, target_full)

    if self.epsilon > self.epsilon_min:
      self.epsilon *= self.epsilon_decay
    
  def update_replay_memory(self, state, action, reward, next_state, done):
    self.memory.store(state, action, reward, next_state, done)




def get_scaler(environment):
#   returns scaler object from scikit-learn to scale states

  states = []
  for _ in range(environment.n_step):
    action = numpy.random.choice(environment.action_space)
    reward, state, done, info = environment.step(action)
    states.append(state)
    if done:
      break

  scaler = StandardScaler()
  scaler.fit(states)
  return scaler



def play_one_episode(agent, environment, arg):
  # after transforming states are already 1xD
  state = environment.reset()
  state = scaler.transform([state])
  done = False

  while not done:
    action = agent.act(state)
    next_state, reward, done, info = environment.step(action)
    next_state = scaler.transform([next_state])
    if arg == 'train':
      agent.update_replay_memory(state, action, reward, next_state, done)
      agent.replay(batch_size)
    state = next_state

  return info['current_value']



if __name__ == '__main__':

  # config
  models_folder = 'models'
  rewards_folder = 'rewards'
  num_episodes = 2000 # num episodes to run
  batch_size = 32 # batch size for sampling from replay memory
  initial_investment = 10000


  parser = argparse.ArgumentParser()
  parser.add_argument('-m', '--mode', type=str, required=True,
                      help='either "train" or "test"')
  args = parser.parse_args()

  make_directory(models_folder)
  make_directory(rewards_folder)

  data = get_data()
  n_timesteps, n_stocks = data.shape

  n_train = n_timesteps // 2

  train_data = data[:n_train]
  test_data = data[n_train:]

  environment = Environment(train_data, initial_investment)
  state_size = environment.state_dim
  action_size = len(environment.action_space)
  agent = Agent(state_size, action_size)
  scaler = get_scaler(environment)

  # store the final value of the portfolio (end of episode)
  portfolio_value = []

  if args.mode == 'test':
    # then load the previous scaler
    with open(f'{models_folder}/scaler.pkl', 'rb') as file:
      scaler = pickle.load(file)

    # remake the environment with test data
    environment = Environment(test_data, initial_investment)

    # epsilon should not be 1, pure exploration
    # no need to run multiple episodes if epsilon = 0, it's deterministic
    agent.epsilon = 0.01

    # load trained weights
    agent.load(f'{models_folder}/dqn.h5')

  # play the game num_episodes times
  for episode in range(num_episodes):
    currenttime = datetime.now()
    value = play_one_episode(agent, environment, args.mode)
    duration = datetime.now() - currenttime
    print(f"episode: {episode + 1}/{num_episodes}, episode end value: {value:.2f}, duration: {duration}")
    portfolio_value.append(value) # append episode end portfolio value

  # save the weights, dqn, scaler when finished
  if args.mode == 'train':

    agent.save(f'{models_folder}/dqn.h5')
    
    with open(f'{models_folder}/scaler.pkl', 'wb') as file:
      pickle.dump(scaler, file)

  # save portfolio value for each episode
  numpy.save(f'{rewards_folder}/{args.mode}.npy', portfolio_value)

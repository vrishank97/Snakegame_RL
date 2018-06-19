from DQNAgent import DQNAgent
from snakegame import SnakeEnv
import os
'''
NUM_THREADS = '96'

os.environ['MKL_NUM_THREADS'] = NUM_THREADS
os.environ['GOTO_NUM_THREADS'] = NUM_THREADS
os.environ['OMP_NUM_THREADS'] = NUM_THREADS
os.environ['openmp'] = 'True'
'''

env = SnakeEnv(6, 6)
agent = DQNAgent(env)

agent.train(episodes=60000, start_mem=10000, save_iter=10000, epsilon_decay_func="exponential")

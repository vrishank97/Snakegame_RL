from collections import deque
from DQNAgent import DQNAgent
from snakegame import SnakeEnv
import numpy as np


env = SnakeEnv(10,15)
agent = DQNAgent()

episodes = 10000
for e in range(episodes):
        state = env.reset()
        for time_t in range(500):
            action = agent.act(env.state.reshape(1, 1, 10, 15))
            next_state, reward, done = env.step(action)
            # Remember the previous state, action, reward, and done
            agent.remember(state, action, reward, next_state, done)
            # make next_state the new current state for the next frame.
            state = next_state
            if e%100 == 0:
                print(env.state)
            # done becomes True when the game ends
            if env.done:
                # print the score and break out of the loop
                print("episode: {}/{}, score: {}, time: {}"
                      .format(e, episodes, len(env.snake)-6, time_t))
                break
        # train the agent with the experience of the episode
        if e>20:
            agent.replay(32)

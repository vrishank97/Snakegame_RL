from collections import deque
from DQNAgent import DQNAgent
from snakegame import SnakeEnv
import numpy as np

env = SnakeEnv(10,10)
agent = DQNAgent()

episodes = 1000
for e in range(episodes):
        state = env.reset()
        for time_t in range(150):
            state = env.getCurrentState()
            action = agent.act(env.project())
            next_state, reward, done = env.step(action)
            next_state = env.project()
            #print("Step: {}", time_t)
            #print(state, next_state)
            agent.remember(state, action, reward, next_state, done)
            # make next_state the new current state for the next frame.
            state = next_state
            # done becomes True when the game ends
            if e%50 == 0:
                print(state)
                #print(agent.model.predict(env.project().reshape(1, 1, 40, 40)))

            if env.done:
                # print the score and break out of the loop
                print("episode: {}/{}, score: {}, time: {}, epsilon: {}"
                      .format(e, episodes, len(env.snake)-3, time_t, agent.epsilon))
                break
        # train the agent with the experience of the episode
        
        agent.replay(min(100, len(agent.memory)))
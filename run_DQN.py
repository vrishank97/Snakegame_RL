from collections import deque
from DQNAgent import DQNAgent
from snakegame import SnakeEnv
import numpy as np
import h5py
import copy
import time

env = SnakeEnv(8,8)
agent = DQNAgent()
episodes = 10000
avg_score=[]
avg_time=[]
tim = time.time()
for e in range(episodes):
        state = env.reset()
        for time_t in range(400):
            state = env.getCurrentState()
            action = agent.act(env.state)
            next_state, reward, done = env.step(action)
            next_state = env.getCurrentState()
            #print("Step: {}", time_t)
            #print(state, next_state)
            agent.remember(state, action, reward, next_state, done)
            # make next_state the new current state for the next frame.
            state = next_state
            # done becomes True when the game ends
            
            if e%100 == 0:
                print(agent.model.predict(env.state.reshape(1, 1, 8, 8)))
                print(agent.memory[0][0])
            
            if env.done:
                # print the score and break out of the loop
                avg_score.append(len(env.snake)-3)
                avg_time.append(time_t)
                break
        # train the agent with the experience of the episode
        
        agent.replay(min(32, len(agent.memory)))
        if e%100==0:
            print("episode: {}/{}, average score: {}, average time: {}, epsilon: {}"
                      .format(e, episodes, np.array(avg_score).mean(), np.array(avg_time).mean(), agent.epsilon))
            avg_score=[]
            avg_time=[]

print(time.time()-tim)
agent.model.save("trained_model.h5")

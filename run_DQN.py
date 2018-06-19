from collections import deque
from DQNAgent import DQNAgent
from snakegame import SnakeEnv
import numpy as np
import h5py
import copy
import time
import os
import keras

NUM_THREADS = '48'

os.environ['MKL_NUM_THREADS'] = NUM_THREADS
os.environ['GOTO_NUM_THREADS'] = NUM_THREADS
os.environ['OMP_NUM_THREADS'] = NUM_THREADS
os.environ['openmp'] = 'True'

env = SnakeEnv(7, 7)
agent = DQNAgent()
episodes = 10500
avg_score=[]
avg_time=[]

output_filename = "trained_models3/output.txt"
tim = time.time()
epsilon_copy = None
counter = 11
outfile = open(output_filename, "w")
outfile.close()
outfile = open(output_filename, "a")

for e in range(1, episodes + 1):
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
            if counter <= 10:
                outfile.write(str(agent.model.predict(env.getCurrentState().reshape(1, 1, 7, 7))))
                outfile.write(str(state))
                outfile.write("\n")

            if env.done:
                # print the score and break out of the loop
                avg_score.append(len(env.snake)-2)
                avg_time.append(time_t)
                break
        # train the agent with the experience of the episode
        if len(agent.memory)>40000:
            agent.replay(min(48, len(agent.memory)))

        if e%100==0:
            outfile.write("episode: {}/{}, average score: {}, average time: {}, epsilon: {} \n"
                      .format(e, episodes, np.array(avg_score[-100:]).mean(), np.array(avg_time[-100:]).mean(), agent.epsilon))

        if e%1000==0:
            agent.model.save("trained_models3/trained_model_%d.h5" %(e))
            outfile.write("\n\nepisode: {}/{}, average score: {}, average time: {}, epsilon: {} \n"
                             .format(e, episodes, np.array(avg_score[-1000:]).mean(), np.array(avg_time[-1000:]).mean(), agent.epsilon))
            outfile.write("\n\n GREEDY POLICY EVALUATION: \n")
            
            epsilon_copy = agent.epsilon
            counter = 0
            agent.epsilon = 0
            # agent.clone()
        
        if counter <= 10:
            counter += 1
        
        if counter == 10:
            print("Episode number : %d" %(e))
            agent.epsilon = epsilon_copy
            counter = 11
            outfile.write("\n\nGREEDY episode: {}/{}, average score: {}, average time: {}, epsilon: {} \n"
                           .format(e, episodes, np.array(avg_score[-10:]).mean(), np.array(avg_time[-10:]).mean(), agent.epsilon))
            avg_score = []
            avg_time = []
            

outfile.close()
print(time.time()-tim)
agent.model.save("trained_models3/trained_model_final.h5")

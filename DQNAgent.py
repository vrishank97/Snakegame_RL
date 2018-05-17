import tensorflow as tf
from models import LightCNN
from collections import deque
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, Dropout, Flatten
from keras.optimizers import Adam, RMSprop
import numpy as np
import random
from keras.layers import Conv2D, MaxPooling2D, Flatten
from keras.layers import Input, LSTM, Embedding, Dense
from keras.models import Model, Sequential


class DQNAgent:
    def __init__(self, env, observation_space, action_space):
        self.action_space = action_space
        self.observation_space = observation_space
        self.memory = deque(maxlen=500)
        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilon_min = 0.075
        self.epsilon_decay = 0.995
        self.learning_rate = 0.01
        self.env = env
        self.model = self._build_model()

    def _build_model(self):
        state_input = Input(shape=self.env.observation_space.shape)
        h1 = Dense(24, activation='relu')(state_input)
        h2 = Dense(48, activation='relu')(h1)
        h3 = Dense(24, activation='relu')(h2)
        output = Dense(self.env.action_space.shape, activation='linear')(h3)
        
        model = Model(input=state_input, output=output)
        adam  = Adam(lr=self.learning_rate)
        model.compile(loss="mse", optimizer=adam)

        return model

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        act_values=self.model.predict(state.reshape(1, 1, 10, 10))
        return np.argmax(act_values[0])  # returns action

    def replay(self, batch_size):

        minibatch=random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
              # predict the future discounted reward
              target = reward + self.gamma * np.amax(self.model.predict(next_state))

            if reward != 0:
                target = reward

            target_f = self.model.predict(state)
            target_f[0][action] = target
            self.model.fit(state, target_f, epochs=1, verbose=0)

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay


    def train(self, episodes):
        for e in range(episodes):
            
            state = env.reset()

            for time_t in range(500):

                action = agent.act(self.env.state)
                next_state, reward, done = self.env.step(action)

                self.remember(state, action, reward, next_state, done)
                # make next_state the new current state for the next frame.
                state = next_state
                # done becomes True when the game ends
                if self.env.done:
                    # print the score and break out of the loop
                    print("episode: {}/{}, score: {}, time: {}, epsilon: {}"
                          .format(e, episodes, len(self.env.snake)-3, time_t, self.epsilon))
                    break

                if e%10 == 0:
                    print(state)
                    print(self.model.predict(state))

            # train the agent with the experience of the episode
            
            self.replay(min(32, len(self.memory)))





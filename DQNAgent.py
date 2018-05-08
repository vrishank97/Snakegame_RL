import tensorflow as tf
from models import LightCNN
from collections import deque
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, Dropout, Flatten
from keras.optimizers import Adam
import numpy as np
import random

class DQNAgent:
    def __init__(self):
        self.action_size = 4
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95    # discount rate
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.9995
        self.learning_rate = 0.01
        self.model = self._build_model()

    def _build_model(self):

        # Neural Net for Deep-Q learning Model
        model = Sequential()
        model.add(Conv2D(16, (5, 5), input_shape=(1, 10, 15), activation='relu', dim_ordering="th"))
        model.add(Dropout(0.2))
        model.add(Flatten())

        model.add(Dense(50, activation='relu'))
        model.add(Dense(50, activation='relu'))
        model.add(Dense(4, activation='relu'))
        model.compile(loss='mse',
                      optimizer=Adam(lr=self.learning_rate))

        return model

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        act_values=self.model.predict(state)
        return np.argmax(act_values[0])  # returns action

    def replay(self, batch_size):
        minibatch=random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            target=reward
            if not done:
              target=reward + self.gamma * \
                       np.amax(self.model.predict(next_state.reshape(1, 1, 10, 15))[0])
            target_f=self.model.predict(state.reshape(1, 1, 10, 15))
            target_f[0][action]=target
            self.model.fit(state.reshape(1, 1, 10, 15), target_f, epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

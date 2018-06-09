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
    def __init__(self):
        self.action_size = 3
        self.memory = deque(maxlen=500)
        self.gamma = 0.95    # discount rate
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.075
        self.EPSILON_DECAY = 0.00000185
        self.learning_rate = 0.001
        self.model = self._build_model()

    def _build_model(self):
        model = Sequential()
        model.add(Conv2D(16, (5, 5), activation='relu', input_shape=(1, 10, 10), dim_ordering="th"))    
        #model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.2))
        model.add(Conv2D(32, (3, 3), activation='relu', dim_ordering="th"))
        model.add(Dropout(0.2))
        model.add(Conv2D(64, (3, 3), activation='relu', dim_ordering="th"))
        model.add(Flatten())
        model.add(Dense(256, activation='relu'))
        model.add(Dense(3, activation='linear'))
        model.compile(loss='mse', optimizer=RMSprop(lr=self.learning_rate, rho=0.9, epsilon=None, decay=0.0))
        model.summary()
        
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
        X = []
        y = []

        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
              target = reward + self.gamma * \
                       np.amax(self.model.predict(next_state.reshape(1, 1, 10, 10))[0])

            target_f = self.model.predict(state.reshape(1, 1, 10, 10))
            target_f[0][action] = target
            X.append(state.reshape(1, 10, 10))
            y.append(target_f[0])

        self.model.fit(np.array(X), np.array(y), epochs=1, verbose=0)

        if self.epsilon > self.epsilon_min:
            self.epsilon -= self.EPSILON_DECAY

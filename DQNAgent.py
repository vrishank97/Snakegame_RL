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
        self.memory = deque(maxlen=105)
        self.gamma = 0.95    # discount rate
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.075
        self.epsilon_decay = 0.995
        self.learning_rate = 0.1
        self.model = self._build_model()

    def _build_model(self):
        state = Input(shape=(1, 10, 10))
        x = Flatten()(state)
        x = Dense(128, activation='relu')(x)
        x = Dense(128, activation='relu')(x)
        x = Dense(128, activation='relu')(x)
        y = Dense(3, activation='linear')(x)        
        model = Model(state, y)
        '''
        model = Sequential()

        model.add(Dense(32, input_shape=(1, 10, 10), activation='relu'))
        model.add(Dropout(0.3))
        model.add(Flatten())
        model.add(Dense(32, activation='relu'))
        model.add(Dropout(0.3))
        model.add(Dense(4, activation='linear'))
        '''
        model.summary()
        model.compile(Adam(), 'MSE')

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
              target = reward + self.gamma * np.amax(self.model.predict(next_state.reshape(1, 1, 10, 10))[0])
            #print (np.amax(self.model.predict(next_state.reshape(1, 1, 10, 10))[0]))
            if done != 0:
                target = reward
            target_f = self.model.predict(state.reshape(1, 1, 10, 10))
            target_f[0][action] = target
            self.model.fit(state.reshape(1, 1, 10, 10), target_f, epochs=1, verbose=0)

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

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
        self.memory = deque(maxlen=100)
        self.gamma = 0.95    # discount rate
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.075
        self.epsilon_decay = 0.9995
        self.learning_rate = 0.01
        self.model = self._build_model()

    def _build_model(self):
    	inputs = Input(shape=(10,10,))
		x = Dense(32)(inputs)
		x = Dense(32)(x)
		x = Dense(32)(x)
		y = Dense(3)(x)
		model = Model(inputs=inputs, outputs=y)
		model.compile(loss='mse', optimizer=Adam(lr=self.learning_rate))
        model.summary()

        return model

    def remember(self, state, action, reward, next_state, done):
    	self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        act_values=self.model.predict(state.reshape(1, 1, 40, 40))
        return np.argmax(act_values[0])

    
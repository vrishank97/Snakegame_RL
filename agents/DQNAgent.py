from collections import deque
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, Dropout, Flatten
from keras.optimizers import Adam, RMSprop, SGD
import numpy as np
import random
from keras.layers import Conv2D, MaxPooling2D, Flatten
from keras.layers import Input, LSTM, Embedding, Dense
from keras.models import Model, Sequential
import keras
from time import time
import pandas as pd

class DQNAgent:
    def __init__(self, env):
        self.env = env
        self.action_size = 4
        self.memory = deque(maxlen=50000)
        self.gamma = 0.99    # discount rate
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.075
        self.epsilon_decay = 0
        self.learning_rate = 0.01
        self.model = self._build_model()
        self.clone()
        self.output_filename = "trained_models/output.txt"
        self.outfile = open(self.output_filename, "w")
        self.outfile.close()
        self.data = []
        self.epsilon_decay_func_dict = {"exponential" : self.epsilon_exponential_decay,
                                        "linear" : self.epsilon_linear_decay,
                                        "constant" : self.epsilon_constant_decay,
                                       }

    def _print_file(self, str):
        with open(self.output_filename, "a") as outfile:
            outfile.write(str)

    def epsilon_exponential_decay(self, epsilon):
         return epsilon * self.epsilon_decay
    
    def epsilon_linear_decay(self, epsilon):
         return epsilon - self.epsilon_decay

    def epsilon_constant_decay(self, epsilon):
         return epsilon
        
    def _build_model(self):
        x_dim = self.env.x
        y_dim = self.env.y
        model = Sequential()
        model.add(Conv2D(16, (2, 2), activation='relu', input_shape=(1, x_dim, y_dim), data_format="channels_first"))
        model.add(Conv2D(32, (2, 2), activation='relu', data_format="channels_first"))
        # model.add(Conv2D(32, (3, 3), activation='relu', data_format="channels_first"))
        model.add(Flatten())
        model.add(Dense(64, activation='relu'))
        # model.add(Dense(64, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss='mse', optimizer=SGD(lr=self.learning_rate))
        model.summary()
        
        return model

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        x_dim = self.env.x
        y_dim = self.env.y

        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        act_values=self.model.predict(state.reshape(1, 1, x_dim, y_dim))
        return np.argmax(act_values[0])  # returns action

    def act_greedy(self, state):
        x_dim = self.env.x
        y_dim = self.env.y

        act_values=self.model.predict(state.reshape(1, 1, x_dim, y_dim))
        return np.argmax(act_values[0])  # returns action

    def play(self, saved_model_file="/home/s.aakhil/trained_models/model_final.h5", episodes=10):
        self.model = keras.models.load_model(saved_model_file)

        greedy_score = []
        greedy_time = []
        
        for e in range(1, episodes+1):
            print("Playing out episode %d : " %(e))
            state = self.env.reset()
            for time_t in range(400):

                state = self.env.getCurrentState()
                action = self.act_greedy(self.env.state)
                next_state, reward, done = self.env.step(action)
                next_state = self.env.getCurrentState()
                state = next_state
                print("\n", state)

                if self.env.done:
                    print("Game Over. \n Episode score : %d, Episode timesteps : %d " %(len(self.env.snake) - 2, time_t) )
                    greedy_score.append(len(self.env.snake)-2)
                    greedy_time.append(time_t)
                    break

        print("Played %d episodes. Average Score : %f Average Time : %f Best Score : %d Best Time : %d" %(episodes, np.mean(np.array(greedy_score)), np.mean(np.array(greedy_time)), max(greedy_score), max(greedy_time)))


    def replay(self, batch_size, epsilon_decay_func):
        x_dim = self.env.x
        y_dim = self.env.y

        minibatch=random.sample(self.memory, batch_size)
        X = []
        y = []

        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
              target = reward + self.gamma * \
                       np.amax(self.model.predict(next_state.reshape(1, 1, x_dim, y_dim))[0])

            target_f = self.model.predict(state.reshape(1, 1, x_dim, y_dim))
            target_f[0][action] = target
            X.append(state.reshape(1, x_dim, y_dim))
            y.append(target_f[0])

        self.model.fit(np.array(X), np.array(y), epochs=1, verbose=0)
        
        if self.epsilon > self.epsilon_min:
            self.epsilon = self.epsilon_decay_func_dict[epsilon_decay_func](self.epsilon)

    def clone(self):
        self.target_network = keras.models.clone_model(self.model)
    
    def greedy_eval(self, episode, episodes=25):
        greedy_score = []
        greedy_time = []

        for e in range(1, episodes+1):
            state = self.env.reset()
            for time_t in range(400):

                state = self.env.getCurrentState()
                action = self.act_greedy(self.env.state)
                next_state, reward, done = self.env.step(action)
                next_state = self.env.getCurrentState()
                state = next_state

                if self.env.done:
                    greedy_score.append(len(self.env.snake)-2)
                    greedy_time.append(time_t)
                    break

        avg_score= np.mean(greedy_score)
        avg_time = np.mean(greedy_time)

        greedy_max_score = np.array(greedy_score).max()
        greedy_max_time = np.array(greedy_time).max()

        if np.isnan(avg_score):
            avg_scores = 0.0
        if np.isnan(avg_time):
            avg_times = 0.0

        self.data.append([episode, avg_score, avg_time, greedy_max_score, greedy_max_time])

        self._print_file("average score: %.2f, average time: %.2f, max score: %f, max time: %f\n" %(avg_score, avg_time, greedy_max_score, greedy_max_time))
        print("average score: %.2f, average time: %.2f, max score: %f, max time: %f\n" %(avg_score, avg_time, greedy_max_score, greedy_max_time))    

    def get_epsilon(self, episodes, epsilon_decay_func):
        if epsilon_decay_func == "exponential":
            return np.exp(np.log(self.epsilon_min)/episodes)
        
        elif epsilon_decay_func == "linear":
            return (1 - self.epsilon_min) / (0.1 * episodes)
        
        elif epsilon_decay_func == "constant":
            return 0

    def train(self, episodes=5000, start_mem=10000, batch_size=24, verbose_eval=1000, save_iter=1000, epsilon_decay_func="exponential"):
        self.get_epsilon(episodes, epsilon_decay_func)
        time_begin = time()
        time_prev = time()

        for e in range(1, episodes + 1):
            state = self.env.reset()
            for time_t in range(400):
                state = self.env.getCurrentState()
                action = self.act(self.env.state)
                next_state, reward, done = self.env.step(action)
                next_state = self.env.getCurrentState()
                self.remember(state, action, reward, next_state, done)
                state = next_state

                if self.env.done:
                    break
            
            if len(self.memory)>start_mem:
                self.replay(batch_size, epsilon_decay_func)
            if e%verbose_eval == 0:
                self._print_file("After %d Episodes :\n" %(e))
                self.greedy_eval(e)
                self._print_file("Episode number : %d\n Time for past %d episodes :%f\n\n" %(e, verbose_eval, time() - time_prev))
                print("Episode number : %d\n Time for past %d episodes :%f\n\n" %(e, verbose_eval, time() - time_prev))
                time_prev = time()
            if e%save_iter == 0:
                self.model.save("trained_models/trained_model_%d.h5" %(e))
        self._print_file(str(time()-time_begin))
        print(time()-time_begin)
        df = pd.DataFrame(self.data, columns=['Episodes', 'Scores', 'Time', 'Best Score', 'Best Time'])
        df.to_csv("trained_models/scores.csv")
        self.model.save("trained_models/final_model.h5")


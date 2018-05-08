import numpy as np
from random import randint
from collections import deque


class SnakeEnv:

    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.state = np.zeros((x, y), dtype=int)
        self.reward_range = (0, 1)
        self.snake = deque([[5, 5], [5, 6], [5, 7], [5, 8], [5, 9], [5, 10]])
        self.score = 0
        self.done = 0
        self.reset()
        self.food()

    def food(self):
        a = randint(0, self.x-1)
        b = randint(0, self.y-1)
        if self.state[a][b] == 0:
            self.state[a][b] = 2
        else:
            self.food()

    def reset(self):
        self.state = np.zeros((self.x, self.y), dtype=int)
        self.snake = deque([[5, 5], [5, 6], [5, 7], [5, 8], [5, 9], [5, 10]])
        for i in self.snake:
            self.state[i[0]][i[1]] = 1
        self.state[0] = 3
        self.state[self.x - 1] = 3
        for i in range(self.x):
            self.state[i][0] = 3
            self.state[i][self.y - 1] = 3
        self.food()
        return self.state

    def render(self):
        for i in self.snake:
            self.state[i[0]][i[1]] = 1

    def step(self, action):
        state = self.state
        snake = self.snake
        head = snake[0]
        self.done = 0
        skore = len(snake)
        if action == 0:
            # collision with wall
            if state[head[0]][head[1]-1] == 3:
                self.done = 1
            # food
            if state[head[0]][head[1]-1] == 2:
                snake.appendleft([head[0], head[1]-1])
                self.food()
            # collision with self
            if state[head[0]][head[1]-1] == 1:
                self.done = 1
            # no obstruction
            if state[head[0]][head[1]-1] == 0:
                snake.appendleft([head[0], head[1]-1])
                state[snake[-1][0]][snake[-1][1]] = 0
                snake.pop()

        # Up
        if action == 1:
            # collision with wall
            if state[head[0]-1][head[1]] == 3:
                self.done = 1
            # food
            if state[head[0]-1][head[1]] == 2:
                snake.appendleft([head[0]-1, head[1]])
                self.food()
            # collision with self
            if state[head[0]-1][head[1]] == 1:
                self.done = 1
            # no obstruction
            if state[head[0]-1][head[1]] == 0:
                snake.appendleft([head[0]-1, head[1]])
                state[snake[-1][0]][snake[-1][1]] = 0
                snake.pop()

        # Right
        if action == 2:
            # collision with wall
            if state[head[0]][head[1]+1] == 3:
                self.done = 1
            # food
            if state[head[0]][head[1]+1] == 2:
                snake.appendleft([head[0], head[1]+1])
                self.food()
            # collision with self
            if state[head[0]][head[1]+1] == 1:
                self.done = 1
            # no obstruction
            if state[head[0]][head[1]+1] == 0:
                snake.appendleft([head[0], head[1]+1])
                state[snake[-1][0]][snake[-1][1]] = 0
                snake.pop()

        # Down
        if action == 3:
            # collision with wall
            if state[head[0]+1][head[1]] == 3:
                self.done = 1
            # food
            if state[head[0]+1][head[1]] == 2:
                snake.appendleft([head[0]+1, head[1]])
                self.food()
            # collision with self
            if state[head[0]+1][head[1]] == 1:
                self.done = 1
            # no obstruction
            if state[head[0]+1][head[1]] == 0:
                snake.appendleft([head[0]+1, head[1]])
                state[snake[-1][0]][snake[-1][1]] = 0
                snake.pop()
        self.render()
        self.score = (len(snake) - skore)*10
        if self.done:
            self.score = -10
        return state, self.score, self.done

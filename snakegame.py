import numpy as np
from random import randint
from collections import deque

FOOD = 10
SNAKE = 7
HEAD = 8
WALL = 1
GROUND =4 

class SnakeEnv:

    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.state = np.ones((x, y), dtype=int)*GROUND
        self.reward_range = (0, 1)
        self.snake = deque([[5, 4],[5, 5],[5, 6]])
        self.score = 0
        self.done = 0
        self.reset()

    def food(self):
        a = randint(0, self.x-1)
        b = randint(0, self.y-1)
        if self.state[a][b] == GROUND:
            self.state[a][b] = FOOD
        else:
            self.food()

    def reset(self):
        x = self.x
        y = self.y

        self.state = np.ones((x, y), dtype=int)*GROUND
        self.snake = deque([[5, 4],[5, 5],[5, 6]])
        for i in self.snake:
            self.state[i[0]][i[1]] = SNAKE
        head = self.snake[0]
        self.state[head[0]][head[1]] = HEAD

        self.state[0] = WALL
        self.state[self.x - 1] = WALL
        for i in range(self.x):
            self.state[i][0] = WALL
            self.state[i][self.y - 1] = WALL
        self.food()
        return self.state

    def render(self):
        for i in self.snake:
            self.state[i[0]][i[1]] = SNAKE
        head = self.snake[0]
        self.state[head[0]][head[1]] = HEAD

    def step(self, action):
        state = self.state
        snake = self.snake
        head = snake[0]
        self.done = 0
        skore = len(snake)
        next_move = [0,0]

        # Left
        if action == 0:
            next_move[0] = head[0]
            next_move[1] = head[1]-1
        # Up
        if action == 1:
            next_move[0] = head[0]-1
            next_move[1] = head[1]
        # Right
        if action == 2:
            next_move[0] = head[0]
            next_move[1] = head[1]+1
        # Down
        if action == 3:
            next_move[0] = head[0]+1
            next_move[1] = head[1]

        # collision with wall
        if state[next_move[0]][next_move[1]] == WALL:
            self.done = 1
        # food
        if state[next_move[0]][next_move[1]] == FOOD:
            snake.appendleft([next_move[0], next_move[1]])
            self.food()
        # collision with self
        if state[next_move[0]][next_move[1]] == SNAKE:
            self.done = 1
        # no obstruction
        if state[next_move[0]][next_move[1]] == GROUND:
            snake.appendleft([next_move[0], next_move[1]])
            state[snake[-1][0]][snake[-1][1]] = GROUND
            snake.pop()

        self.render()
        self.score = (len(snake) - skore)
        if self.done:
            self.score = 0
        return state, self.score, self.done

import numpy as np
from random import randint
from collections import deque
import copy

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
        self.snake = deque([[3, 3],[3, 4],[3, 5]])
        self.score = 0
        self.done = 0
        self.screen = np.ones((x*4, y*4), dtype=int)*GROUND
        self.reset()

    def project(self):
        state = self.state
        screen = self.screen
        for i in range(10):
            for j in range(10):
                screen[i*4][j*4] = state[i][j]
                screen[i*4+1][j*4] = state[i][j]
                screen[i*4][j*4+1] = state[i][j]
                screen[i*4+1][j*4+1] = state[i][j]
        return screen

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
        self.food()
        return self.state

    def render(self):
        for i in self.snake:
            self.state[i[0]][i[1]] = SNAKE
        head = self.snake[0]
        self.state[head[0]][head[1]] = HEAD

    def getDirection(self):
        snake = self.snake
        head = snake[0]
        neck = snake[1]

        if head[1] - 1 == neck[1]:
            # facing up
            return 0
        elif head[0] - 1 == neck[0]:
            # facing right
            return 1
        elif head[1] + 1 == neck[1]:
            # facing down
            return 2
        elif head[0] + 1 == neck[0]:
            # facing left
            return 3

    def step_absolute(self, direction):
        head = self.snake[0]
        last_pos = self.snake[1]

        if direction == 0:
            # screen up
            return [head[0], head[1] + 1]
        elif direction == 1:
            # screen right
            return [head[0] + 1, head[1]]
        elif direction == 2:
            # screen down
            return [head[0], head[1] - 1]
        elif direction == 3:
            #screen left
            return [head[0] - 1, head[1]] 

    def step(self, action):
        #relative 

        state = self.state
        snake = self.snake
        head = snake[0]
        self.done = 0
        skore = len(snake)
        next_move = [0,0]
        last_pos = self.snake[1]
        neck_pos = self.getDirection()

        if action == 0:
            if last_pos == self.step_absolute(0):
                next_move = self.step_absolute(2)
            else:
                next_move = self.step_absolute(0)

        if action == 1:
            if last_pos == self.step_absolute(1):
                next_move = self.step_absolute(3)
            else:
                next_move = self.step_absolute(1)

        if action == 2:
            if last_pos == self.step_absolute(2):
                next_move = self.step_absolute(0)
            else:
                next_move = self.step_absolute(2)

        if action == 3:
            if last_pos == self.step_absolute(3):
                next_move = self.step_absolute(1)
            else:
                next_move = self.step_absolute(3)

        '''

        # Left
        if action == 0:
            if neck_pos == 0:
                next_move = self.step_absolute(3)
            elif neck_pos == 1:
                next_move = self.step_absolute(0)
            elif neck_pos == 2:
                next_move = self.step_absolute(1)
            else:
                next_move = self.step_absolute(2)
        # Up
        if action == 1:
            if neck_pos == 0:
                next_move = self.step_absolute(0)
            elif neck_pos == 1:
                next_move = self.step_absolute(1)
            elif neck_pos == 2:
                next_move = self.step_absolute(2)
            else:
                next_move = self.step_absolute(3)
        # Right
        if action == 2:
            if neck_pos == 0:
                next_move = self.step_absolute(1)
            elif neck_pos == 1:
                next_move = self.step_absolute(2)
            elif neck_pos == 2:
                next_move = self.step_absolute(3)
            else:
                next_move = self.step_absolute(0)

        '''

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
            self.score = -1
        return self.state, self.score, self.done

    def getCurrentState(self):
        state = copy.deepcopy(self.state)
        return state

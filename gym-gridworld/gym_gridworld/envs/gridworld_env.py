import gym
from gym import error, spaces, utils
from gym.utils import seeding


class GridWorldEnv(gym.Env):
    metadata = {'render.modes': ['human', 'graphic']}

    def __init__(self, x_y_dim=3):
        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Tuple((spaces.Discrete(x_y_dim), spaces.Discrete(x_y_dim)))
        # self.reward_range = None  # by default [-inf, inf]

        self.x_y_dim = x_y_dim
        self.done = False
        self.curr_state = [0, 0]
        self.terminal_state = (self.x_y_dim, self.x_y_dim)
        self.info = {}

    def _step(self, action):
        action_str = ACTION_MEANING[action]
        if action_str == "UP":
            self.curr_state[0] -= 1
        elif action_str == "DOWN":
            self.curr_state[0] += 1
        elif action_str == "LEFT":
            self.curr_state[1] -= 1
        elif action_str == "RIGHT":
            self.curr_state[1] += 1

        if self.curr_state[0] < 0:
            self.curr_state[0] = 0
        elif self.curr_state[0] > self.x_y_dim:
            self.curr_state[0] = self.x_y_dim
        if self.curr_state[1] < 0:
            self.curr_state[1] = 0
        elif self.curr_state[1] > self.x_y_dim:
            self.curr_state[1] = self.x_y_dim

        observation = self.curr_state
        reward = 0
        if self.curr_state[0] == self.terminal_state[0] and self.curr_state[1] == self.terminal_state[1]:
            reward = 10
            self.done = True

        return observation, reward, self.done, self.info

    def _reset(self):
        self.done = False
        self.curr_state = [0, 0]
        self.terminal_state = (self.x_y_dim, self.x_y_dim)
        return self.curr_state

    def _render(self, mode='human', close=False):
        # TODO: must RETURN either a display, terminal or array. It shouldn't print or render inside the function
        if mode == 'human':
            grid = [['o', 'o', 'o', 'o'], ['o', 'o', 'o', 'o'], ['o', 'o', 'o', 'o'], ['o', 'o', 'o', 'o']]

            grid[self.curr_state[0]][self.curr_state[1]] = 'x'
            grid[self.terminal_state[0]][self.terminal_state[1]] = 'T'

            for row in grid:
                for el in row:
                    print(el, end=' ')
                print()
            print()
        elif mode == 'graphic':
            #TODO: Rendering of the GridWorld in a window
            raise NotImplementedError
        else:
            super(GridWorld, self).render(mode=mode)

    def _close(self):
        raise NotImplementedError

    def _seed(self, seed=None):
        raise NotImplementedError


ACTION_MEANING = {
    0: "UP",
    1: "RIGHT",
    2: "DOWN",
    3: "LEFT",
}

if __name__ == '__main__':
    pass
    # from random import randint
    # env = GridWorld()
    # for i_episode in range(1):
    #     observation = env.reset()
    #     for t in range(100):
    #         env.render()
    #         # print(observation)
    #         action = randint(0, 3)
    #         observation, reward, done, info = env.step(action)
    #
    #         if done:
    #             print("Episode finished after {} timesteps".format(t + 1))
    #             env.render()
    #             break

import sys
import random
import time

import numpy as np
from six import StringIO
import gym
from gym import spaces
from gym.utils import seeding

from core.envs import config
from core.envs import maze_generation


class GridUniverseEnv(gym.Env):
    metadata = {'render.modes': ['human', 'ansi', 'graphic']}

    def __init__(self, grid_shape=(4, 4), *, initial_state=0, goal_states=None, lava_states=None, walls=None,
                 lemons=None, melons=None, apples=None, custom_world_fp=None, random_maze=False):
        """
        The constructor for creating a GridUniverse environment. The default GridUniverse is a square grid of 4x4 where the
        agent starts in the top left corner and the terminal goal state is in the bottom right corner.

        :param grid_shape: Tuple of size 2 to specify (width, height) of grid
        :param initial_state: int for single initial state or list of possible states chosen uniform randomly
        "Terminal states". The episode ends if the agent reaches this type of state (done = True).
        :param goal_states: Terminal states with positive reward
        :param lava_states: Terminal states with negative reward
        :param walls: list of walls. These are blocked states where the agent can't enter/walk on
        :param lemons: If agent lands on a state containing a lemon: small negative reward
        :param melons: If agent lands on a state containing a melon: large reward
        :param apples: If agent lands on a state containing an apple: small reward
        :param custom_world_fp: optional parameter to create the grid from a text file.
        :param random_maze: optional parameter to randomly generate a maze from the algorithm within maze_generation.py
                            This will override the params initial_state, goal_states, lava_states,
                            walls and custom_world_fp params
        """
        self._check_param_types(grid_shape, initial_state, goal_states,
                                lava_states, walls, lemons, melons, apples)
        self.x_max = grid_shape[0] # num columns
        self.y_max = grid_shape[1] # num rows
        self.world = self._generate_world()
        # set action space params
        self.action_space = spaces.Discrete(4)
        # main boundary check for edges of map done here.
        # To get to another row, we subtract or add the width of the grid (self.x_max) since the state is an integer
        self.action_state_to_next_state = [lambda s: s - self.x_max if self.world[s][1] > 0 else s,                # up
                                           lambda s: s + 1 if self.world[s][0] < (self.x_max - 1) else s,          # right
                                           lambda s: s + self.x_max if self.world[s][1] < (self.y_max - 1) else s, # down
                                           lambda s: s - 1 if self.world[s][0] > 0 else s]                         # left

        self.action_descriptors = ['UP', 'RIGHT', 'DOWN', 'LEFT']
        self.action_descriptor_to_int = {desc: idx for idx, desc in enumerate(self.action_descriptors)}
        # set observed params: [current state, world state]
        self.observation_space = spaces.Discrete(self.world.size)
        # set initial state for the agent. If initial_state is a list, choose randomly
        if isinstance(initial_state, int):
            initial_state = [initial_state] # convert to list
        self.initial_states = initial_state
        self.previous_state = self.current_state = self.initial_state = random.choice(self.initial_states)
        # set terminal goal states state(s) and default terminal state if None given
        if goal_states is None or len(goal_states) == 0:
            self.goal_states = [self.world.size - 1]
        else:
            self.goal_states = goal_states
        # set lava terminal states
        if lava_states is None:
            self.lava_states = []
        else:
            self.lava_states = lava_states
        for t_s in self.lava_states:
            if t_s < 0 or t_s > (self.world.size - 1):
                raise ValueError("lava state {} is out of grid bounds".format(t_s))
        # set lemons, melons and apples
        # Initial fruit lists
        if lemons is None: self.lemons = []
        else: self.lemons = lemons
        if apples is None: self.apples = []
        else: self.apples = apples
        if melons is None: self.melons = []
        else: self.melons = melons
        # current fruit can get eaten and has to placed back after reset
        self.current_lemons = self.lemons[:]
        self.current_apples = self.apples[:]
        self.current_melons = self.melons[:]
        # set walls
        self.wall_indices = []
        self.wall_grid = np.zeros(self.world.shape)
        self._generate_walls(walls)
        # set reward matrix
        self.LEMON_REWARD = config.default_rewards['LEMON_REWARD']
        self.APPLE_REWARD = config.default_rewards['APPLE_REWARD']
        self.MELON_REWARD = config.default_rewards['MELON_REWARD']
        self.LAVA_REWARD = config.default_rewards['LAVA_REWARD']
        self.TERMINAL_GOAL_REWARD = config.default_rewards['TERMINAL_GOAL_REWARD']
        self.MOVEMENT_REWARD = config.default_rewards['MOVEMENT_REWARD']
        self.reward_matrix = np.full(self.world.shape, self.MOVEMENT_REWARD)
        self._generate_reward_matrix()
        for terminal_state in self.goal_states:
            try:
                self.reward_matrix[terminal_state] = 10
            except IndexError:
                raise IndexError("Terminal goal state {} is out of grid bounds or is wrong type. Should be an integer.".format(terminal_state))
        for terminal_state in self.lava_states:
            try:
                self.reward_matrix[terminal_state] = -10
            except IndexError:
                raise IndexError("Lava terminal state {} is out of grid bounds or is wrong type. Should be an integer.".format(terminal_state))
        # self.reward_range = [-inf, inf] # default values already
        self.num_previous_states_to_store = 500
        self.last_n_states = []
        # set additional parameters for the environment
        self.done = False
        self.info = {}
        self.screen_width = config.default_screen_dimension['SCREEN_WIDTH']
        self.screen_height = config.default_screen_dimension['SCREEN_HEIGHT']

        self.viewer = None
        self._seed()
        self.np_random, seed = seeding.np_random(55)

        if custom_world_fp:
            self._create_custom_world_from_file(custom_world_fp)
        if random_maze:
            self._create_random_maze(self.x_max, self.y_max)

        # After every entity has been placed, check collisions
        self._check_specific_collisions()

    def _check_param_types(self, grid_shape, initial_state, goal_states,
                                lava_states, walls, lemons, melons, apples):
        """
        Check that each parameter is the correct type
        """

        if not isinstance(grid_shape, tuple) or len(grid_shape) != 2 or not isinstance(grid_shape[0], int):
            raise TypeError("grid_shape parameter must be tuple of two integers")
        if not isinstance(initial_state, list) and not isinstance(initial_state, int):
            raise TypeError("initial_state parameter must be a list or an int")
        if goal_states is not None and not isinstance(goal_states, list):
            raise TypeError("goal_states parameter must be a list of integer indices")
        if lava_states is not None and not isinstance(lava_states, list):
            raise TypeError("lava_states parameter must be a list of integer indices")
        if walls is not None and not isinstance(walls, list):
            raise TypeError("walls parameter must be a list of integer indices")
        if lemons is not None and not isinstance(lemons, list):
            raise TypeError("lemons parameter must be a list of integer indices")
        if melons is not None and not isinstance(melons, list):
            raise TypeError("melons parameter must be a list of integer indices")
        if apples is not None and not isinstance(apples, list):
            raise TypeError("apples parameter must be a list of integer indices")

    def _check_specific_collisions(self):
        """
        Check that objects/entities/terminal states don't collide in specific ways.
        Especially with themselves
        """

        if len(self.current_apples) != len(set(self.current_apples)):
            raise ValueError('Duplicate apples not allowed')
        if len(self.current_lemons) != len(set(self.current_lemons)):
            raise ValueError('Duplicate lemons not allowed')
        if len(self.current_melons) != len(set(self.current_melons)):
            raise ValueError('Duplicate melons not allowed')
        if len(self.wall_indices) != len(set(self.wall_indices)):
            raise ValueError('Duplicate walls not allowed')
        if len(self.initial_states) != len(set(self.initial_states)):
            raise ValueError('Duplicate starting states not allowed')
        if len(self.goal_states) != len(set(self.goal_states)):
            raise ValueError('Duplicate goal states not allowed')
        if len(self.lava_states) != len(set(self.lava_states)):
            raise ValueError('Duplicate lava states not allowed')

        # Test that starting states, wall indices, terminal states don't collide
        if len(set(self.initial_states) & set(self.wall_indices)) > 0:
            raise ValueError('Collision between starting states and wall indices. Not allowed.')

        if len(set(self.goal_states) & set(self.wall_indices)) > 0:
            raise ValueError('Collision between goal state and wall indices. Not allowed.')

        if len(set(self.lava_states) & set(self.wall_indices)) > 0:
            raise ValueError('Collision between goal state and wall indices. Not allowed.')

        all_fruit = set(self.current_apples + self.current_lemons + self.current_melons)
        if len(all_fruit) != len(self.current_apples + self.current_lemons + self.current_melons) > 0:
            raise ValueError('Some fruit has been placed on top of another. This is not allowed. \
                                      Check melon, lemon, and apple parameters.')

        if len(all_fruit & set(self.wall_indices)) > 0:
            raise ValueError('Some fruit has been placed on top of a wall. This is not allowed. \
                                                  Check melon, lemon, and apple parameters.')

    def _generate_world(self):
        """
        Creates and returns the griduniverse map as a numpy array.

        The states are defined by their index and contain a tuple of uint16 values that represent the
        coordinates (x,y) of a state in the grid.
        """
        world = np.fromiter(((x, y) for y in np.nditer(np.arange(self.y_max))
                             for x in np.nditer(np.arange(self.x_max))), dtype='int64, int64')
        return world

    def _generate_walls(self, walls):
        """
        Given a list of wall indices, fills in self.wall_indices list
        and places "1"s appropriately within self.walls numpy array

        self.walls: need index positioning for efficient check in _is_wall() but
        self.wall_indices: we also need list of indices to easily access each wall sequentially (e.g in render())
        """
        if walls is not None:
            for wall_state in walls:
                if wall_state < 0 or wall_state > (self.world.size - 1):
                    raise ValueError("Wall state {} is out of grid bounds".format(wall_state))

                self.wall_grid[wall_state] = 1
                self.wall_indices.append(wall_state)

    def _generate_reward_matrix(self):
        """
        Set reward matrix accordingly between non-terminal and terminal states.
        apples, lemons and melons (small positive, small negative, large positive respectively)
        Terminal states: goal_states and lava_states
        Every walkable state (except terminal states) you lose -1 (self.MOVEMENT_REWARD)
        so if there is fruit you gain the fruit's reward added to -1 self.MOVEMENT_REWARD.
        """
        # non-terminal specific rewards
        for state in self.current_apples:
            self.reward_matrix[state] += self.APPLE_REWARD
        for state in self.current_lemons:
            self.reward_matrix[state] += self.LEMON_REWARD
        for state in self.current_melons:
            self.reward_matrix[state] += self.MELON_REWARD
        # terminal states override melons, lemons and apples. Also no immediate reward obtained within them.
        for terminal_state in self.goal_states:
            self.reward_matrix[terminal_state] = self.TERMINAL_GOAL_REWARD
        for terminal_state in self.lava_states:
            self.reward_matrix[terminal_state] = self.LAVA_REWARD

    def reward_function(self, next_state, collect=False):
        """
        Reward function which handles removing fruit from grid in case they are collected.
        """

        if collect:
            reward = self.reward_matrix[next_state]
            if next_state in self.current_lemons or next_state in self.current_apples or next_state in self.current_melons:
                print('Collecting fruit at: {} with reward: {}'.format(next_state, reward))
                if next_state in self.current_lemons:
                    next_state_index = self.lemons.index(next_state)
                    self.current_lemons.remove(next_state)

                    if self.viewer:
                        self.viewer.lemon_sprites[next_state_index].visible = False
                elif next_state in self.current_apples:
                    next_state_index = self.apples.index(next_state)
                    self.current_apples.remove(next_state)

                    if self.viewer:
                        self.viewer.apple_sprites[next_state_index].visible = False
                elif next_state in self.current_melons:
                    next_state_index = self.melons.index(next_state)
                    self.current_melons.remove(next_state)

                    if self.viewer:
                        self.viewer.melon_sprites[next_state_index].visible = False

                self.reward_matrix[next_state] = self.MOVEMENT_REWARD
            return reward
        else:
            return self.reward_matrix[next_state]

    def look_step_ahead(self, state, action, care_about_terminal=True):
        """
        Computes the results of a hypothetical action taking place at the given state.

        Returns the state to what that action would lead, the reward at that new state and a boolean value that
        determines if the next state is terminal
        """

        if care_about_terminal:
            if self.is_terminal(state):
                next_state = state
            else:
                next_state = self.action_state_to_next_state[action](state)
                next_state = next_state if not self._is_wall(next_state) else state
        else:
            # repeating code for now, but for good reason
            next_state = self.action_state_to_next_state[action](state)
            next_state = next_state if not self._is_wall(next_state) else state

        return next_state, self.reward_function(next_state, collect=True), self.is_terminal(next_state)

    def _is_wall(self, state):
        """
        Checks if a given state is a wall or any other element that shall not be trespassed.
        """
        return True if self.wall_grid[state] == 1 else False

    def is_terminal(self, state):
        """
        Check if the input state is terminal.
        Which can either be a lava (negative reward) or goal state (positive reward)
        """
        return True if self.is_lava(state) or self.is_terminal_goal(state) else False

    def is_lava(self, state):
        return True if state in self.lava_states else False

    def is_terminal_goal(self, state):
        return True if state in self.goal_states else False

    def _step(self, action):
        """
        Moves the agent one step according to the given action.
        """
        self.previous_state = self.current_state
        self.current_state, reward, self.done = self.look_step_ahead(self.current_state, action)
        self.last_n_states.append(self.world[self.current_state])
        if len(self.last_n_states) > self.num_previous_states_to_store:
            self.last_n_states.pop(0)
        return self.current_state, reward, self.done, self.info

    def _reset(self):
        self.done = False
        self.previous_state = self.current_state = self.initial_state = random.choice(self.initial_states)
        self.last_n_states = []
        self.current_lemons = self.lemons[:]
        self.current_apples = self.apples[:]
        self.current_melons = self.melons[:]
        self._generate_reward_matrix()
        if self.viewer:
            self.viewer.change_face_sprite()
            for sprite in self.viewer.apple_sprites:
                sprite.visible = True
            for sprite in self.viewer.lemon_sprites:
                sprite.visible = True
            for sprite in self.viewer.melon_sprites:
                sprite.visible = True
        return self.current_state

    def _render(self, mode='human', close=False):
        if close:
            if self.viewer is not None:
                self.viewer.close()
                self.viewer = None
            return

        if mode == 'human' or mode == 'ansi':
            new_world = np.fromiter(('o' for _ in np.nditer(np.arange(self.x_max))
                                     for _ in np.nditer(np.arange(self.y_max))), dtype='S1')
            new_world[self.current_state] = 'x'
            for t_state in self.goal_states:
                new_world[t_state] = 'G'
            for t_state in self.lava_states:
                new_world[t_state] = 'L'
            for w_state in self.wall_indices:
                new_world[w_state] = '#'
            for m_state in self.current_melons:
                new_world[m_state] = 'm'
            for l_state in self.current_lemons:
                new_world[l_state] = 'l'
            for a_state in self.current_apples:
                new_world[a_state] = 'a'

            outfile = StringIO() if mode == 'ansi' else sys.stdout
            for row in np.reshape(new_world, (self.y_max, self.x_max)):
                for state in row:
                    outfile.write((state.decode('UTF-8') + ' '))
                outfile.write('\n')
            outfile.write('\n')
            return outfile

        elif mode == 'graphic':
            if self.viewer is None:
                from core.envs import rendering
                self.viewer = rendering.Viewer(self, self.screen_width, self.screen_height)

            return self.viewer.render(return_rgb_array=mode == 'rgb_array')
        else:
            super(GridUniverseEnv, self).render(mode=mode)

    def render_policy_arrows(self, policy):
        if self.viewer is None:
            from core.envs import rendering
            self.viewer = rendering.Viewer(self, self.screen_width, self.screen_height)

        self.viewer.render_policy_arrows(policy)

    def _close(self):
        pass

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def _create_custom_world_from_file(self, fp):
        with open(fp, 'r') as f:
            all_lines = [line.rstrip() for line in f.readlines()]
            all_lines = ["".join(line.split()) for line in all_lines if line] # remove empty lines and any whitespace

            self._create_custom_world_from_text(all_lines)

    def _create_custom_world_from_text(self, text_world_lines):
        """
        Creates the world from a rectangular text file in the format of:

        ooo#
        oxol
        oooL
        oooG

        Where:
         "o" is an empty walkable area.
         "#" is a blocked "wall"
         "G" is a terminal goal state
         "L" is a lava terminal state
         "l" is a lemon
         "a" is an apple
         "m" is a melon
         "x" is a possible starting location. Chosen uniform randomly if multiple "x"s.
        """

        self.goal_states = []
        self.initial_states = []
        self.lava_states = []
        self.initial_states = []
        self.lemons = []
        self.apples = []
        self.melons = []
        walls_indices = []

        curr_index = 0
        width_of_grid = len(text_world_lines[0])  # first row length will be width from now on
        for y, line in enumerate(text_world_lines):
            if len(line) != width_of_grid:
                raise ValueError("Input text file is not a rectangle")

            for char in line:
                if char == 'G':
                    self.goal_states.append(curr_index)
                elif char == 'L':
                    self.lava_states.append(curr_index)
                elif char == 'l':
                    self.lemons.append(curr_index)
                elif char == 'a':
                    self.apples.append(curr_index)
                elif char == 'm':
                    self.melons.append(curr_index)
                elif char == 'o':
                    pass
                elif char == '#':
                    walls_indices.append(curr_index)
                elif char == 'x':
                    self.initial_states.append(curr_index)
                else:
                    raise ValueError('Invalid Character "{}". Returning'.format(char))

                curr_index += 1

        if len(self.initial_states) == 0:
            raise ValueError("No starting states set in text file. Place \"x\" within grid. ")
        if len(self.goal_states) == 0:
            raise ValueError("No terminal goal states set in text file. Place \"T\" within grid. ")

        self.y_max = len(text_world_lines)
        self.x_max = width_of_grid
        self.world = self._generate_world()

        self.wall_grid = np.zeros(self.world.shape)
        self.wall_indices = []
        self._generate_walls(walls_indices)

        self.reward_matrix = np.full(self.world.shape, self.MOVEMENT_REWARD)
        self.reset()

    def _create_random_maze(self, width, height):
        all_textworld_lines = maze_generation.create_random_maze(width, height)

        self._create_custom_world_from_text(all_textworld_lines)

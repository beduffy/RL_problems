import time
import sys
import unittest
import os

from core.envs.griduniverse_env import GridUniverseEnv

class TestGridUniverse(unittest.TestCase):
    def test_lever_parameter_raise_correct_exceptions(self):
        """
        Test the numerous ways in which levers should raise exceptions
        """
        # Check if crashes if not a dictionary
        with self.assertRaises(TypeError):
            GridUniverseEnv(levers=[5, 3])

        # Need walls for levers
        with self.assertRaises(ValueError):
            GridUniverseEnv(levers={5: 3})

        # Wall (3) linked to lever state (5) is not a wall state
        with self.assertRaises(ValueError):
            GridUniverseEnv(walls=[4], levers={5: 3})

        # How it should work
        try:
            GridUniverseEnv(walls=[3], levers={5: 3})
        except:
            self.fail("Should not crash here. Wall state 3 is the value of lever state 5. ")

        # Lever state can not be placed on top of a wall
        with self.assertRaises(ValueError):
            GridUniverseEnv(walls=[4, 8], levers={4: 8})

        # non-int key
        with self.assertRaises(TypeError):
            GridUniverseEnv(walls=[4, 8], levers={'a': 8})

    def test_lever_opens_door_and_both_reset_on_reset(self):
        """
        Create default environment, test that wall exists, move to lever, check wall doesn't exist
        reset environment, check that lever is unactivated, and repeat do previous steps
        """
        env = GridUniverseEnv(walls=[3], levers={1: 3})

        self.assertTrue(env.wall_grid[3] == 1 and 3 in env.wall_indices)
        env.step(env.action_descriptor_to_int['RIGHT'])
        self.assertTrue(env.wall_grid[3] == 0 and 3 not in env.wall_indices)
        self.assertTrue(1 not in env.unactivated_levers)

        env.reset()
        self.assertTrue(1 in env.unactivated_levers)

        self.assertTrue(env.wall_grid[3] == 1 and 3 in env.wall_indices)
        env.step(env.action_descriptor_to_int['RIGHT'])
        self.assertTrue(env.wall_grid[3] == 0 and 3 not in env.wall_indices)

        self.assertTrue(1 not in env.unactivated_levers)

    def test_terminal_out_of_bounds_error(self):
        """
        Default Env contains 16 states (0-15) so state 16 should crash environment.
        """

        with self.assertRaises(IndexError):
            GridUniverseEnv(goal_states=[16])

    def test_wrong_terminal_type_error(self):
        with self.assertRaises(IndexError):
            GridUniverseEnv(goal_states=['a'])

    def test_incorrect_parameter_types(self):
        """
        Test that TypeError is raised if goal_states, lava_states, walls are not a list.
        Test that TypeError is raised if grid_shape is over 2 dimensions, not a tuple/list or contains non-int
        """

        # todo don't show errors/red writing (env.close() is always called) for testing aesthetics
        with self.assertRaises(TypeError):
            GridUniverseEnv(goal_states=5.0)

        with self.assertRaises(TypeError):
            GridUniverseEnv(lava_states='a')

        with self.assertRaises(TypeError):
            GridUniverseEnv(walls='aaaa')

        # Test grid_shape with over 2 dimensions
        with self.assertRaises(TypeError):
            GridUniverseEnv(grid_shape=(2, 2, 2))

        # Test grid_shape if not a tuple/list
        with self.assertRaises(TypeError):
            GridUniverseEnv(grid_shape=set([2, 3]))

        with self.assertRaises(TypeError):
            GridUniverseEnv(grid_shape=[2, 2.0])

        with self.assertRaises(TypeError):
            GridUniverseEnv(grid_shape=2)

    def test_griduniverse_wall_not_trespassed(self):
        """
        Test whether agent is still in the same place after moving into a wall
        """
        env = GridUniverseEnv(walls=[1])
        env.render()
        action = 1 # go right

        observation, reward, done, info = env.step(action)
        print('go ' + env.action_descriptors[action])

        env.render()
        self.assertTrue(observation == 0) # check if in same starting place

    def test_default_griduniverse_completion_in_six_steps(self):
        """
        Test whether the agent reaches a terminal state within the
        default square GridUniverse within six steps by going right 3 times
        and then going down 3 times.
        """
        env = GridUniverseEnv()

        actions_to_take = [1, 1, 1, 2, 2, 2] # 3 rights and 3 downs
        for t in range(100):
            env.render()
            action = actions_to_take[t]
            print('go ' + env.action_descriptors[action])
            observation, reward, done, info = env.step(action)

            if done:
                print("Episode finished after {} timesteps".format(t + 1))
                self.assertTrue((t + 1) == 6)
                break

    def test_large_griduniverse_completion_in_53_steps(self):
        """
        Test whether the agent completes the a large rectangular GridUniverse in the expected 53 steps
        """
        env = GridUniverseEnv(grid_shape=(25, 30))

        actions_to_take = [1] * 24 + [2] * 29 # 24 steps right + 29 steps down
        num_actions = len(actions_to_take)
        print('Num actions to take to get to terminal state: {}'.format(num_actions))
        for t in range(100):
            action = actions_to_take[t]
            observation, reward, done, info = env.step(action)

            if done:
                print("Episode finished after {} timesteps".format(t + 1))
                break
        self.assertTrue((t + 1) == num_actions and done)

    def test_custom_griduniverse_from_text_file(self):
        """
        Test whether we can complete the GridUniverse created from the text file within
        """

        env = GridUniverseEnv(textworld_fp='../core/envs/textworld_map_files/test_env.txt')
        actions_to_take = [2, 2, 2, 2, 2, 2, 2, 1]
        for step_no, action in enumerate(actions_to_take):
            env.render()
            print('go ' + env.action_descriptors[action])
            observation, reward, done, info = env.step(action)
            if done:
                print("Episode finished after {} timesteps".format(step_no + 1))

        self.assertTrue((step_no + 1) == len(actions_to_take) and done)

    def test_lever_textworld_created_from_text_file_with_wrong_and_right_lever_metadata(self):
        """
        Test whether we can complete this griduniverse created from a text file
        within textworld_map_files folder. This level contains levers so it also tests that the
        functionality of levers works correctly. If anything changes anywhere, this test will fail.
        """

        env = GridUniverseEnv(textworld_fp='../core/envs/textworld_map_files/lever_level_2.txt')
        env.render()

        # Very particular path to take for agent to go to each lever and finally get to terminal state
        actions_to_take = [env.action_descriptor_to_int['DOWN']] + [env.action_descriptor_to_int['RIGHT']] * 6 + \
                          [env.action_descriptor_to_int['LEFT']] * 5 + [env.action_descriptor_to_int['DOWN']] * 5 + \
                          [env.action_descriptor_to_int['RIGHT']] * 3 + [env.action_descriptor_to_int['UP']] * 2 + \
                          [env.action_descriptor_to_int['RIGHT']] * 2 + [env.action_descriptor_to_int['LEFT']] * 2 + \
                          [env.action_descriptor_to_int['DOWN']] * 2 + [env.action_descriptor_to_int['RIGHT']] * 2

        for step_no, action in enumerate(actions_to_take):
            # env.render()  # set mode='graphic' for pyglet render
            observation, reward, done, info = env.step(action)

            if done:
                print("Episode finished after {} timesteps".format(step_no + 1))
                break

        self.assertTrue((step_no + 1) == len(actions_to_take) and done)

    def test_lever_textworld_breaks_from_reading_wrong_lever_metadata_from_text_file(self):
        """
        Creates temporary file with correct map/grid text but wrong lever metadata (broken python dictionary in text) e.g. "{5: {"
        Final case tests empty dictionary which should not crash.
        """

        file_path = '../core/envs/textworld_map_files/test_text_file_with_broken_metadata.txt'

        # Test wrong format of metadata (list) breaks loading
        with self.assertRaises(TypeError):
            with open(file_path, 'w') as file:
                file.write('oooo\n')
                file.write('oooo\n')
                file.write('oooo\n')
                file.write('oooG\n')
                file.write('----\n')
                file.write('[5]')
            env = GridUniverseEnv(textworld_fp=file_path)

        # Test totally broken syntax
        with self.assertRaises(TypeError):
            with open(file_path, 'w') as file:
                file.write('oooo\n')
                file.write('oooo\n')
                file.write('oooo\n')
                file.write('oooG\n')
                file.write('----\n')
                file.write('----\n')
                file.write('{5: {')
            env = GridUniverseEnv(textworld_fp=file_path)

        # How it should work but empty dictionary. So no crash expected.
        try:
            with open(file_path, 'w') as file:
                file.write('oooo\n')
                file.write('oooo\n')
                file.write('oooo\n')
                file.write('oooG\n')
                file.write('----\n')
                file.write('{}')
            env = GridUniverseEnv(textworld_fp=file_path)
        except:
            self.fail("Should not crash here. Correct map file.")

        # Remove file. Leaving it there would only confuse and clutter.
        os.remove(file_path)

    def test_each_boundary_within_default_env(self):
        """
        On default env, the agent follows a sequence of steps to check each boundary acts as expected (the current
        observation should be the same as the previous if you move into a boundary).
        The agent tries the top-left, top-right and bottom-right corners while avoiding the Terminal state.
        The step numbers where the agent ends up in the same state as previously
        are stored and then compared to the expected values and if exactly the same the test passes.
        """
        env = GridUniverseEnv()

        # self.action_descriptors = ['up', 'right', 'down', 'left']
        actions_to_take = [3, 0, 1, 1, 1, 1, 2, 2, 3, 2, 2, 3, 3, 3]
        boundary_test_step_numbers = [0, 1, 5, 10, 13]
        collected_boundary_step_numbers = []
        prev_observation = env.reset()
        for step_no, action in enumerate(actions_to_take):
            env.render()
            print('go ' + env.action_descriptors[action])
            observation, reward, done, info = env.step(action)

            if observation == prev_observation:
                collected_boundary_step_numbers.append(step_no)

            prev_observation = observation

        print('collected_boundary_steps:', collected_boundary_step_numbers)
        print('boundary_test_steps', boundary_test_step_numbers)
        boolean_elementwise_comparison = [a == b for a, b in zip(collected_boundary_step_numbers, boundary_test_step_numbers)]
        print(boolean_elementwise_comparison)
        print(all(boolean_elementwise_comparison))
        self.assertTrue(all(boolean_elementwise_comparison))

    def test_lava(self):
        """
        Run agent into lava, test to see if episode ends with negative reward
        """

        env = GridUniverseEnv(lava_states=[1])

        env.render()
        action = env.action_descriptor_to_int['RIGHT']
        observation, reward, done, info = env.step(action)

        self.assertTrue(reward == -10 and done)

    def test_lava_works_from_text_file(self):
        """
        Test whether we can end the episode by making the agent travel into lava in environment created from text file
        """

        env = GridUniverseEnv(textworld_fp='../core/envs/textworld_map_files/test_env.txt')
        actions_to_take = [env.action_descriptor_to_int[action_desc] for action_desc in ['DOWN', 'DOWN', 'DOWN', 'RIGHT', 'RIGHT']]
        for step_no, action in enumerate(actions_to_take):
            env.render()
            print('go ' + env.action_descriptors[action])
            observation, reward, done, info = env.step(action)
            if done:
                print("Episode finished after {} timesteps".format(step_no + 1))

        self.assertTrue(reward == -10 and done)


if __name__ == '__main__':
    unittest.main()

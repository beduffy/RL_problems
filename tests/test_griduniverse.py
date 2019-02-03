import unittest

from core.envs.griduniverse_env import GridUniverseEnv


class TestGridUniverse(unittest.TestCase):
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
        self.assertTrue(observation == 0) # check if in same place

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

        env = GridUniverseEnv(custom_world_fp='../core/envs/maze_text_files/test_env.txt')
        actions_to_take = [2, 2, 2, 2, 2, 2, 2, 1]
        for step_no, action in enumerate(actions_to_take):
            env.render()
            print('go ' + env.action_descriptors[action])
            observation, reward, done, info = env.step(action)
            if done:
                print("Episode finished after {} timesteps".format(step_no + 1))

        self.assertTrue((step_no + 1) == len(actions_to_take) and done)

    def test_each_boundary_within_default_env(self):
        """
        The agent follows a sequence of steps to check each boundary acts as expected (the current
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

        env = GridUniverseEnv(custom_world_fp='../core/envs/maze_text_files/test_env.txt')
        actions_to_take = [env.action_descriptor_to_int[action_desc] for action_desc in ['DOWN', 'DOWN', 'DOWN', 'RIGHT', 'RIGHT']]
        for step_no, action in enumerate(actions_to_take):
            env.render()
            print('go ' + env.action_descriptors[action])
            observation, reward, done, info = env.step(action)
            if done:
                print("Episode finished after {} timesteps".format(step_no + 1))

        self.assertTrue(reward == -10 and done)

    def test_lemons_melons_apples(self):
        """
        Test whether correct cumulative reward is received for collecting certain number of melons, lemons and apples.
        In places where there are no fruit, we lose -1 immediate reward (self.MOVEMENT_REWARD)
        """

        env = GridUniverseEnv(custom_world_fp='../core/envs/maze_text_files/melons_lemons_apples_env.txt')
        actions_to_take = [env.action_descriptor_to_int[action_desc] for action_desc in
                           ['RIGHT', 'RIGHT', 'RIGHT', 'RIGHT', 'RIGHT', 'RIGHT',
                            'DOWN', 'LEFT', 'LEFT', 'DOWN', 'RIGHT', 'RIGHT']]

        num_lemons = num_apples = num_melons = 3
        expected_total_reward = num_apples * env.APPLE_REWARD + num_lemons * env.LEMON_REWARD + \
                                num_melons * env.MELON_REWARD + len(actions_to_take) * env.MOVEMENT_REWARD  # -1 for immediate reward
        cumulative_reward = 0

        for step_no, action in enumerate(actions_to_take):
            env.render()
            print('go ' + env.action_descriptors[action])
            observation, reward, done, info = env.step(action)
            cumulative_reward += reward

        print(cumulative_reward, expected_total_reward)
        self.assertTrue(cumulative_reward == expected_total_reward)

    def test_object_duplicates_raise_exceptions(self):
        """
        If any of the specific objects/entities have duplicates make sure an exception is raised
        """
        with self.assertRaises(ValueError):
            env = GridUniverseEnv(lemons=[5, 5])

        with self.assertRaises(ValueError):
            env = GridUniverseEnv(melons=[5, 5])

        with self.assertRaises(ValueError):
            env = GridUniverseEnv(apples=[5, 5])

        with self.assertRaises(ValueError):
            env = GridUniverseEnv(walls=[5, 5])

        with self.assertRaises(ValueError):
            env = GridUniverseEnv(initial_state=[5, 5])

        with self.assertRaises(ValueError):
            env = GridUniverseEnv(lava_states=[5, 5])

        with self.assertRaises(ValueError):
            env = GridUniverseEnv(goal_states=[5, 5])

    def test_different_object_collisions_raise_exceptions(self):
        """
        Check if specific object combinations collide i.e. two objects in the same state index.
        There are probably more to add since it's 2^n with n being the amount of objects.
        But these are the most important.

        Current not testing (and no need to test (allow these situations to be the mistake of the user))
        these collisions:

        initial_state <-> lava (ok since the user will see what his mistake was)
        initial_state <-> goal_state
        """

        # Check if starting state is within a wall
        with self.assertRaises(ValueError):
            env = GridUniverseEnv(initial_state=5, walls=[5])

        # check if goal state is within a wall
        with self.assertRaises(ValueError):
            env = GridUniverseEnv(goal_states=[5], walls=[5])

        # check if lava state is within a wall
        with self.assertRaises(ValueError):
            env = GridUniverseEnv(lava_states=[5], walls=[5])

        # check if any fruit has been placed on wall
        with self.assertRaises(ValueError):
            env = GridUniverseEnv(melons=[5], walls=[5])

        # Check if fruit collides with another fruit
        with self.assertRaises(ValueError):
            env = GridUniverseEnv(apples=[5], lemons=[5])

    def test_fruit_disappears(self):
        """
        Test that if you collect fruit, it disappears. Apples in this case
        """

        env = GridUniverseEnv(apples=[1])

        env.render()
        action = env.action_descriptor_to_int['RIGHT']
        observation, reward1, done, info = env.step(action)

        env.render()
        action = env.action_descriptor_to_int['LEFT']
        observation, reward2, done, info = env.step(action)

        env.render()
        action = env.action_descriptor_to_int['RIGHT']
        observation, reward3, done, info = env.step(action)
        env.render()

        self.assertTrue(reward1 == (env.APPLE_REWARD + env.MOVEMENT_REWARD) and reward3 != (env.APPLE_REWARD + env.MOVEMENT_REWARD))

    def test_fruit_reset(self):
        """
        Test that after collecting fruit and the environment is reset, fruit reappears
        and that you still get same reward. Melon in this case
        """

        env = GridUniverseEnv(melons=[1])

        env.render()
        action = env.action_descriptor_to_int['RIGHT']
        observation, reward1, done, info = env.step(action)
        env.render()

        print('Resetting environment')
        env.reset()
        env.render()
        observation, reward2, done, info = env.step(action)
        env.render()

        self.assertTrue(reward1 == reward2 and reward1 == (env.MELON_REWARD + env.MOVEMENT_REWARD))

if __name__ == '__main__':
    unittest.main()

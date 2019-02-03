import time
import random

from core.envs.griduniverse_env import GridUniverseEnv


def run_default_griduniverse():
    """
    Run a random agent on the default griduniverse.
    This piece of code shows the main interface to the environment. This runs in ascii format
    """

    print('\n' + '*' * 20 + 'Starting to run random agent on default GridUniverse' + '*' * 20 + '\n')
    env = GridUniverseEnv()
    for i_episode in range(1):
        observation = env.reset()
        for t in range(100):
            env.render()  # set mode='graphic for pyglet render
            action = env.action_space.sample()
            print('go ' + env.action_descriptors[action])
            observation, reward, done, info = env.step(action)

            if done:
                print("Episode finished after {} timesteps".format(t + 1))
                break


def run_griduniverse_from_text_file():
    """
    Run a random agent on an environment that was save via ascii text file.
    Check core/envs/maze_text_files for examples or the _create_custom_world_from_text() function within the environment
    """

    print('\n' + '*' * 20 + 'Creating a pre-made GridUniverse from text file and running random agent on it' + '*' * 20 + '\n')
    env = GridUniverseEnv(custom_world_fp='../core/envs/maze_text_files/test_env.txt')
    # env = GridUniverseEnv(custom_world_fp='../core/envs/maze_text_files/maze_21x21.txt')
    # env = GridUniverseEnv(custom_world_fp='../core/envs/maze_text_files/maze_101x101.txt')
    for i_episode in range(1):
        observation = env.reset()
        for t in range(1000):
            env.render(mode='graphic')
            action = env.action_space.sample()
            # print('go ' + env.action_descriptors[action])
            # time.sleep(0.1) # uncomment to watch slower
            observation, reward, done, info = env.step(action)
            if done:
                print("Episode finished after {} timesteps".format(t + 1))
                break


def run_random_maze(grid_shape=(11, 11)):
    """
    Run a random agent on a randomly generated maze. If random_maze parameter is set to True,
    a maze generation algorithm will place walls to form the maze in the requested shape.
    """

    print('\n' + '*' * 20 + 'Creating a random GridUniverse and running random agent on it' + '*' * 20 + '\n')
    env = GridUniverseEnv(grid_shape=grid_shape, random_maze=True)
    for i_episode in range(1):
        observation = env.reset()
        for t in range(1000):
            env.render(mode='graphic')
            env.step_num = t
            action = env.action_space.sample()
            # print('go ' + env.action_descriptors[action])
            observation, reward, done, info = env.step(action)
            if done:
                print("Episode finished after {} timesteps".format(t + 1))
                break


def run_griduniverse_with_lava():
    """
    Run a random agent on an environment with lava
    """

    print('\n' + '*' * 20 + 'Starting to run random agent on GridUniverse with lava' + '*' * 20 + '\n')
    env = GridUniverseEnv(grid_shape=(10, 10), lava_states=[4, 14, 24, 34, 44, 54, 64, 74])
    for i_episode in range(5):
        observation = env.reset()
        for t in range(100):
            env.render(mode='graphic')  # set mode='graphic for pyglet render
            action = env.action_space.sample()
            observation, reward, done, info = env.step(action)

            if done:
                print("Episode finished after {} timesteps".format(t + 1))
                print('Final states reward: ', reward)
                break

def run_griduniverse_filled_with_fruit(grid_shape=(30, 30), fill_mode_random=True, random_actions=True):
    """
    Run a random agent on an environment with fruit. Fill whole grid with fruit.

    :param world_shape: the shape of the
    :param fill_mode_random: if True the type of fruit is placed randomly on the grid.
                             If False, it is placed in grids,
    """

    print('\n' + '*' * 20 + 'Starting to run random agent on GridWorld with fruit' + '*' * 20 + '\n')
    world_size = grid_shape[0] * grid_shape[1]

    num_of_each_fruit = world_size // 3

    if fill_mode_random:
        all_indices = list(range(0, world_size))
        apples = [all_indices.pop(random.randrange(len(all_indices))) for _ in range(num_of_each_fruit)]
        lemons = [all_indices.pop(random.randrange(len(all_indices))) for _ in range(num_of_each_fruit)]
        melons = [all_indices.pop(random.randrange(len(all_indices))) for _ in range(num_of_each_fruit)]
    else:
        apples = [i for i in range(world_size // 3)]
        lemons = [i for i in range(world_size // 3, int(2 * world_size // 3))]
        melons = [i for i in range(int(2 * world_size // 3), world_size)]

    env = GridUniverseEnv(grid_shape=grid_shape, initial_state=world_size//2 + grid_shape[0], apples=apples, melons=melons, lemons=lemons)
    string_actions_to_take = (['DOWN'] * grid_shape[1] + ['RIGHT'] + ['UP'] * grid_shape[1] + ['RIGHT']) * grid_shape[1]

    actions_to_take = [env.action_descriptor_to_int[action_desc] for action_desc in string_actions_to_take]
    for i_episode in range(5):
        observation = env.reset()
        for t in range(1000):
            env.render(mode='graphic')  # set mode='graphic for pyglet render

            if random_actions:
                action = env.action_space.sample()
            else:
                action = actions_to_take[t]
            observation, reward, done, info = env.step(action)
            if done:
                print("Episode finished after {} timesteps".format(t + 1))
                print('Final states reward: ', reward)
                break

if __name__ == '__main__':
    # Run random agent on environment variations
    run_default_griduniverse()
    run_griduniverse_from_text_file()
    run_random_maze()
    run_griduniverse_filled_with_fruit()  # fill_mode_random=False if you want different fruit to be placed in order
    run_griduniverse_with_lava()

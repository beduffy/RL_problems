import time

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
    Check core/envs/textworld_map_files for examples or the _create_custom_world_from_text() function within the environment
    """

    print('\n' + '*' * 20 + 'Creating a pre-made GridUniverse from text file and running random agent on it' + '*' * 20 + '\n')
    env = GridUniverseEnv(textworld_fp='../core/envs/textworld_map_files/test_env.txt')
    # env = GridUniverseEnv(textworld_fp='../core/envs/textworld_map_files/maze_21x21.txt')
    # env = GridUniverseEnv(textworld_fp='../core/envs/textworld_map_files/maze_101x101.txt')
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


def run_random_maze():
    """
    Run a random agent on a randomly generated maze. If random_maze parameter is set to True,
    a maze generation algorithm will place walls to form the maze in the requested shape.
    """

    print('\n' + '*' * 20 + 'Creating a random GridUniverse and running random agent on it' + '*' * 20 + '\n')
    env = GridUniverseEnv(grid_shape=(11, 11), random_maze=True)
    # env = GridUniverseEnv(grid_shape=(101, 101), random_maze=True)
    # env = GridUniverseEnv(grid_shape=(49, 51), random_maze=True)
    # env = GridUniverseEnv(grid_shape=(51, 49), random_maze=True)
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

    print('\n' + '*' * 20 + 'Starting to run random agent on default GridUniverse' + '*' * 20 + '\n')
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

def run_lever_griduniverse():
    """
    Run a random agent on a GridUniverse with a lever and a doors
    """

    # test if random agent ever opens door to terminal and then goes to terminal
    print('\n' + '*' * 20 + 'Starting to run random agent on GridUniverse with levers' + '*' * 20 + '\n')
    env = GridUniverseEnv((7, 7), walls=[47, 47 - 7, 48 - 7], levers={42: 47})
    for i_episode in range(1):
        observation = env.reset()
        for t in range(1000000):
            env.render(mode='graphic')  # set mode='graphic' for pyglet render
            action = env.action_space.sample()
            observation, reward, done, info = env.step(action)

            if done:
                env.render(mode='graphic')
                time.sleep(3)
                print("Episode finished after {} timesteps".format(t + 1))
                break

def run_lever_griduniverse_from_text_file():
    # env = GridUniverseEnv(textworld_fp='../core/envs/textworld_map_files/lever_level_1.txt')
    env = GridUniverseEnv(textworld_fp='../core/envs/textworld_map_files/lever_level_2.txt')
    # env = GridUniverseEnv(textworld_fp='../core/envs/textworld_map_files/lever_level_3.txt')

    for i_episode in range(1):
        observation = env.reset()
        for t in range(1000000):
            #env.render()  # set mode='graphic' for pyglet render
            env.render(mode='graphic')  # set mode='graphic' for pyglet render
            action = env.action_space.sample()
            observation, reward, done, info = env.step(action)

            if done:
                env.render(mode='graphic')
                time.sleep(3)
                print("Episode finished after {} timesteps".format(t + 1))
                break

if __name__ == '__main__':
    # Run random agent on environment variations
    run_default_griduniverse()
    run_griduniverse_from_text_file()
    run_random_maze()
    run_griduniverse_with_lava()
    run_lever_griduniverse()
    run_lever_griduniverse_from_text_file()

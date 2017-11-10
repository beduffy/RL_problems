import gym
import time
import sys


def sample_from_policy1(state, previous_action):
    """
    state is defined as a 4-length numpy array
    cart position
    cart velocity
    pole angle
    pole velocity at tip
    """
    if previous_action:
        next_action = 0
    else:
        next_action = 1

    return next_action


if __name__=="__main__":
    env = gym.make('CartPole-v0')
    for i_episode in range(1):
        observation = env.reset()
        action = 1
        episode_reward = 0
        for t in range(1000):
            env.render()
            print(observation)
            # action = env.action_space.sample()
            action = sample_from_policy1(state=observation, previous_action=action)
            print("action chosen: ", action)
            observation, reward, done, info = env.step(action)
            episode_reward += reward
            time.sleep(0.3)

            if done:
                print("Episode finished after {} timesteps".format(t+1))
                break



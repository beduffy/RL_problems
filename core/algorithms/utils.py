import sys

import numpy as np
from six import StringIO


def reshape_as_griduniverse(input_matrix, world_shape):
    """
    Helper function to reshape a griduniverse state matrix into a visual representation of the griduniverse with origin on the
    low left corner and x,y corresponding to cartesian coordinates.
    """
    return np.reshape(input_matrix, (world_shape[0], world_shape[1]))


def single_step_policy_evaluation(policy, env, discount_factor=1.0, value_function=None):
    """
    Returns an update of the input value function using the input policy.
    """
    v = np.zeros(env.world.size) if value_function is None else value_function
    v_new = np.zeros(env.world.size)

    for state in range(env.world.size):
        v_new[state] += env.reward_matrix[state]
        for action, action_prob in enumerate(policy[state]):
            next_state, reward, done = env.look_step_ahead(state, action)
            v_new[state] += action_prob * (discount_factor * v[next_state])
    return v_new


def get_policy_map(policy, world_shape, mode='human'):
    """
    Generates a visualization grid from the policy to be able to print which action is most likely from every state
    """
    unicode_arrows = np.array([u'\u2191', u'\u2192', u'\u2193', u'\u2190' # up, right, down, left
                      u'\u2194', u'\u2195'], dtype='<U1')  # left-right, up-down
    policy_arrows_map = np.empty(policy.shape[0], dtype='<U4')
    for state in np.nditer(np.arange(policy.shape[0])):
        # find index of actions where the probability is > 0 # todo could be cleaner
        optimal_actions = np.where(np.around(policy[state], 8) > np.around(np.float64(0), 8))[0]
        # match actions to unicode values of the arrows to be displayed
        for action in optimal_actions:
            policy_arrows_map[state] = np.core.defchararray.add(policy_arrows_map[state], unicode_arrows[action])
    policy_probabilities = np.empty(policy.shape[0], dtype=np.object)
    policy_probabilities[:] = [policy[state] for state in np.nditer(np.arange(policy.shape[0]))]
    outfile = StringIO() if mode == 'ansi' else sys.stdout
    for row in reshape_as_griduniverse(policy_arrows_map, world_shape):
        for state in row:
            outfile.write((state + u'  '))
        outfile.write('\n')
    outfile.write('\n')

    return policy_arrows_map, reshape_as_griduniverse(policy_probabilities, world_shape)


def greedy_policy_from_value_function(policy, env, value_function, discount_factor=1.0):
    """
    Returns a greedy policy based on the input value function.

    If no value function was provided the defaults from a single step starting with a value function of zeros
    will be used.
    """
    q_function = np.zeros((env.world.size, env.action_space.n))
    for state in range(env.world.size):
        for action in range(env.action_space.n):
            next_state, reward, done = env.look_step_ahead(state, action)
            q_function[state][action] += reward + discount_factor * value_function[next_state]
        max_value_actions = np.where(np.around(q_function[state], 8) == np.around(np.amax(q_function[state]), 8))[0]

        policy[state] = np.fromiter((1 / len(max_value_actions) if action in max_value_actions and
                                    not env.is_terminal(state) else 0
                                     for action in np.nditer(np.arange(env.action_space.n))), dtype=np.float)
    return policy

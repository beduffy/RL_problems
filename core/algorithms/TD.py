import sys

import numpy as np

from core.envs.gridworld_env import GridWorldEnv
from core.algorithms import utils


# Hyperparameters
alpha = 0.01
discount_factor = 0.99 # todo change to 1 to test equivalence
online = True
lambda_factor = 0.9 # todo change to 1 to test equivalence
lambda_return_mode = True
n = 5 # set to -1 for Monte Carlo # todo shouldn't work with online == True and lambda_return_mode == True?

def calculate_n_step_return(n_steps, from_step_num, all_states, all_rewards):
    """
    :param n_steps:
    :param from_idx:
    :param all_states:
    :param all_rewards:
    :return: n_step_return
    """
    # todo pass in value func.
    # todo pass in values better. stop using globals
    # todo instead of complex from_idx, n indexing, I could just pass the right arrays as parameters
    # todo assert that that from_idx + final n == number of steps in episode
    # todo rename n to n_global_hyperparameter to avoid confusino and check everywhere I use
    # if (discount_factor ** i) < threshold]) todo or not

    if n_steps == 0:
        raise ValueError('No such thing as 0-step return')

    if n_steps == -1 and not lambda_return_mode: # todo check and cleaner
        # MC uses all rewards from from_step_num
        first_n_rewards = all_rewards[from_step_num:]
    else:
        first_n_rewards = all_rewards[from_step_num:from_step_num + n_steps]

    print('\nFrom step index: {}'.format(from_step_num))
    print('First N = {} rewards: {}'.format(n_steps, first_n_rewards))
    # print('Value of St+{}: {}, value of St: {}'.format(n_steps, value_func[all_states[from_idx + n_steps]], value_func[all_states[from_idx]]))
    discounted_immediate_rewards = sum([(discount_factor ** i) * r for i, r in enumerate(first_n_rewards)])
    # if from_step + n goes over number of states, stop bootstrapping/predicting value
    # and only use correct amount of rewards e.g. 5-step return uses 5 rewards and predicted value
    if n_steps != -1 and from_step_num + n_steps < len(all_states):
        discounted_future_value = (discount_factor ** n_steps) * value_func[all_states[from_step_num + n_steps]]
    elif n_steps == -1: # todo better else structure
        # MC doesn't estimate future
        discounted_future_value = 0.0
    else:
        print('More steps than steps left in episode')
        discounted_future_value = 0.0
    n_step_return = discounted_immediate_rewards + discounted_future_value
    print('discounted_immediate_rewards:', discounted_immediate_rewards)
    print('discounted_future_value:', discounted_future_value)
    print('{}_step_return: {} for index: {}'.format(n_steps, n_step_return, from_step_num))

    return n_step_return

if __name__ == '__main__':
    # TD Learning
    # V(St) ← V(St) + alpha * (Rt+1 + lambda * V(St+1) − V(St))

    world_shape = (4, 4)
    env = GridWorldEnv(world_shape=world_shape)

    value_func = np.zeros(env.world.size)
    eligibility_traces = np.zeros(env.world.size)
    print('Initial value function:', value_func)

    # Forward-view TD(lambda)
    # 1. Store each state and reward in episode
    # 2. At end of episode (offline) compute, for each step, lambda-return
    #    2.1. 1-step return, 2-step return, ..., all-step return (MC)

    # Make agent act randomly and evaluate policy
    for i_episode in range(17):
        curr_state = env.reset()

        all_states = [curr_state]
        all_rewards = []

        # todo replacing or accumulating traces?
        eligibility_traces[curr_state] = discount_factor * lambda_factor * eligibility_traces[curr_state] + 1 # todo check

        print('\n\n\nSTARTING NEXT EPISODE!!!!!!!!!!!!!!!!!!!!!!!!! \n\n\n')
        print(value_func)
        print(eligibility_traces)

        for t in range(100):
            action = env.action_space.sample()
            curr_state, reward, done, info = env.step(action)
            all_states.append(curr_state)
            all_rewards.append(reward)

            # G(n)t = Rt+1 + γRt+2 + ... + γ ^ n−1 * Rt+n + γ ^ n V(St+n)
            # V(St) ← V(St) + α * (G(n)t − V(St))
            # n = 1 = TD(0)
            # G(1)t = Rt+1 + γ * V(St+1)
            # V(St) ← V(St) + α * (Rt+1 + γ * V(St+1) − V(St))
            # n = 2
            # G(2)t = Rt+1 + γ * Rt+2 + γ ^ 2 * V(St+2)
            # so we have to wait 2 steps at the start before calculating anything.
            # And then every step, we do the above.

            # online n-step return
            if online:
                if lambda_return_mode:
                    eligibility_traces[curr_state] = eligibility_traces[curr_state] + 1  # todo check
                    eligibility_traces = discount_factor * lambda_factor * eligibility_traces
                    n_step_return = calculate_n_step_return(1, t, all_states, all_rewards) # todo check 1-step return

                    # for state in

                    TD_error = n_step_return - value_func[all_states[t]]
                    # value_func[all_states[t]] = value_func[all_states[t]] + alpha * TD_error * eligibility_traces[curr_state]
                    value_func = value_func + alpha * TD_error * eligibility_traces
                else: # todo fix to elif?
                    if n != -1 and len(all_rewards) > n:
                        # calculate n-step return for state n-steps ago
                        n_step_return = calculate_n_step_return(n, t - n, all_states, all_rewards)

                        TD_error = n_step_return - value_func[all_states[t - n]]
                        value_func[all_states[t - n]] = value_func[all_states[t - n]] + alpha * TD_error

            if done:
                if online:
                    # todo DRY
                    # We need to go into the future to get rewards and sometimes the future is the end of the episode
                    # hence offline (episode is over), we have to calculate the returns for each state that wasn't covered.
                    # e.g. 10 step episode with 5-step return, if there are only 3 steps left,
                    # we wait until the episode is over to calculate returns for the final 3 states/steps.
                    if not lambda_return_mode:
                        if n == -1: # Monte-carlo
                            offline_step_range = range(len(all_rewards))
                        else:
                            offline_step_range = range(t - n + 1, len(all_rewards))
                        for step_no in offline_step_range:
                            n_step_return = calculate_n_step_return(n, step_no, all_states, all_rewards)

                            TD_error = n_step_return - value_func[all_states[step_no]]
                            value_func[all_states[step_no]] = value_func[all_states[step_no]] + alpha * TD_error

                else: # not online
                    # 2. At end of episode (offline) compute, for each step, calculate lambda-return or n-step return
                    #    2.1. 1-step return, 2-step return, ..., all-step (inf) return (MC)
                    print('episode over, calculating returns offline')
                    print('All states: {}'.format(all_states))
                    print('All rewards: {}'.format(all_rewards))

                    # todo test case for averaging 1-step and 2-step return

                    for start_idx, state in enumerate(all_states):
                        if lambda_return_mode:
                            lambda_return_sum = 0
                            for n_steps, s in enumerate(all_states[start_idx:]):
                                # no such thing as 0-step
                                if n_steps == 0: # begin at 1
                                    continue
                                # 1st parameter: n_steps is 1-indexed e.g. 1-step, 2-step etc.
                                # start_idx == current_state. start_idx + n_steps == value_of_nth_state
                                lambda_return_sum += (lambda_factor ** (n_steps - 1)) * calculate_n_step_return(n_steps, start_idx, all_states, all_rewards)

                            # (1 - lambda_factor) is multiplied from outside sum
                            lambda_return = (1 - lambda_factor) * lambda_return_sum
                            TD_error = lambda_return - value_func[state]
                        else:
                            n_step_return = calculate_n_step_return(n, start_idx, all_states, all_rewards)
                            TD_error = n_step_return - value_func[state]
                            # only calculate one n_step_return (e.g. 5 steps, look 5 steps)

                        value_func[state] = value_func[state] + alpha * TD_error # update towards error

                print("Episode finished after {} timesteps".format(t + 1))
                break  # next episode

    print('Final value function:', value_func)
    #sys.exit()

    # Test algorithm by creating greedy policy from value function
    # todo change so we don't need or always fill policy till end. Or... always have value as 0 for unseen states. probably ok.
    policy0 = np.ones([env.world.size, len(env.action_state_to_next_state)]) / len(env.action_state_to_next_state)
    policy1 = utils.greedy_policy_from_value_function(policy0, env, value_func)
    # print(policy1)

    policy_map1 = utils.get_policy_map(policy1, world_shape)
    # print('Policy: (up, right, down, left)\n', policy_map1)

    print('Starting greedy policy run')
    curr_state = env.reset()
    prev_state = curr_state
    for t in range(10):
        env.render()
        # action = np.argmax(policy1[curr_state])
        action = np.random.choice(policy1[curr_state].size, p=policy1[curr_state]) # take uniform choice if equal
        print('go ' + env.action_descriptors[action])
        curr_state, reward, done, info = env.step(action)

        if done:
            print('DONE in {} steps'.format(t + 1))
            break

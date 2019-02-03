# RL_problems

This repository was created with the goal of developing and reproducing many different Reinforcement Learning algorithms 
and then being able to run large scale experiments using these algorithms on different open-source 
environments as well as our own custom environments.  

Currently working exclusively on OpenAI gym and a grid world environment made by us, which we called "GridUniverse".

## GridUniverse

We have created a highly customisable world framed in a discrete state space OpenAI gym environment to enable research and large scale testing, evaluation and comparison of new and old RL algorithms. 
Eventually, it will also allow large scale experimentation in exciting new areas like Language Grounding, Meta-Learning and Exploration.

![GIF](docs/maze_solver_BFS_10_times.gif)

<p align="center" style="width: 100%;">
    <img src="docs/SUPER-COLLAGE-ACTUAL-GIF.gif" />
</p>

## GridUniverse features and plans

While there have been plenty of grid world and maze RL environments created in the past, 
none have attempted as large a feature list as planned for this one. 
The goal is to contain this extensive list of features and algorithms __all within one place__. This will enable groundbreaking research while also being a very useful educational source for any level of expertise (beginners to advanced).
 
This environment defines a bidimensional grid representing the world as a discrete state and action space which includes the following entities that can be customized:
- [x] Wall states. Blocked states.
- [x] Goal and Lava states, terminal states with a high positive and negative reward respectively.
- [x] Random maze generator, with multiple different maze generation algorithms.
- [x] “Classic” pathfinding/graph search algorithms to compare against RL algorithms.
- [x] ASCII and OpenGL based rendering (using pyglet)
- [x] An easy interface to create and store levels in text files
- [x] Levers and keys. Elements that modify the environment by removing particular walls/doors.

Additionally, we plan to include the following features:
- [ ] 3 different “fruits” that can be collected. Objects that can provide positive and negative rewards in varying amounts. 
- [ ] Wind
- [ ] Human control of the agent
- [ ] Different sensor configurations for the agent so it can be defined whether the agent field of view is constrained to the current state, surrounding states or the complete grid.
- [ ] Natural language grounding. Implementation of algorithms to follow natural language instructions e.g. “Collect the lemon and then the apple”
- [ ] Meta-Learning/Multi-task learning environment interface and algorithms run on a large number of levels of varying difficulty
- [ ] Sokoban extension. An unsolved state of the art AI problem based on a classic video game.

Using David Silver's Reinforcement Learning course and Sutton and Barto (2018) as a reference,
there are a number of algorithms we have implemented from scratch and plan to implement soon. 
These can then be compared and benchmarked against each other on this environment. 
These algorithms are:
- [x] Policy Iteration
- [x] Value Iteration
- [x] Monte-Carlo (MC) Learning with variations
- [ ] Temporal Difference (TD) Learning with variations
- [ ] Value Approximation
- [ ] On-policy control (SARSA) 
- [ ] Off-policy control (Q-Learning, Importance Sampling)
- [ ] Policy Gradients (MC Policy Gradients and Actor-critic)
- [ ] Integrating learning and planning (Dyna, MC/TD Tree search, Forward and Simulation-based search)
- [ ] Exploration vs Exploitation (Optimistic policy, optimistic policy with uncertainty, Thompson sampling, UCB)

## Running GridUniverse

To run examples of algorithms like policy/value iteration or Monte Carlo evaluation on our GridUniverse environment:  

`python examples/griduniverse_alg_examples.py`

To run random agents on different variations of the environment (with different features):

`python examples/griduniverse_env_examples.py`

The `run_default_griduniverse()` function in the above file shows the simplest way to use the environment.  

For much more detailed info on how to use the environment, check the [GridUniverse Documentation](https://github.com/beduffy/RL_problems/tree/master/docs/GridUniverse.md)

To run our tests:  

`python tests/test_griduniverse.py`

## Installation

Follow the instructions at this link: [Installation instructions](https://github.com/beduffy/RL_problems/tree/master/docs/Installation.md)

## Running OpenAI gym

Now you should be able to run: 

`python examples/atari_example.py`

You should see a small window that automatically plays "Space Invaders" if everything is working correctly.

For a general documentation on how the environment works refer to the [official OpenAI documentation](https://gym.openai.com/docs).

## API Reference

You can look within the docs, examples and tests folders for more info on all aspects of this repo. 

## Contributors

Currently not looking for help from contributors. Further information to be added in the future.

## License

For information about the license of this code please refer to the corresponding file "license.txt"

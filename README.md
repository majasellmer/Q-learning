# Q-learning

A supplement to the Bachelor thesis "On convergence of Q-learning algorithms in reinforcement learning".

code written using:
* Python version 3.7.9
* matplotlib version 3.5.1
* numpy version 1.21.5

This example is inspired by but different from the Frozen Lake environment provided in OpenAI's Gymnasium.

problem setup:
* A maze of 5x5 squares, some of which contain obstacles/walls.
* The goal is for the agent to learn to solve the labyrinth in as few steps as possible using Q-learning.
* Each state corresponds to one state in the process, we shall number them from left to right and top to bottom from 0 to 24. The agent starts out in the top left corner, and the bottom right corner is the absorbing state.
* The possible actions at each step are a subset of the actions {left, right, up, down}, with the condition that the agent cannot step outside the board or into a wall. There is a strong south wind, so there is a chance of being pushed one square up or unable to move, unless there is a wall shielding the agent from the south. Moving to the goal state 24 is exempt from this, i.e. moving right from state 23 or down from state 19 always succeeds.

resulting plots:
* the original maze
* learned optimal actions after 200 episodes
* total reward over number of episodes
	
choice of parameters which can be adjusted:
* number of learning episodes: num_epochs = 200
* obstacles: six possible configurations saved in the list obstacle_combinations
* epsilon_decay_factor = 0.01**(1/(num_epochs\*100)), resulting in the probability for exploration max(0.01,0.01**(1/(epochs\*100)))
* learning rates: alpha[action] = 1/(num_visits[current_state][action]+1)
* discount factor: gamma = 1
* probability of being thrown off by the wind: p = 0.1
* cost per step: distributed according to np.random.normal(-1,0.1)
* reward for reaching goal state: 14

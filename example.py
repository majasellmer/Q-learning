"""Example of solving a maze using Q-learning
   Zusatz zur Bachelorarbeit
   16.07.2023
   author: Maja Sellmer
"""

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

# choose the maze to be solved
# calculate which states are ''sheltered'' from the wind
# determine possible actions for each state
obstacle_combinations = [[8,9,10,17,18],[5,6,13,17,18],[8,13,18,16,21],
    [6,7,12,13,15],[9,11,16,17,18],[7,8,12,13,15]]
n = np.random.randint(0,6)
obstacles = obstacle_combinations[n]
obstacle_coords = []
sheltered = []
possible_actions = []
for obstacle in obstacles:
    i = obstacle // 5
    j = obstacle - 5*i
    obstacle_coords.append([i,j])
    for k in range(i):
        sheltered.append(5*k+j)
for i in range(5):
    for j in range(5):
        state = 5*i+j
        possible_actions_state = []
        if j != 0 and state-1 not in obstacles:
            possible_actions_state.append(0)
        if j != 4 and state+1 not in obstacles:
            possible_actions_state.append(1)
        if i!= 0 and state-5 not in obstacles:
            possible_actions_state.append(2)
        if i!= 4 and state+5 not in obstacles:
            possible_actions_state.append(3)
        possible_actions.append(possible_actions_state)

# plot the maze
x = range(5)
y = range(5)
Z = np.zeros((5,5))
Z[0][0]=0.5
Z[4][4]=0.5
for obstacle in obstacle_coords:
    Z[obstacle[0]][obstacle[1]] = 1
fig, ax = plt.subplots()
ax.invert_yaxis()
ax.get_xaxis().set_ticks([])
ax.get_yaxis().set_ticks([])
ax.set_aspect('equal', adjustable='box')
ax.pcolormesh(x, y, Z, cmap='Accent')
plt.text(-0.32, 0.1, "start", size=12, c='white')
plt.text(3.7, 4.1, "goal", size=12, c='white')
plt.show()

def best_action_and_reward(s, Q):
    """
    Finds the maximum in row Q[s] and correspondning action.

    Parameters
    ----------
    s : int in [0,24]
    Q : numpy.ndarray of size (25,4)

    Returns
    -------
    best_action : int in [0,3]
    best_reward : float
    """
    best_action = possible_actions[s][0]
    best_reward = Qtable[s][possible_actions[s][0]]
    for a in possible_actions[s]:
        if Q[s][a] > best_reward:
            best_action = a
            best_reward = Q[s][a]
    return best_action, best_reward

def generate_next_state(s,a):
    """
    Generates next state based on current state and action chosen.

    Parameters
    ----------
    s : int in [0,24]
    a : int in [0,3]

    Returns
    -------
    s' : int in [0,24]
    """
    # 0.1 probability of being thrown off by the wind
    p = np.random.random()
    if s in sheltered: p = 1
    if p < 0.1:
        if s < 5 or s-5 in obstacles: return s
        return s-5
    #left
    if a == 0: return s-1
    #right
    if a == 1: return s+1
    #up
    if a == 2: return s-5
    #down
    if a == 3: return s+5

# initialize parameters
Qtable = np.zeros((25,4))
starting_state = 0
goal_state = 24
gamma = 1
t = 0
num_epochs = 200
epsilon = 1
epsilon_decay_factor = 0.01**(1/(num_epochs*100))
total_rewards = []
history_counter = np.zeros((25,4))

# execute the algorithm
for epoch in range(num_epochs):
    step_counter = 0
    total_reward = 0
    current_state = starting_state
    while current_state != goal_state:
        # generate learning rates
        alpha = np.zeros(4)
        for action in possible_actions[current_state]:
            alpha[action] = 1/(history_counter[current_state][action]+1)
        # choose action
        best_action, best_reward = best_action_and_reward(current_state, Qtable)
        p = np.random.random()
        if p < epsilon:
            action = np.random.choice(possible_actions[current_state])
        else:
            action = best_action
        # generate reward and next state
        reward = np.random.normal(-1,0.1)
        next_state = generate_next_state(current_state, action)
        if (current_state == 23 and action==1) or (current_state==19 and action==3):
            reward += 14
            next_state = goal_state
        # update Q values
        Qtable[current_state][action] *= (1-alpha[action])
        Qtable[current_state][action] += alpha[action]*(reward+gamma*max(Qtable[next_state]))
        # update number of visits
        history_counter[current_state][action] += 1
        # decrease epsilon
        epsilon = max(0.01, epsilon*epsilon_decay_factor)
        # transition to next state
        current_state = next_state
        t += 1
        step_counter += 1
        total_reward += reward
    total_rewards.append(total_reward)

print(Qtable)
# plot optimal actions and expected rewards
x = range(5)
y = range(5)
Z = np.zeros((5,5))
directions = np.zeros((5,5))
for i in range(5):
    for j in range(5):
        state = 5*i+j
        if state in obstacles:
            Z[i][j] = -4
        else:
            best_action, best_reward = best_action_and_reward(state, Qtable)
            Z[i][j] = best_reward
            directions[i][j] = best_action
Z[4][4] = 14
fig, ax = plt.subplots()
ax.invert_yaxis()
ax.get_xaxis().set_ticks([])
ax.get_yaxis().set_ticks([])
ax.set_aspect('equal', adjustable='box')
plot = ax.pcolormesh(x, y, Z, cmap='Greens')
fig.colorbar(plot)
for i in range(5):
    for j in range(5):
        if 5*i+j in obstacles or (i==4 & j==4):
            continue
        if directions[i][j] == 0:
            ax.add_artist(mpatches.Arrow(j+0.2,i,-0.4,0, width=0.2, color='white'))
        elif directions[i][j] == 1:
            ax.add_artist(mpatches.Arrow(j-0.2,i,0.4,0, width=0.2, color='white'))
        elif directions[i][j] == 2:
            ax.add_artist(mpatches.Arrow(j,i+0.2,0,-0.4, width=0.2, color='white'))
        elif directions[i][j] == 3:
            ax.add_artist(mpatches.Arrow(j,i-0.2,0,0.4, width=0.2, color='white'))
plt.show()

# plot total reward over number of episodes
plt.plot(range(num_epochs), total_rewards, color='green')
plt.hlines(0, 0, num_epochs, color='black')
plt.xlabel('number of episodes')
plt.ylabel('total cost')
plt.show()

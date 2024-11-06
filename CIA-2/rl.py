import numpy as np
import matplotlib.pyplot as plt
import random
import time

# Define the Grid Environment
class GridEnvironment:
    def __init__(self, grid_size=100, obstacle_prob=0.2, start=None, goal=None):
        self.grid_size = grid_size
        self.state_space = grid_size * grid_size
        self.action_space = 4  # Up, Down, Left, Right
        self.grid = np.zeros((grid_size, grid_size))
        self.start = start if start else (0, 0)
        self.goal = goal if goal else (grid_size - 1, grid_size - 1)
        self.create_obstacles(obstacle_prob)
        
    def create_obstacles(self, prob):
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                if random.random() < prob and (i, j) not in [self.start, self.goal]:
                    self.grid[i, j] = -1  # Obstacle marked as -1

    def is_valid_state(self, x, y):
        if x < 0 or x >= self.grid_size or y < 0 or y >= self.grid_size:
            return False
        if self.grid[x, y] == -1:
            return False
        return True

    def step(self, state, action):
        x, y = state
        if action == 0:   # Up
            x -= 1
        elif action == 1: # Down
            x += 1
        elif action == 2: # Left
            y -= 1
        elif action == 3: # Right
            y += 1
        if self.is_valid_state(x, y):
            new_state = (x, y)
            reward = 100 if new_state == self.goal else -1
            done = new_state == self.goal
        else:
            new_state = state
            reward = -10
            done = False
        return new_state, reward, done

    def reset(self):
        return self.start

# Value Iteration (DP) Algorithm
def value_iteration(env, gamma=0.9, theta=1e-6):
    value_table = np.zeros((env.grid_size, env.grid_size))
    policy_table = np.zeros((env.grid_size, env.grid_size), dtype=int)
    delta = float("inf")
    start_time = time.time()

    while delta > theta:
        delta = 0
        for x in range(env.grid_size):
            for y in range(env.grid_size):
                state = (x, y)
                if state == env.goal or env.grid[x, y] == -1:
                    continue
                v = value_table[x, y]
                values = []
                for action in range(env.action_space):
                    new_state, reward, _ = env.step(state, action)
                    new_x, new_y = new_state
                    values.append(reward + gamma * value_table[new_x, new_y])
                best_value = max(values)
                value_table[x, y] = best_value
                policy_table[x, y] = np.argmax(values)
                delta = max(delta, abs(v - best_value))

    time_taken = time.time() - start_time
    steps_taken = simulate_policy(env, policy_table)
    
    return value_table, policy_table, steps_taken, time_taken

# Q-Learning Algorithm
def q_learning(env, episodes=500, alpha=0.1, gamma=0.9, epsilon=0.1):
    q_table = np.zeros((env.grid_size, env.grid_size, env.action_space))
    start_time = time.time()

    for _ in range(episodes):
        state = env.reset()
        done = False
        while not done:
            x, y = state
            # Epsilon-greedy action selection
            if random.uniform(0, 1) < epsilon:
                action = random.randint(0, env.action_space - 1)
            else:
                action = np.argmax(q_table[x, y])

            new_state, reward, done = env.step(state, action)
            new_x, new_y = new_state
            best_next_action = np.argmax(q_table[new_x, new_y])

            # Q-learning update
            q_table[x, y, action] += alpha * (reward + gamma * q_table[new_x, new_y, best_next_action] - q_table[x, y, action])
            state = new_state

    time_taken = time.time() - start_time
    policy_table = np.argmax(q_table, axis=2)
    steps_taken = simulate_policy(env, policy_table)
    
    return q_table, policy_table, steps_taken, time_taken

# Simulate policy execution to count steps to reach the goal
def simulate_policy(env, policy_table):
    state = env.reset()
    steps = 0
    while state != env.goal:
        x, y = state
        action = policy_table[x, y]
        state, _, done = env.step(state, action)
        steps += 1
        if done or steps > env.grid_size ** 2:  # Safety break in case of no convergence
            break
    return steps if state == env.goal else float('inf')

# Visualization Function
def visualize_policy(env, policy_table, title="Policy"):
    plt.figure(figsize=(10, 10))
    grid = env.grid
    for x in range(env.grid_size):
        for y in range(env.grid_size):
            if (x, y) == env.start:
                plt.text(y, x, 'S', ha='center', va='center', color='blue', fontsize=12)
            elif (x, y) == env.goal:
                plt.text(y, x, 'G', ha='center', va='center', color='green', fontsize=12)
            elif grid[x, y] == -1:
                plt.text(y, x, 'X', ha='center', va='center', color='red', fontsize=12)
            else:
                if policy_table[x, y] == 0:
                    plt.arrow(y, x, 0, -0.3, head_width=0.3, head_length=0.3, fc='black', ec='black')
                elif policy_table[x, y] == 1:
                    plt.arrow(y, x, 0, 0.3, head_width=0.3, head_length=0.3, fc='black', ec='black')
                elif policy_table[x, y] == 2:
                    plt.arrow(y, x, -0.3, 0, head_width=0.3, head_length=0.3, fc='black', ec='black')
                elif policy_table[x, y] == 3:
                    plt.arrow(y, x, 0.3, 0, head_width=0.3, head_length=0.3, fc='black', ec='black')
    plt.xlim(-1, env.grid_size)
    plt.ylim(-1, env.grid_size)
    plt.gca().invert_yaxis()
    plt.title(title)
    plt.show()

# Initialize environment and apply both algorithms
env = GridEnvironment()
value_table_dp, policy_table_dp, steps_dp, time_dp = value_iteration(env)
_, policy_table_qlearning, steps_qlearning, time_qlearning = q_learning(env)

# Visualize the policies and print comparisons
print("Value Iteration (DP) Policy:")
visualize_policy(env, policy_table_dp, title="Value Iteration Policy")

print("Q-Learning Policy:")
visualize_policy(env, policy_table_qlearning, title="Q-Learning Policy")

# Output performance comparison
print(f"Dynamic Programming (Value Iteration): Steps Taken = {steps_dp}, Time Taken = {time_dp:.4f} seconds")
print(f"Q-Learning: Steps Taken = {steps_qlearning}, Time Taken = {time_qlearning:.4f} seconds")

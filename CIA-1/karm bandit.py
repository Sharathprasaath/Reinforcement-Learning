import numpy as np
import random

# Number of recommendations/items (arms) and epsilon for exploration
K = 10  # Number of arms (recommendations)
EPSILON = 0.1  # Probability of exploration
NUM_TRIALS = 1000  # Number of trials (or recommendation rounds)

# Initialize estimates and counts for each arm
value_estimates = np.zeros(K)  # Estimated reward for each arm
arm_counts = np.zeros(K)  # Number of times each arm has been pulled

# Generate true reward probabilities for each recommendation (for simulation purposes)
true_rewards = np.random.rand(K)

def recommend_item():
    """Selects an item (arm) to recommend using an epsilon-greedy approach."""
    if random.random() < EPSILON:
        # Explore: randomly select an arm
        return random.randint(0, K - 1)
    else:
        # Exploit: select the arm with the highest estimated reward
        return np.argmax(value_estimates)

def get_user_feedback(arm):
    """Simulates user feedback (reward) based on the true reward probability of the arm."""
    return 1 if random.random() < true_rewards[arm] else 0

# Simulation loop for recommendation trials
for _ in range(NUM_TRIALS):
    # Step 1: Recommend an item
    arm = recommend_item()

    # Step 2: Get user feedback (reward)
    reward = get_user_feedback(arm)

    # Step 3: Update counts and value estimates for the chosen arm
    arm_counts[arm] += 1
    # Incremental update of value estimates to avoid recalculating from scratch
    value_estimates[arm] += (reward - value_estimates[arm]) / arm_counts[arm]

# Display Results
print("True reward probabilities:", true_rewards)
print("Estimated rewards:", value_estimates)
print("Number of times each item was recommended:", arm_counts)

# Determine the best recommended items
best_arm = np.argmax(value_estimates)
print(f"\nBest recommendation based on K-arm bandit: Item {best_arm}")

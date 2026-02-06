import matplotlib.pyplot as plt
import torch
import os
import random
from matplotlib.patches import Ellipse, Circle
import numpy as np
import math
from core.agent import ACLearningAgentWithEmbedding, softmax_with_temperature
from core.continuous_env import ReachTask
from adaptation.config import config
from adaptation.adaptation_plotting import load_test_policies
import distinctipy
from matplotlib.colors import ListedColormap
def signed_circular_distance(numbers, target, period):
    # Calculate the raw difference
    diff = numbers - target
    # Wrap the difference within the range [-period/2, period/2]
    signed_circular_dist = (diff + period / 2) % period - period / 2
    return signed_circular_dist

def compute_circular_mean(indices, num_values):
    """
    Compute the circular mean of indices on a circle with num_values positions.
    """
    angles = np.array(indices) * (2 * np.pi / num_values)  # Convert indices to angles
    sin_sum = np.sum(np.sin(angles))
    cos_sum = np.sum(np.cos(angles))
    mean_angle = np.arctan2(sin_sum, cos_sum)  # Circular mean in radians
    if mean_angle < 0:
        mean_angle += 2 * np.pi  # Ensure the angle is in [0, 2*pi)
    return mean_angle * (num_values / (2 * np.pi))  # Convert back to circular index

def compute_average_indices(agent, embedding_grid, n_repeats=10, num_actions=30):
    averaged_indices = []
    for embedding in embedding_grid:
        indices = []
        for _ in range(n_repeats):
            action_logits = agent.f(torch.tensor(embedding, dtype=torch.float32))
            action_probs = torch.softmax(action_logits, dim=0)
            action_idx = torch.multinomial(action_probs, 1).item()
            indices.append(action_idx)
        circular_mean = compute_circular_mean(indices, num_actions)
        averaged_indices.append(circular_mean)
    return np.array(averaged_indices)

outdir = config['fig_dir']
adapted_model = os.path.join(outdir, 'post_adaptation_model_seed_{}_rotation_{}_temp_{}_weight_decay_0.0001_tanh_policy_mean.pth'.format(config['seed'],
                                                                                    config['rotation_angle'], config['inv_temp']))
# Set plotting defaults
random.seed(config["seed"])
np.random.seed(config["seed"])
torch.manual_seed(config["seed"])

# Environment setup
grid_size = config["grid_size"]
reach_length = config["reach_length"]
centre_coord = (grid_size[0] / 2, grid_size[1] / 2)
target_angles = np.linspace(config["reach_angle"], 2 * np.pi, 1, endpoint=False)
target_xs = centre_coord[0] + reach_length * np.cos(target_angles)
target_ys = centre_coord[1] + reach_length * np.sin(target_angles)
targets = [(xy[0], xy[1]) for xy in zip(target_xs, target_ys)]
max_steps = config["max_steps"]
goal1 = centre_coord if config["random_start_point"] else targets[0]

env = ReachTask(goal1, grid_size_rows=grid_size[0], grid_size_cols=grid_size[1],
                num_actions=config["num_actions"], max_trial_length=max_steps,
                basis_type='Fourier', fourier_order=3, adaptation_rotation=config['rotation_angle'] * math.pi / 180)

# Create agents
agent1 = ACLearningAgentWithEmbedding(env, grid_size[0], grid_size[1], embedding_dim=config["embedding_dim"],
                                      actor_lr=config["actor_lr"], critic_lr=config["critic_lr"],
                                      fg_lr=config["fg_lr"], inv_temp=config["inv_temp"],
                                      policy_std=config["policy_std"],
                                      full_model_load_path=adapted_model,
                                      actor_plastic=False, critic_plastic=False, g_plastic=False)

agent2 = ACLearningAgentWithEmbedding(env, grid_size[0], grid_size[1], embedding_dim=config["embedding_dim"],
                                      actor_lr=config["actor_lr"], critic_lr=config["critic_lr"],
                                      fg_lr=config["fg_lr"], inv_temp=config["inv_temp"],
                                      policy_std=config["policy_std"],
                                      fg_load_path=config['fg_load_path'],
                                      actor_plastic=False, critic_plastic=False, g_plastic=False)

test_angles = [3 * math.pi/4, math.pi/2, math.pi/4, 0, -math.pi/4]
test_policy_means, test_goals = load_test_policies(test_angles, config, outdir, centre_coord, reach_length, grid_size)


# Generate a grid of embeddings
x_range = np.linspace(-1.2, 1.2, 40)
y_range = np.linspace(-1.2, 1.2, 40)
x, y = np.meshgrid(x_range, y_range)
embedding_grid = np.vstack([x.ravel(), y.ravel()]).T

# Function to compute average predicted action indices

# Compute average action indices
avg_indices_agent1 = compute_average_indices(agent1, embedding_grid, n_repeats=100)
avg_indices_agent2 = compute_average_indices(agent2, embedding_grid, n_repeats=100)

# Plotting
colors = distinctipy.get_colors(env.n_actions, pastel_factor=0.7)
cmap = ListedColormap(colors)

fig, axs1 = plt.subplots(1, 1)
scatter1 = axs1.scatter(x, y, c=avg_indices_agent1, cmap=cmap, s=60)
plt.colorbar(scatter1, ax=axs1)
for test_angle_policy in test_policy_means:
    ellipse1 = Circle(test_angle_policy, config['policy_std'], edgecolor='w', alpha=0.3)
    axs1.add_patch(ellipse1)
axs1.set_title("Agent 1 - Averaged Action Indices")

fig, axs2 = plt.subplots(1, 1)
scatter2 = axs2.scatter(x, y, c=avg_indices_agent2, cmap=cmap, s=60)
plt.colorbar(scatter2, ax=axs2)
for test_angle_policy in test_policy_means:
    ellipse2 = Circle(test_angle_policy, config['policy_std'], edgecolor='w', alpha=0.3)
    axs2.add_patch(ellipse2)
axs2.set_title("Agent 2 - Averaged Action Indices")


# Compute the circular difference between averaged indices
circular_diff_indices = avg_indices_agent1 - avg_indices_agent2 #signed_circular_distance(avg_indices_agent2, avg_indices_agent1, period=config["num_actions"])

# Normalize the circular distances for colormap
norm = plt.Normalize(vmin=min(circular_diff_indices), vmax=max(circular_diff_indices))

# Plot the circular differences
fig, axs3 = plt.subplots(1, 1)
scatter_circular = axs3.scatter(x, y, c=circular_diff_indices, cmap='coolwarm', norm=norm, s=60)
plt.colorbar(scatter_circular, ax=axs3, label="Circular Difference (Agent 2 - Agent 1)")
axs3.set_title("Circular Difference in Averaged Action Indices")
axs3.set_xlabel("Embedding X")
axs3.set_ylabel("Embedding Y")

plt.show()
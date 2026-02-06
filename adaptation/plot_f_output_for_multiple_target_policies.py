import matplotlib.pyplot as plt
import pandas as pd
import torch
import random
import os
import numpy as np
import math
from scipy.stats import circmean
import seaborn as sns
from adaptation.adaptation_plotting import load_test_policies
from core.agent import ACLearningAgentWithEmbedding, softmax_with_temperature, \
    add_noise_to_action_probs
from core.continuous_env import ReachTask
from core.plotting import set_plotting_defaults, find_angle_difference
from adaptation.config import config
from definitions import full_model_load_path


def signed_circular_distance(x, y, N):
    """Compute signed circular distance between two points x and y, given circular range N."""
    diff = x - y
    if diff > N / 2:
        return diff - N
    elif diff < -N / 2:
        return diff + N
    else:
        return diff


def get_action_distributions(agent, embedding_grid, next_states, n_repeats=10, n_actions=30):
    distributions = []
    ind_distributions = []
    diff_distributions = []
    for i, embedding_centre in enumerate(embedding_grid):
        indices = []
        actions_taken = []
        angle_diffs = []
        goal = next_states[i]
        for _ in range(n_repeats):
            std_emb = torch.tensor(agent.internal_policy_std)
            embedding = torch.normal(torch.tensor(embedding_centre), std_emb)
            action_logits = agent.f(torch.tensor(embedding, dtype=torch.float32))
            action_probs = softmax_with_temperature(action_logits, temperature=agent.softmax_inv_temp)
            action_probs = add_noise_to_action_probs(action_probs, noise_level=0.005)
            action_idx = torch.multinomial(action_probs, 1).item()
            indices.append(action_idx)
            action_taken = agent.env.actions[action_idx]
            actions_taken.append(action_taken)
            angle_diffs.append(find_angle_difference(agent.env, action_taken, goal=goal))
        distributions.append(np.array(actions_taken))
        ind_distributions.append(np.array(indices))
        diff_distributions.append(np.array(angle_diffs))
    return distributions, ind_distributions, diff_distributions


def epsilon_greedy_selection(action_logits, epsilon=0.1):
    """
    Epsilon-Greedy action selection with one-hot encoded output.
    :param action_logits: A PyTorch tensor or list of action values (logits)
    :param epsilon: Probability of choosing a random action (exploration)
    :return: One-hot encoded action selection
    """
    if not isinstance(action_logits, torch.Tensor):
        action_logits = torch.tensor(action_logits, dtype=torch.float32)

    num_actions = len(action_logits)
    selected_action = torch.randint(num_actions, (1,)).item() if torch.rand(1).item() < epsilon else torch.argmax(
        action_logits).item()

    one_hot = torch.zeros(num_actions, dtype=torch.float32)
    one_hot[selected_action] = 1.0
    return one_hot


def circmedian(angles):
    def circ_dist(a, b):
        return np.abs(np.angle(np.exp(1j * (a - b))))  # Shortest path around the circle

    dist_sums = np.array([np.sum(circ_dist(a, angles)) for a in angles])
    return angles[np.argmin(dist_sums)]


def circular_shift_pre_post(pre_samples, post_samples, range_max=30):
    """
    Compute the circular shift between two sets of samples in a circular variable.

    Args:
        pre_samples (array-like): Pre-samples (values between 0 and range_max).
        post_samples (array-like): Post-samples (values between 0 and range_max).
        range_max (int): Maximum value of the circular variable (default: 30).

    Returns:
        float: Signed circular shift in the range [-range_max/2, range_max/2].
    """

    # Compute circular means
    pre_mean = circmean(pre_samples, high=2 * np.pi, low=0)
    post_mean = circmean(post_samples, high=2 * np.pi, low=0)

    # Compute circular shift in radians
    shift_rad = np.arctan2(np.sin(post_mean - pre_mean), np.cos(post_mean - pre_mean))

    return shift_rad, pre_mean, post_mean


num_seeds = 1
outdir = config['fig_dir']
full_model_load_dir = full_model_load_path
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

# get original agent, env, embeddings and action distributions
base_env = ReachTask(goal1, grid_size_rows=grid_size[0], grid_size_cols=grid_size[1],
                     num_actions=config["num_actions"], max_trial_length=max_steps,
                     basis_type='Fourier', fourier_order=3, adaptation_rotation=0)

base_agent = ACLearningAgentWithEmbedding(base_env, grid_size[0], grid_size[1], embedding_dim=config["embedding_dim"],
                                          actor_lr=config["actor_lr"], critic_lr=config["critic_lr"],
                                          fg_lr=config["fg_lr"], inv_temp=config["inv_temp"],
                                          policy_std=config["policy_std"],
                                          full_model_load_path=config['full_model_load_path'],
                                          actor_plastic=False, critic_plastic=False, g_plastic=False)

# instead of using the output of g as the mean policy, we use the actual pre-trained policy means for different targets
# instead of base_next_states we need the goals from these agents
test_angles = [3 * math.pi/4, math.pi/2, math.pi/4, 0, -math.pi/4]
test_policy_means, test_goals = load_test_policies(test_angles, config, full_model_load_dir, centre_coord,
                                                   reach_length, grid_size)

dfs = []
set_plotting_defaults()
for seed in range(0, num_seeds):
    adapted_model = os.path.join(outdir,
                                 'post_adaptation_model_seed_{}_rotation_{}_temp_{}_weight_decay_0.0001_tanh_policy_mean.pth'.format(
                                     seed,
                                     config['rotation_angle'], config['inv_temp']))
    # Set plotting defaults
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    rotated_env = ReachTask(goal1, grid_size_rows=grid_size[0], grid_size_cols=grid_size[1],
                            num_actions=config["num_actions"], max_trial_length=max_steps,
                            basis_type='Fourier', fourier_order=3,
                            adaptation_rotation=config['rotation_angle'] * math.pi / 180)

    # Create agent
    adapted_agent = ACLearningAgentWithEmbedding(rotated_env, grid_size[0], grid_size[1],
                                                 embedding_dim=config["embedding_dim"],
                                                 actor_lr=config["actor_lr"], critic_lr=config["critic_lr"],
                                                 fg_lr=config["fg_lr"], inv_temp=config["inv_temp"],
                                                 policy_std=config["policy_std"],
                                                 full_model_load_path=adapted_model,
                                                 actor_plastic=False, critic_plastic=False, g_plastic=False)

    # Compute average action indices
    adapted_agent_distributions, adapted_agent_ind_distributions, adapted_agent_diffs = get_action_distributions(
        base_agent, test_policy_means, test_goals, n_repeats=1000)
    n_actions = base_agent.env.n_actions
    shifts = []
    pre_means = []
    post_means = []
    adaptation_amount = []
    angular_error = []
    if seed == num_seeds - 1:
        fig_example, axs_example = plt.subplots(len(test_angles), sharex=True, sharey=True)
        axs_ind = axs_example.ravel()
        fig_example_ix, axs_example_ix = plt.subplots(len(test_angles), sharex=True, sharey=True)
        axs_ix_ind = axs_example_ix.ravel()
    cmap = plt.get_cmap('twilight')
    for test_angle in range(len(test_angles)):
        angular_error.append(circmean(adapted_agent_diffs[test_angle], high=180, low=-180))
        adaptation_amount.append(100 * (1 - (angular_error[test_angle] * -1 / config['rotation_angle'])))
        if seed == num_seeds - 1:
            index = test_angle
            # Define bins
            bin_edges = np.arange(0, 360 + 12, 12)  # These are the valid bin values
            bin_centers = bin_edges  # Since values directly match bin edges

            # Count occurrences of each bin value in base and adapted distributions
            adapted_counts = np.bincount(np.searchsorted(bin_edges, np.degrees(adapted_agent_distributions[test_angle])),
                                         minlength=len(bin_edges))

            # Plot the bar chart instead of histogram
            axs_ind[test_angle].bar(bin_centers, adapted_counts, width=12, color='green', alpha=0.5, label="Adapted Agent")

            axs_ind[test_angle].set_xlim([0, 360])
            axs_ind[test_angle].axvline(np.degrees(base_agent.env.actions[1]), linewidth=0.5, color='k')

            axs_ix_ind[test_angle].hist(adaptation_amount[test_angle],
                                    color=cmap(index), alpha=0.8)
    rotation_generalization = 100 * (adaptation_amount / adaptation_amount[4])

    df = pd.DataFrame({'test angle': np.degrees(test_angles), 'seed': np.ones(len(test_angles)) * seed,
                       'adaptation amount': adaptation_amount, 'rotation generalization': rotation_generalization,
                       'angular error': angular_error})
    dfs.append(df)
df_to_plot = pd.concat(dfs)

fig, axs = plt.subplots(1, 1, figsize=(1.5, 1.6))
sns.lineplot(data=df_to_plot, x='test angle', y='rotation generalization', errorbar='sd', ax=axs)
axs.set_ylabel('rotation generalization')
plt.tight_layout()

fig, axs = plt.subplots(1, 1, figsize=(1.5, 1.6))
sns.lineplot(data=df_to_plot, x='test angle', y='angular error', errorbar='sd', ax=axs)
axs.set_ylabel('angular error')
plt.tight_layout()
plt.show()

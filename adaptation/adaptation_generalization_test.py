import matplotlib.pyplot as plt
import pandas as pd
import torch
import random
import os
import numpy as np
import seaborn as sns

from core.agent import softmax_with_temperature, add_noise_to_action_probs
from core.plotting import set_plotting_defaults, find_angle_difference
from adaptation.config import config
from core.utils import calc_circular_shift


def get_action_distributions(agent, embedding_grid, next_states, n_repeats=10):
    # todo: looks to me like this can be a class method of the agent?
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
            action_logits = agent.f(embedding)
            action_probs = softmax_with_temperature(action_logits, temperature=agent.softmax_inv_temp)
            action_probs = add_noise_to_action_probs(action_probs, noise_level=0.008)
            action_idx = torch.multinomial(action_probs, 1).item()
            indices.append(action_idx)
            action_taken = agent.env.actions[action_idx]
            actions_taken.append(action_taken)
            angle_diffs.append(find_angle_difference(agent.env, action_taken, goal=goal))
        distributions.append(np.array(actions_taken))
        ind_distributions.append(np.array(indices))
        diff_distributions.append(np.array(angle_diffs))
    return distributions, ind_distributions, diff_distributions


def calculate_generalization(adapted_agent, base_agent, cfg):
    seed = cfg['seed']

    # get original agent, env, embeddings and action distributions
    embeddings, base_next_states = base_agent.get_action_embeddings_via_g(return_next_state=True)

    actions_in_degrees = np.round(np.degrees(base_agent.env.actions))
    set_plotting_defaults()
    target = np.round(np.degrees(cfg['reach_angle'])).astype(int)

    # Set plotting defaults
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Compute average action indices
    _, adapted_next_states = adapted_agent.get_action_embeddings_via_g(return_next_state=True)
    adapted_agent_distributions, adapted_agent_ind_distributions, adapted_agent_diffs = get_action_distributions(
        adapted_agent, embeddings, base_next_states, n_repeats=1000)
    n_actions = base_agent.env.n_actions
    adaptation_amount = []
    angular_error = []

    for action in range(n_actions):
        angular_error.append(np.mean(adapted_agent_diffs[action]))
        adaptation_amount.append(100 * (1 - (angular_error[action] * -1 / cfg['rotation_angle'])))
    trained_angle_ind = np.where(actions_in_degrees == target)[0][0]
    rotation_generalization = np.array(adaptation_amount) / adaptation_amount[trained_angle_ind] * 100

    df = pd.DataFrame({'action inds': np.arange(0, n_actions), 'seed': np.ones(n_actions) * seed,
                       'adaptation amount': adaptation_amount, 'rotation generalization': rotation_generalization,
                       'angular error': angular_error, 'action angle': actions_in_degrees})
    df['angle from target'] = [round(np.degrees(calc_circular_shift(np.radians(target), np.radians(angle))).item()) for
                               angle in df['action angle']]
    return df


def make_generalization_plot(df, only_paper_angs=False):
    angles = [-135.0, -90.0, -45.0, 0.0, 45.0, 90.0, 135.0, 180.0]
    if only_paper_angs:
        df = df[df['angle from target'].isin(angles)]

    rot_generalization_fig, axs = plt.subplots(1, 1, figsize=(1.5, 1.6))
    sns.lineplot(data=df, x='angle from target', y='rotation generalization', errorbar='sd', ax=axs)
    axs.set_ylabel('rotation generalization')
    plt.tight_layout()
    if config['save_figs_locally']:
        plt.savefig(
            os.path.join(config['fig_dir'], 'adaptation_{}_degrees.pdf'.format(config['rotation_angle'])))

    angular_error_fig, axs = plt.subplots(1, 1, figsize=(1.5, 1.6))
    sns.lineplot(data=df, x='angle from target', y='angular error', errorbar='sd', ax=axs)
    axs.set_ylabel('angular error')
    plt.tight_layout()
    if config['save_figs_locally']:
        plt.savefig(os.path.join(config['fig_dir'],
                                'angular_error_post_adaptation_test_targets_{}_degrees.pdf'.format(config['rotation_angle'])))
    return rot_generalization_fig, angular_error_fig
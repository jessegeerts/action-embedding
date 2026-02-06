"""
Analyze baseline action samples using different policy mean embeddings.

This script systematically tests how the agent's action distribution changes
when different action embeddings are imported as the policy mean, without
any training. This helps understand the embedding space and how policy means
influence action selection.
"""
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import pandas as pd
import argparse
from typing import Dict, List, Tuple

from definitions import data_root
from core.model_loading_utils import load_trained_full_model_basetask
from core.continuous_env import ReachTask


def collect_action_samples_with_policy_mean(agent, env, cfg, policy_mean_embedding, n_samples=5000):
    """
    Collect action samples using a specific policy mean embedding.

    Args:
        agent: The trained agent
        env: Environment to sample from
        cfg: Configuration dict
        policy_mean_embedding: Tensor embedding to use as policy mean
        n_samples: Number of action samples to collect

    Returns:
        List of action indices sampled
    """
    sampled_actions = []
    agent.env = env

    while len(sampled_actions) < n_samples:
        agent.env.reset()
        features = agent.env.get_features(agent.env.current_xy)
        done = False
        steps = 0
        max_steps = cfg["max_steps"]

        while steps < max_steps and len(sampled_actions) < n_samples:
            # Select action with imported policy mean, NO gradient
            with torch.no_grad():
                action_ind, embedding, mean_emb, logstd_emb = agent.select_action(
                    features,
                    random_policy=False,
                    import_policy_mean=True,
                    policy_mean=policy_mean_embedding
                )
            action = agent.env.actions[action_ind]

            # Take action but don't learn
            next_state, reward, done = agent.env.act(action)
            next_state_features = agent.env.get_features(next_state)

            # Track action
            sampled_actions.append(action_ind)
            steps += 1
            features = next_state_features

            if done:
                break

    return sampled_actions[:n_samples]


def analyze_all_embeddings(agent, env, cfg, n_samples=5000, rotation_angle=0.0):
    """
    Collect baseline samples for each action's embedding used as policy mean.

    Args:
        agent: Trained agent with embeddings
        env: Environment (should have NO rotation for true baseline)
        cfg: Configuration dict
        n_samples: Number of samples per embedding
        rotation_angle: Optional rotation to apply to environment (default 0 for baseline)

    Returns:
        Dictionary mapping action_angle -> sampled_actions
    """
    # Get all action embeddings
    embeddings, _ = agent.get_action_embeddings_via_g(return_next_state=True)
    action_angles = env.actions  # radians

    print(f"Analyzing {len(action_angles)} different policy mean embeddings...")
    print(f"Collecting {n_samples} samples per embedding...")

    results = {}

    for action_idx, action_angle in enumerate(action_angles):
        print(f"  Sampling with embedding for action {action_idx} (angle: {np.degrees(action_angle):.1f}°)")

        # Get the embedding for this action
        policy_mean_embedding = torch.tensor(embeddings[action_idx])

        # Collect samples
        samples = collect_action_samples_with_policy_mean(
            agent, env, cfg, policy_mean_embedding, n_samples
        )

        results[action_angle] = samples

    print("Sampling complete!")
    return results


def plot_embedding_analysis(results: Dict[float, List[int]],
                            action_angles: np.ndarray,
                            target_angle: float,
                            save_path: str = None):
    """
    Create visualization showing action distributions for each policy mean embedding.

    Args:
        results: Dictionary mapping source_angle -> list of sampled actions
        action_angles: Array of all possible action angles (in radians)
        target_angle: The target angle for this environment (in radians)
        save_path: Optional path to save the figure
    """
    n_embeddings = len(results)
    palette = sns.color_palette('husl', n_embeddings)

    # Create polar plot
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))

    bar_width = 2 * np.pi / len(action_angles)

    # Plot each embedding's action distribution
    for i, (source_angle, actions) in enumerate(sorted(results.items())):
        if len(actions) == 0:
            continue

        # Calculate action distribution
        unique_actions, counts = np.unique(actions, return_counts=True)
        proportions = counts / len(actions)
        taken_angles = action_angles[unique_actions]

        # Offset bars slightly for visibility when overlapping
        offset = (i - n_embeddings / 2) * (bar_width / n_embeddings)

        ax.bar(taken_angles + offset, proportions,
               width=bar_width / n_embeddings,
               alpha=0.6,
               edgecolor='black',
               linewidth=0.3,
               color=palette[i],
               label=f'{np.degrees(source_angle):.0f}° embedding')

    # Mark target angle
    ax.axvline(target_angle, color='red', linestyle='--', linewidth=2,
               label='Target angle', zorder=100)

    ax.set_theta_zero_location('E')
    ax.set_theta_direction(1)
    ax.set_title(
        f'Action Distributions Using Different Policy Mean Embeddings\n(Target: {np.degrees(target_angle):.0f}°)',
        pad=20, fontsize=14)
    ax.set_ylabel('Proportion', labelpad=30)

    # Legend outside plot
    ax.legend(loc='upper left', bbox_to_anchor=(1.1, 1.0), fontsize=8)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    plt.show()
    return fig


def plot_heatmap_analysis(results: Dict[float, List[int]],
                          action_angles: np.ndarray,
                          save_path: str = None):
    """
    Create heatmap showing: for each source embedding, which actions are selected.
    Rows = source embedding angle, Columns = selected action angle

    Args:
        results: Dictionary mapping source_angle -> list of sampled actions
        action_angles: Array of all possible action angles (in radians)
        save_path: Optional path to save the figure
    """
    n_actions = len(action_angles)
    source_angles_sorted = sorted(results.keys())

    # Build matrix: rows = source embeddings, cols = action taken
    matrix = np.zeros((len(source_angles_sorted), n_actions))

    for i, source_angle in enumerate(source_angles_sorted):
        actions = results[source_angle]
        unique_actions, counts = np.unique(actions, return_counts=True)
        proportions = counts / len(actions)
        matrix[i, unique_actions] = proportions

    # Create figure
    fig, ax = plt.subplots(figsize=(12, 10))

    # Plot heatmap
    im = ax.imshow(matrix, aspect='auto', cmap='viridis', interpolation='nearest')

    # Labels
    action_labels = [f"{np.degrees(a):.0f}°" for a in action_angles]
    source_labels = [f"{np.degrees(a):.0f}°" for a in source_angles_sorted]

    ax.set_xticks(np.arange(n_actions))
    ax.set_yticks(np.arange(len(source_angles_sorted)))
    ax.set_xticklabels(action_labels, rotation=90, fontsize=8)
    ax.set_yticklabels(source_labels, fontsize=8)

    ax.set_xlabel('Action Taken (degrees)', fontsize=12)
    ax.set_ylabel('Source Embedding (degrees)', fontsize=12)
    ax.set_title('Action Selection Matrix: Source Embedding → Action Taken', fontsize=14, pad=20)

    # Colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Proportion of Actions', rotation=270, labelpad=20)

    # Grid
    ax.set_xticks(np.arange(n_actions) - 0.5, minor=True)
    ax.set_yticks(np.arange(len(source_angles_sorted)) - 0.5, minor=True)
    ax.grid(which='minor', color='gray', linestyle='-', linewidth=0.5, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    plt.show()
    return fig


def compute_embedding_statistics(results: Dict[float, List[int]],
                                 action_angles: np.ndarray) -> pd.DataFrame:
    """
    Compute statistics for each embedding's induced action distribution.

    Returns DataFrame with columns:
    - source_angle_deg: angle of the embedding used as policy mean
    - most_common_action_deg: most frequently selected action
    - most_common_proportion: proportion of most common action
    - entropy: Shannon entropy of action distribution
    - mean_action_deg: circular mean of selected actions
    """
    stats = []

    for source_angle, actions in sorted(results.items()):
        unique_actions, counts = np.unique(actions, return_counts=True)
        proportions = counts / len(actions)

        # Most common action
        most_common_idx = unique_actions[np.argmax(proportions)]
        most_common_angle = action_angles[most_common_idx]
        most_common_prop = np.max(proportions)

        # Entropy (higher = more uniform)
        entropy = -np.sum(proportions * np.log2(proportions + 1e-10))

        # Circular mean of actions (weighted by frequency)
        action_angles_taken = action_angles[actions]
        x_mean = np.mean(np.cos(action_angles_taken))
        y_mean = np.mean(np.sin(action_angles_taken))
        mean_angle = np.arctan2(y_mean, x_mean)

        stats.append({
            'source_angle_deg': np.degrees(source_angle),
            'most_common_action_deg': np.degrees(most_common_angle),
            'most_common_proportion': most_common_prop,
            'entropy': entropy,
            'mean_action_deg': np.degrees(mean_angle) % 360
        })

    return pd.DataFrame(stats)


if __name__ == '__main__':
    from adaptation.config import config

    parser = argparse.ArgumentParser(description='Analyze policy mean embeddings')
    parser.add_argument('--seed', type=int, default=9,
                        help='Random seed for loading model')
    parser.add_argument('--target_angle', type=int, default=135,
                        help='Target angle in degrees for the environment')
    parser.add_argument('--n_samples', type=int, default=5000,
                        help='Number of action samples per embedding')
    parser.add_argument('--rotation', type=float, default=0.0,
                        help='Rotation angle to apply (0 for baseline)')
    args = parser.parse_args()

    # Setup
    data_save_dir = os.path.join(data_root, 'policy_mean_embedding_analysis')
    if not os.path.exists(data_save_dir):
        os.makedirs(data_save_dir)

    config['seed'] = args.seed
    np.random.seed(config['seed'])
    torch.manual_seed(config['seed'])

    # Configuration
    target_angle_rad = np.radians(args.target_angle)
    cfg = dict(config)
    cfg['reach_angle'] = float(target_angle_rad)
    cfg['rotation_angle'] = args.rotation
    cfg['load_different_fg'] = False
    cfg['save_model'] = False

    # Load model
    load_seed = args.seed
    cfg['load_full_model_file'] = os.path.join(
        config['full_model_load_path'],
        f'fully_trained_policy_model_one_target_seed_{load_seed}_weight_decay_0.0001_tanh_policy_mean_target_{args.target_angle}_n_actions_24.pth'
    )

    # Create environment (no rotation for baseline)
    env = ReachTask(cfg, adaptation_rotation=np.radians(args.rotation))

    # Load agent
    print(f"Loading agent trained on {args.target_angle}° target...")
    agent = load_trained_full_model_basetask(cfg, env, args.target_angle, load_seed)

    # Analyze all embeddings
    print(f"\nAnalyzing all policy mean embeddings...")
    results = analyze_all_embeddings(agent, env, cfg, n_samples=args.n_samples,
                                     rotation_angle=args.rotation)

    # Save raw results
    print("\nSaving raw results...")
    results_df = pd.DataFrame({
        f'{np.degrees(angle):.0f}deg_embedding': pd.Series(actions)
        for angle, actions in sorted(results.items())
    })
    results_path = os.path.join(data_save_dir,
                                f'embedding_samples_target{args.target_angle}_seed{args.seed}.csv')
    results_df.to_csv(results_path, index=False)
    print(f"Saved to: {results_path}")

    # Compute statistics
    print("\nComputing statistics...")
    stats_df = compute_embedding_statistics(results, env.actions)
    stats_path = os.path.join(data_save_dir,
                              f'embedding_stats_target{args.target_angle}_seed{args.seed}.csv')
    stats_df.to_csv(stats_path, index=False)
    print(f"Saved statistics to: {stats_path}")
    print("\nStatistics summary:")
    print(stats_df.to_string(index=False))

    # Visualizations
    print("\nGenerating visualizations...")

    # Polar plot
    polar_path = os.path.join(data_save_dir,
                              f'embedding_polar_target{args.target_angle}_seed{args.seed}.png')
    plot_embedding_analysis(results, env.actions, target_angle_rad, save_path=polar_path)

    # Heatmap
    heatmap_path = os.path.join(data_save_dir,
                                f'embedding_heatmap_target{args.target_angle}_seed{args.seed}.png')
    plot_heatmap_analysis(results, env.actions, save_path=heatmap_path)

    print("\nAnalysis complete!")
    print(f"Results saved to: {data_save_dir}")
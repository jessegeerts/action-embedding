"""
In this file, we run an adaptation experiment with many combinations of two different targets (with opposite rotations)
Now with INTERLEAVED training to reduce catastrophic interference between phases.

Creates:
- 1 W&B run combining both phases with interleaved trials

FIXED: Embeddings are now extracted with neutral (non-rotated) environment to ensure correct policy mean transfer
ADDED: CSV logging for losses and angular errors similar to adaptation_exp.py
"""
import copy
import math
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import wandb
import pandas as pd
import argparse

from definitions import data_root, paper_fig_dir
from core.model_loading_utils import load_trained_full_model_basetask
from core.plotting import find_angle_difference


def collect_action_samples(env, cfg, n_samples):
    """Collect actions without training to establish baseline/post-training behavior"""
    sampled_actions = []
    agent.env = env

    while len(sampled_actions) < n_samples:
        agent.env.reset()
        features = agent.env.get_features(agent.env.current_xy)
        done = False
        steps = 0
        max_steps = cfg["max_steps"]

        while steps < max_steps and len(sampled_actions) < n_samples:
            # Select action WITHOUT updating weights
            with torch.no_grad():  # No gradients
                action_ind, embedding, mean_emb, logstd_emb = agent.select_action(
                    features,
                    random_policy=False,
                    import_policy_mean=cfg.get('import_policy_mean', False),
                    policy_mean=cfg.get('policy_mean_to_import', None)
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

    return sampled_actions[:n_samples]  # Return exactly n_samples


def train_interleaved_adaptation(cfg_phase1, cfg_phase2, agent,
                                 env_phase1, env_phase2,
                                 env_phase1_baseline, env_phase2_baseline,
                                 total_episodes, phase_prob=0.5, n_baseline_samples=5000):
    """
    Train with randomly interleaved episodes from phase 1 and phase 2.
    CRITICAL: Uses a SINGLE agent that switches between two environments,
    so all weight updates affect the same network (testing interference).

    Args:
        cfg_phase1: configuration dict for phase 1
        cfg_phase2: configuration dict for phase 2
        agent: single agent that will experience both phases
        env_phase1: environment for phase 1 WITH rotation (for training)
        env_phase2: environment for phase 2 WITH rotation (for training)
        env_phase1_baseline: environment for phase 1 WITHOUT rotation (for baseline)
        env_phase2_baseline: environment for phase 2 WITHOUT rotation (for baseline)
        total_episodes: total number of episodes to run
        phase_prob: probability of sampling phase 1 on each episode (default 0.5 for equal mixing)
        n_baseline_samples: number of actions to sample before/after training (default 5000)

    Returns:
        dict with training results split by phase
    """

    # -----------------------------------------
    # PRE-TRAINING BASELINE: Sample behavior before any adaptation
    # Use UNROTATED environments to get true baseline behavior
    # -----------------------------------------
    print(f"Collecting {n_baseline_samples} baseline samples for each phase (NO rotation)...")

    phase1_baseline_actions = collect_action_samples(env_phase1_baseline, cfg_phase1, n_baseline_samples)
    phase2_baseline_actions = collect_action_samples(env_phase2_baseline, cfg_phase2, n_baseline_samples)

    print(f"Baseline collection complete. Starting adaptation training...")

    # -----------------------------------------
    # TRAINING: Now do the actual interleaved adaptation
    # -----------------------------------------
    # Initialize tracking for both phases
    phase1_action_history = []
    phase2_action_history = []
    phase1_reward_history = []
    phase2_reward_history = []
    phase1_critic_loss_history = []
    phase2_critic_loss_history = []
    phase1_actor_loss_history = []
    phase2_actor_loss_history = []
    phase1_nll_loss_history = []
    phase2_nll_loss_history = []
    phase1_angle_err_history = []
    phase2_angle_err_history = []
    phase_sequence = []  # track which phase was trained on each episode

    phase1_episode_count = 0
    phase2_episode_count = 0

    max_steps = cfg_phase1["max_steps"]  # assuming both configs have same max_steps

    # Training loop with random interleaving at the episode level
    for episode in range(total_episodes):
        # Randomly choose which phase to train on for this episode
        train_phase1 = np.random.rand() < phase_prob
        phase_sequence.append(1 if train_phase1 else 2)

        # Select environment and config for this episode
        if train_phase1:
            env = env_phase1
            cfg = cfg_phase1
            phase1_episode_count += 1
        else:
            env = env_phase2
            cfg = cfg_phase2
            phase2_episode_count += 1

        # CRITICAL: Point the single agent to the current environment
        agent.env = env

        # Run one episode (following the original structure)
        agent.env.reset()
        features = agent.env.get_features(agent.env.current_xy)
        done = False
        steps = 0
        episode_actions = []
        episode_reward = 0
        total_nll_loss = 0
        total_actor_loss = 0
        total_critic_loss = 0

        random_policy = cfg.get('use_random_policy', False)

        while steps < max_steps:
            # Zero the gradients for all modules
            agent.actor_optimizer.zero_grad()
            agent.critic_optimizer.zero_grad()
            agent.f_g_optimizer.zero_grad()

            # Select action
            action_ind, embedding, mean_emb, logstd_emb = agent.select_action(
                features,
                random_policy=random_policy,
                import_policy_mean=cfg.get('import_policy_mean', False),
                policy_mean=cfg.get('policy_mean_to_import', None)
            )
            action = agent.env.actions[action_ind]

            # Calculate angular error
            angular_error = find_angle_difference(agent.env, action)

            # Take action in environment
            next_state, reward, done = agent.env.act(action)
            next_state_features = agent.env.get_features(next_state)

            # Update agent (THIS UPDATES THE SHARED WEIGHTS)
            nll_loss, actor_loss, critic_loss = agent.update(
                features, action_ind, embedding, next_state_features, reward, done
            )

            # Accumulate losses for this episode
            total_actor_loss += actor_loss
            total_critic_loss += critic_loss
            total_nll_loss += nll_loss

            # Track episode data
            episode_actions.append(action_ind)
            episode_reward += reward
            steps += 1

            # Update features for next step
            features = next_state_features

            if done:
                break

        # Calculate average losses for the episode
        avg_critic_loss_ep = total_critic_loss / steps if steps > 0 else 0
        avg_actor_loss_ep = total_actor_loss / steps if steps > 0 else 0
        avg_emb_loss_ep = total_nll_loss / steps if steps > 0 else 0

        # Store episode results in appropriate phase history
        if train_phase1:
            phase1_action_history.append(episode_actions)
            phase1_reward_history.append(episode_reward)
            phase1_critic_loss_history.append(avg_critic_loss_ep)
            phase1_actor_loss_history.append(avg_actor_loss_ep)
            phase1_nll_loss_history.append(avg_emb_loss_ep)
            phase1_angle_err_history.append(angular_error)
        else:
            phase2_action_history.append(episode_actions)
            phase2_reward_history.append(episode_reward)
            phase2_critic_loss_history.append(avg_critic_loss_ep)
            phase2_actor_loss_history.append(avg_actor_loss_ep)
            phase2_nll_loss_history.append(avg_emb_loss_ep)
            phase2_angle_err_history.append(angular_error)

        # Optional: Log progress periodically
        if cfg_phase1.get('log_to_wandb', False) and episode % 100 == 0:
            wandb.log({
                'episode': episode,
                'phase1_episodes': phase1_episode_count,
                'phase2_episodes': phase2_episode_count,
                'phase1_mean_reward': np.mean(phase1_reward_history[-100:]) if len(phase1_reward_history) >= 100 else (
                    np.mean(phase1_reward_history) if phase1_reward_history else 0),
                'phase2_mean_reward': np.mean(phase2_reward_history[-100:]) if len(phase2_reward_history) >= 100 else (
                    np.mean(phase2_reward_history) if phase2_reward_history else 0),
                'phase1_mean_critic_loss': np.mean(phase1_critic_loss_history[-100:]) if len(
                    phase1_critic_loss_history) >= 100 else (
                    np.mean(phase1_critic_loss_history) if phase1_critic_loss_history else 0),
                'phase2_mean_critic_loss': np.mean(phase2_critic_loss_history[-100:]) if len(
                    phase2_critic_loss_history) >= 100 else (
                    np.mean(phase2_critic_loss_history) if phase2_critic_loss_history else 0),
                'phase1_mean_actor_loss': np.mean(phase1_actor_loss_history[-100:]) if len(
                    phase1_actor_loss_history) >= 100 else (
                    np.mean(phase1_actor_loss_history) if phase1_actor_loss_history else 0),
                'phase2_mean_actor_loss': np.mean(phase2_actor_loss_history[-100:]) if len(
                    phase2_actor_loss_history) >= 100 else (
                    np.mean(phase2_actor_loss_history) if phase2_actor_loss_history else 0),
                'phase1_mean_nll_loss': np.mean(phase1_nll_loss_history[-100:]) if len(
                    phase1_nll_loss_history) >= 100 else (
                    np.mean(phase1_nll_loss_history) if phase1_nll_loss_history else 0),
                'phase2_mean_nll_loss': np.mean(phase2_nll_loss_history[-100:]) if len(
                    phase2_nll_loss_history) >= 100 else (
                    np.mean(phase2_nll_loss_history) if phase2_nll_loss_history else 0),
                'phase1_mean_angle_err': np.mean(phase1_angle_err_history[-100:]) if len(
                    phase1_angle_err_history) >= 100 else (
                    np.mean(phase1_angle_err_history) if phase1_angle_err_history else 0),
                'phase2_mean_angle_err': np.mean(phase2_angle_err_history[-100:]) if len(
                    phase2_angle_err_history) >= 100 else (
                    np.mean(phase2_angle_err_history) if phase2_angle_err_history else 0),
            })

    # -----------------------------------------
    # POST-TRAINING: Sample behavior after all adaptation
    # Use ROTATED environments to see adapted behavior
    # -----------------------------------------
    print(f"Training complete. Collecting {n_baseline_samples} post-training samples for each phase (WITH rotation)...")

    phase1_post_actions = collect_action_samples(env_phase1, cfg_phase1, n_baseline_samples)
    phase2_post_actions = collect_action_samples(env_phase2, cfg_phase2, n_baseline_samples)

    print(f"Post-training collection complete.")

    results = {
        'phase1_baseline_actions': phase1_baseline_actions,
        'phase2_baseline_actions': phase2_baseline_actions,
        'phase1_post_actions': phase1_post_actions,
        'phase2_post_actions': phase2_post_actions,
        'phase1_action_history': phase1_action_history,
        'phase2_action_history': phase2_action_history,
        'phase1_reward_history': phase1_reward_history,
        'phase2_reward_history': phase2_reward_history,
        'phase1_critic_loss_history': phase1_critic_loss_history,
        'phase2_critic_loss_history': phase2_critic_loss_history,
        'phase1_actor_loss_history': phase1_actor_loss_history,
        'phase2_actor_loss_history': phase2_actor_loss_history,
        'phase1_nll_loss_history': phase1_nll_loss_history,
        'phase2_nll_loss_history': phase2_nll_loss_history,
        'phase1_angle_err_history': phase1_angle_err_history,
        'phase2_angle_err_history': phase2_angle_err_history,
        'phase_sequence': phase_sequence,
        'agent': agent,  # Return the single agent that experienced both phases
        'phase1_episode_count': phase1_episode_count,
        'phase2_episode_count': phase2_episode_count
    }

    return results


if __name__ == '__main__':
    from adaptation.config import config
    from adaptation.adaptation_exp import train_adapatation_experiment
    from core.continuous_env import ReachTask

    data_save_dir = os.path.join(data_root, 'doubleAdaptationExp_interleaved')
    if not os.path.exists(data_save_dir):
        os.makedirs(data_save_dir)

    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=9, help='Random seed for the experiment')
    parser.add_argument('--total_episodes', type=int, default=None,
                        help='Total training episodes (default: 2x max_episodes from config)')
    parser.add_argument('--phase_prob', type=float, default=0.5,
                        help='Probability of sampling phase 1 (default 0.5 for equal mixing)')
    parser.add_argument('--n_baseline', type=int, default=5000,
                        help='Number of baseline samples to collect before training (default 5000)')
    parser.add_argument('--phase2_rel_angle', type=float, default=180.,
                        help='Angle relative to initial training angle for phase 2 (default 180 degrees)')
    args = parser.parse_args()

    config['seed'] = args.seed
    np.random.seed(config['seed'])
    torch.manual_seed(config['seed'])

    palette = sns.color_palette('Paired')

    exp_save_dir = os.path.join(data_save_dir, f'phase2relangle_{int(args.phase2_rel_angle)}deg')
    if not os.path.exists(exp_save_dir):
        os.makedirs(exp_save_dir)
    # -----------------------------------------
    # Setup Phase 1 configuration
    # -----------------------------------------
    phase_1_reach_angle = np.radians(135)  # radians (135 degrees)
    phase_1_rotation = -30  # degrees

    config_phase1 = dict(config)
    config_phase1['reach_angle'] = float(phase_1_reach_angle)
    config_phase1['rotation_angle'] = phase_1_rotation
    config_phase1['load_different_fg'] = False
    config_phase1['save_model'] = False  # We'll save at the end
    config_phase1['import_policy_mean'] = False  # Use naturally learned mean for this angle
    config_phase1['policy_mean_to_import'] = None

    load_seed = config['seed']
    config_phase1['load_full_model_file'] = os.path.join(
        config['full_model_load_path'],
        'fully_trained_policy_model_one_target_seed_{}_weight_decay_0.0001_tanh_policy_mean_target_{}_n_actions_24.pth'
        .format(load_seed, int(np.degrees(config_phase1['reach_angle'])))
    )

    # FIXED: Create baseline environment FIRST (no rotation) and load agent with it
    env_phase1_baseline = ReachTask(config_phase1, adaptation_rotation=0.0)

    # Load agent with neutral environment
    print(f"Loading agent with neutral environment (no rotation)...")
    agent = load_trained_full_model_basetask(
        config_phase1, env_phase1_baseline,
        np.round(np.degrees(config_phase1['reach_angle'])).astype(int),
        load_seed
    )

    # FIXED: Get embeddings BEFORE creating rotated environments
    # This ensures embeddings are extracted in a neutral state
    print(f"Extracting action embeddings from neutral state...")
    embeddings, _ = agent.get_action_embeddings_via_g(return_next_state=True)
    action_idx_map = {a: idx for a, idx in zip(agent.env.actions, range(agent.env.n_actions))}


    def get_action_idx_for_angle(target_angle_rad, actions):
        """Find the action index closest to the target angle, handling floating point precision."""
        # Normalize angles to [0, 2*pi)
        target_normalized = target_angle_rad % (2 * np.pi)
        actions_normalized = np.array(actions) % (2 * np.pi)

        # Find closest match
        angle_diffs = np.abs(actions_normalized - target_normalized)
        # Handle wraparound at 0/2*pi
        angle_diffs = np.minimum(angle_diffs, 2 * np.pi - angle_diffs)

        closest_idx = np.argmin(angle_diffs)

        # Sanity check: should be within ~1 degree
        if angle_diffs[closest_idx] > np.radians(1):
            raise ValueError(f"No action found within 1° of target {np.degrees(target_angle_rad):.1f}°")

        return closest_idx


    # NOW create phase 1 environment with rotation
    env_phase1 = ReachTask(config_phase1, adaptation_rotation=np.radians(config_phase1['rotation_angle']))

    # -----------------------------------------
    # Setup Phase 2 configuration and environment
    # -----------------------------------------
    phase_2_reach_angles = [np.radians(135 + args.phase2_rel_angle)]  # 315 degrees only
    phase_2_rotation = 30  # degrees

    for i, second_reach_angle in enumerate(phase_2_reach_angles):

        # Get the embedding for the second angle
        pre_rotation_a_idx = get_action_idx_for_angle(second_reach_angle, agent.env.actions)

        print(f"\nPhase 2 setup:")
        print(f"  Target angle: {np.degrees(second_reach_angle):.1f}°")
        print(f"  Embedding index: {pre_rotation_a_idx}")
        print(f"  Embedding value: {embeddings[pre_rotation_a_idx]}")

        config_phase2 = dict(config)
        config_phase2['reach_angle'] = float(second_reach_angle)
        config_phase2['rotation_angle'] = phase_2_rotation
        config_phase2['load_full_model_file'] = None
        config_phase2['save_model'] = False
        # Store the policy mean for this angle (will be used during action selection)
        config_phase2['import_policy_mean'] = True
        config_phase2['policy_mean_to_import'] = torch.tensor(embeddings[pre_rotation_a_idx])

        # Create phase 2 environment WITH rotation (for training)
        env_phase2 = ReachTask(config_phase2, adaptation_rotation=np.radians(config_phase2['rotation_angle']))

        # Create phase 2 environment WITHOUT rotation (for baseline)
        env_phase2_baseline = ReachTask(config_phase2, adaptation_rotation=0.0)

        # -----------------------------------------
        # W&B logging
        # -----------------------------------------
        second_deg = np.degrees(second_reach_angle)
        run = None
        if config.get('log_to_wandb', False):
            run = wandb.init(
                project='ActionEmbeddingRotationMultiTarget_Interleaved',
                name=f"interleaved_angles_{int(np.degrees(phase_1_reach_angle))}deg_vs_{second_deg:.1f}deg_seed{load_seed}",
                group="interleaved_training",
                config={
                    **config,
                    "phase_1_angle_deg": np.degrees(phase_1_reach_angle),
                    "phase_2_angle_deg": second_deg,
                    "phase_1_rotation": phase_1_rotation,
                    "phase_2_rotation": phase_2_rotation,
                    "phase_prob": args.phase_prob
                },
                reinit=True
            )

        # -----------------------------------------
        # Interleaved training
        # -----------------------------------------
        total_episodes = args.total_episodes if args.total_episodes is not None else 2 * config.get('max_episodes',
                                                                                                    10000)

        print(f"\nStarting interleaved training for {total_episodes} episodes...")
        print(f"Phase 1: {np.degrees(phase_1_reach_angle):.1f}° with {phase_1_rotation}° rotation")
        print(f"Phase 2: {second_deg:.1f}° with {phase_2_rotation}° rotation")
        print(f"Phase probability: {args.phase_prob:.2f}\n")

        results = train_interleaved_adaptation(
            config_phase1, config_phase2, agent,
            env_phase1, env_phase2,
            env_phase1_baseline, env_phase2_baseline,
            total_episodes=total_episodes,
            phase_prob=args.phase_prob,
            n_baseline_samples=args.n_baseline
        )

        # -----------------------------------------
        # Analysis: Use baseline (pre) + post-training samples
        # -----------------------------------------
        # Use the pre-training baseline samples (true "before" state)
        first_phase1 = results['phase1_baseline_actions']  # 5000 samples before training
        first_phase2 = results['phase2_baseline_actions']  # 5000 samples before training

        # Use the post-training samples (true "after" state)
        last_phase1 = results['phase1_post_actions']  # 5000 samples after training
        last_phase2 = results['phase2_post_actions']  # 5000 samples after training

        # Flatten training action histories for saving
        phase1_actions_flat = [action for episode in results['phase1_action_history'] for action in episode]
        phase2_actions_flat = [action for episode in results['phase2_action_history'] for action in episode]

        # -----------------------------------------
        # Save run logs to CSV (similar to adaptation_exp.py)
        # -----------------------------------------
        # Create episode indices for each phase
        phase1_episodes = np.arange(len(results['phase1_reward_history']))
        phase2_episodes = np.arange(len(results['phase2_reward_history']))

        # Flatten angle history for phase 1
        phase1_angle_history_flat = [angle for episode in results['phase1_action_history']
                                     for angle in [env_phase1.actions[act] for act in episode]]

        # Flatten angle history for phase 2
        phase2_angle_history_flat = [angle for episode in results['phase2_action_history']
                                     for angle in [env_phase2.actions[act] for act in episode]]

        # Save Phase 1 run log
        run_log_phase1 = pd.DataFrame({
            'episode': phase1_episodes,
            'reward': results['phase1_reward_history'],
            'critic_loss': results['phase1_critic_loss_history'],
            'actor_loss': results['phase1_actor_loss_history'],
            'nll_loss': results['phase1_nll_loss_history'],
            'angle_diff': results['phase1_angle_err_history'],
        })
        run_log_phase1_fn = 'interleaved_run_log_phase1_seed_{}_target_{}_rotation_{}.csv'.format(
            config['seed'],
            int(np.degrees(phase_1_reach_angle)),
            phase_1_rotation
        )
        run_log_phase1.to_csv(os.path.join(exp_save_dir, run_log_phase1_fn), index=False)
        print(f"Saved Phase 1 run log to {run_log_phase1_fn}")

        # Save Phase 2 run log
        run_log_phase2 = pd.DataFrame({
            'episode': phase2_episodes,
            'reward': results['phase2_reward_history'],
            'critic_loss': results['phase2_critic_loss_history'],
            'actor_loss': results['phase2_actor_loss_history'],
            'nll_loss': results['phase2_nll_loss_history'],
            'angle_diff': results['phase2_angle_err_history'],
        })
        run_log_phase2_fn = 'interleaved_run_log_phase2_seed_{}_target_{}_rotation_{}.csv'.format(
            config['seed'],
            int(second_deg),
            phase_2_rotation
        )
        run_log_phase2.to_csv(os.path.join(exp_save_dir, run_log_phase2_fn), index=False)
        print(f"Saved Phase 2 run log to {run_log_phase2_fn}")

        # Save data (existing action histogram data)
        df = pd.DataFrame({
            'baseline_phase1': pd.Series(first_phase1),
            'baseline_phase2': pd.Series(first_phase2),
            'post_phase1': pd.Series(last_phase1),
            'post_phase2': pd.Series(last_phase2),
        })
        fn = f'action_hist_data_interleaved_seed_{load_seed}_angle_{int(second_deg)}deg.csv'
        df.to_csv(os.path.join(exp_save_dir, fn), index=False)

        # -----------------------------------------
        # Plotting
        # -----------------------------------------
        action_angles = env_phase1.actions  # radians (both envs should have same actions)
        fig, ax = plt.subplots(figsize=(7, 7), subplot_kw=dict(projection='polar'))

        action_sets = [first_phase1, first_phase2, last_phase1, last_phase2]
        labels = [
            'Phase 1 baseline (pre-adaptation)',
            'Phase 2 baseline (pre-adaptation)',
            'Phase 1 late (post-adaptation)',
            'Phase 2 late (post-adaptation)'
        ]

        for k, actions in enumerate(action_sets):
            if len(actions) == 0:
                continue
            unique_actions, counts = np.unique(actions, return_counts=True)
            assert len(unique_actions) == len(action_angles)
            proportions = counts / len(actions)
            taken_angles = action_angles[unique_actions]
            bar_width = 2 * np.pi / len(env_phase1.actions)
            ax.bar(taken_angles, proportions, width=bar_width, alpha=0.7,
                   edgecolor='black', linewidth=0.5, color=palette[k], label=labels[k])

        ax.set_theta_zero_location('E')
        ax.set_theta_direction(1)
        ax.set_title(f'Interleaved Training: Action Distribution\n(n=5000 samples, pre vs. post training)', pad=20)
        ax.set_ylabel('Proportion', labelpad=30)
        ax.tick_params(axis='both', which='major', labelsize=12)

        # Reference lines
        compensated_phase1_angle = phase_1_reach_angle - np.radians(phase_1_rotation)
        compensated_phase2_angle = second_reach_angle - np.radians(phase_2_rotation)

        ax.axvline(phase_1_reach_angle, color=palette[0], linestyle='--', linewidth=2,
                   zorder=5, alpha=0.5, label=f'Phase 1 target')
        ax.axvline(second_reach_angle, color=palette[1], linestyle='--', linewidth=2,
                   zorder=5, alpha=0.5, label=f'Phase 2 target')
        ax.axvline(compensated_phase1_angle, color=palette[2], linestyle=':', linewidth=2,
                   zorder=5, alpha=0.5, label=f'Phase 1 compensated')
        ax.axvline(compensated_phase2_angle, color=palette[3], linestyle=':', linewidth=2,
                   zorder=5, alpha=0.5, label=f'Phase 2 compensated')

        ax.legend(loc='upper left', bbox_to_anchor=(0.05, 1.15), fontsize=9)
        plt.tight_layout()

        # Log to W&B
        if run is not None:
            wandb.log({
                "interleaved_action_distribution": wandb.Image(fig),
                "phase1_mean_reward": np.mean(results['phase1_reward_history']),
                "phase2_mean_reward": np.mean(results['phase2_reward_history']),
                "phase1_final_reward": np.mean(results['phase1_reward_history'][-100:]) if len(
                    results['phase1_reward_history']) >= 100 else np.mean(results['phase1_reward_history']),
                "phase2_final_reward": np.mean(results['phase2_reward_history'][-100:]) if len(
                    results['phase2_reward_history']) >= 100 else np.mean(results['phase2_reward_history']),
                "total_phase1_episodes": results['phase1_episode_count'],
                "total_phase2_episodes": results['phase2_episode_count'],
                "total_phase1_actions": len(phase1_actions_flat),
                "total_phase2_actions": len(phase2_actions_flat)
            })
            wandb.finish()

        # Save figure
        plt.savefig(os.path.join(exp_save_dir,
                                 f'interleaved_action_hist_angle_{int(second_deg)}deg.png'), dpi=300)
        plt.close(fig)

        print(f"\nCompleted training for angle {second_deg:.1f}°")
        print(f"Phase 1 episodes: {results['phase1_episode_count']}")
        print(f"Phase 2 episodes: {results['phase2_episode_count']}")
        print(f"Phase 1 total actions: {len(phase1_actions_flat)}")
        print(f"Phase 2 total actions: {len(phase2_actions_flat)}")
        print(f"Phase 1 mean reward: {np.mean(results['phase1_reward_history']):.3f}")
        print(f"Phase 2 mean reward: {np.mean(results['phase2_reward_history']):.3f}")

"""
In this file, we run an adaptation experiment with many combinations of two different targets (with opposite rotations)
Creates:
- 1 W&B run for Phase 1 (first reach angle, +30°)
- N W&B runs for Phase 2 (one per second_reach_angle, -30°), named by the angle in degrees
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

from definitions import data_root  # todo: load/save model from path relative to this
from core.model_loading_utils import load_trained_full_model_basetask


phase_1_reach_angle = np.radians(135)  # radians (90 degrees)
# phase_2_reach_angles = np.linspace(0, 2 * np.pi, config['num_actions'], endpoint=False)
phase_2_reach_angles = [np.radians(135 + 180)]  # 270 degrees only; change to full set as needed
phase_1_rotation = -30  # degrees
phase_2_rotation = 30  # degrees


if __name__ == '__main__':
    from adaptation.config import config
    from adaptation.adaptation_exp import train_adapatation_experiment
    from core.continuous_env import ReachTask

    data_save_dir = os.path.join(data_root, 'doupleAdaptationExp')
    if not os.path.exists(data_save_dir):
        os.makedirs(data_save_dir)

    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=9, help='Random seed for the experiment')
    args = parser.parse_args()
    config['seed'] = args.seed

    palette = sns.color_palette('Paired')

    # -----------------------------------------
    # Phase 1: first reach angle (+30° rotation)
    # -----------------------------------------
    # second_reach_angles = np.linspace(0, 2 * np.pi, config['num_actions'], endpoint=False)

    # configure Phase 1 on the base config (mutates config intentionally)
    config['reach_angle'] = float(phase_1_reach_angle)
    config['rotation_angle'] = phase_1_rotation        # degrees
    config['load_different_fg'] = False  # use the action embedding model learned after initial learning
    config['save_model'] = True
    config['post_adaptation_model_save_dir'] = os.path.join(
        data_root, 'savedModels', 'doubleAdaptationExp', 'phase_1'
    )

    # choose seed based on reach angle
    load_seed = config['seed']

    config['load_full_model_file'] = os.path.join(
        config['full_model_load_path'],
        'fully_trained_policy_model_one_target_seed_{}_weight_decay_0.0001_tanh_policy_mean_target_{}_n_actions_24.pth'
        .format(load_seed, int(np.degrees(config['reach_angle'])))
    )

    # W&B run for Phase 1
    run_phase1 = None
    if config.get('log_to_wandb', False):
        run_phase1 = wandb.init(
            project='ActionEmbeddingRotationGeneralization',
            name=f"phase1_first_angle_{int(np.degrees(config['reach_angle']))}deg_rot{config['rotation_angle']}",
            group="phase_1",
            config=config,
            reinit=True
        )

    env = ReachTask(config, adaptation_rotation=np.radians(config['rotation_angle']))
    agent = load_trained_full_model_basetask(config, env, np.round(np.degrees(config['reach_angle'])).astype(int),
                                             load_seed)

    phase1_results = train_adapatation_experiment(config, agent)

    first_5000_steps = phase1_results['action_history'][:5000]
    last_5000_steps = phase1_results['action_history'][-5000:]

    if run_phase1 is not None:
        wandb.log({
            "mean_reward_phase1": np.mean(phase1_results['reward_history'])
        })
        wandb.finish()

    # ---------------------------------------------------------
    # Phase 2: per-second_reach_angle runs (-30°), separate W&B
    # ---------------------------------------------------------
    embeddings, _ = agent.get_action_embeddings_via_g(return_next_state=True)
    # embeddings_dict = {a_id: emb for a_id, emb in enumerate(embeddings)}

    action_idx = {a: idx for a, idx in zip(agent.env.actions, range(agent.env.n_actions))}

    for i, second_reach_angle in enumerate(phase_2_reach_angles):

        pre_rotation_a_idx = action_idx[second_reach_angle]

        # Build a per-run config snapshot so each W&B run logs its own state
        phase2_cfg = dict(config)  # shallow copy is fine for flat config dicts
        phase2_cfg['reach_angle'] = float(second_reach_angle)  # radians
        phase2_cfg['rotation_angle'] = phase_2_rotation  # degrees

        # For the second angle we don't load the policy (simulate via mean over g for that action)
        phase2_cfg['load_full_model_file'] = None
        phase2_cfg['post_adaptation_model_save_dir'] = os.path.join(
            data_root, 'savedModels', 'doubleAdaptationExp', 'phase_2'
        )
        phase2_cfg['import_policy_mean'] = True
        phase2_cfg['policy_mean_to_import'] = torch.tensor(embeddings[pre_rotation_a_idx])

        # Name run by angle in degrees
        second_deg = np.degrees(second_reach_angle)

        run_phase2 = None
        if phase2_cfg.get('log_to_wandb', False):
            run_phase2 = wandb.init(
                project='ActionEmbeddingRotationMultiTarget',
                name=f"phase2_second_angle_{second_deg:.1f}deg_rot{phase2_cfg['rotation_angle']}",
                group="phase_2",
                config={**phase2_cfg, "second_reach_angle_deg": second_deg},
                reinit=True
            )

        # Train for this angle
        agent_phase2 = copy.deepcopy(phase1_results['agent'])

        env_phase2 = ReachTask(phase2_cfg, adaptation_rotation=np.radians(phase2_cfg['rotation_angle']))
        agent_phase2.env = env_phase2

        phase2_results = train_adapatation_experiment(phase2_cfg, agent_phase2)

        first_5000_steps2 = phase2_results['action_history'][:5000]
        last_5000_steps2 = phase2_results['action_history'][-5000:]

        df = pd.DataFrame({
            'first_5000_steps_phase1': [i for i in first_5000_steps for i in i],
            'last_5000_steps_phase1': [i for i in last_5000_steps for i in i],
            'first_5000_steps_phase2': [i for i in first_5000_steps2 for i in i],
            'last_5000_steps_phase2': [i for i in last_5000_steps2 for i in i]
        })
        fn = f'action_hist_data_seed_{load_seed}_angle_{int(second_deg)}deg.csv'
        df.to_csv(os.path.join(data_save_dir, fn), index=False)

        # ---------- Plot ----------
        action_angles = agent_phase2.env.actions  # radians, shape: (num_actions,)
        fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(projection='polar'))

        for k, actions in enumerate([first_5000_steps, first_5000_steps2, last_5000_steps, last_5000_steps2]):
            unique_actions, counts = np.unique(actions, return_counts=True)
            assert len(unique_actions) == len(agent.env.actions), "Not all actions were taken"
            proportions = counts / len(actions)
            taken_angles = action_angles[unique_actions]
            bar_width = 2 * np.pi / len(agent.env.actions)
            ax.bar(taken_angles, proportions, width=bar_width, alpha=0.7,
                   edgecolor='black', linewidth=0.5, color=palette[k])

        ax.set_theta_zero_location('E')  # 0° at the right (East)
        ax.set_theta_direction(1)        # positive angles counter-clockwise
        ax.set_title('RL Agent Action Distribution\n(Proportion of Total Actions)', pad=20)
        ax.set_ylabel('Proportion', labelpad=30)
        ax.tick_params(axis='both', which='major', labelsize=12)

        legend_elements = [
            plt.Rectangle((0, 0), 1, 1, facecolor=palette[0], label='Before +30° adaptation'),
            plt.Rectangle((0, 0), 1, 1, facecolor=palette[1], label='Before -30° adaptation'),
            plt.Rectangle((0, 0), 1, 1, facecolor=palette[2], label='After +30° adaptation'),
            plt.Rectangle((0, 0), 1, 1, facecolor=palette[3], label='After -30° adaptation')
        ]
        ax.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(0.1, 1.1))

        # Reference lines (all radians)
        second_reach_angle_rad = phase2_cfg['reach_angle']
        compensated_phase1_angle = phase_1_reach_angle - np.radians(phase_1_rotation)
        compensated_phase2_angle = second_reach_angle_rad - np.radians(phase_2_rotation)

        ax.axvline(phase_1_reach_angle, color=palette[0], linestyle='--', linewidth=2, zorder=5)
        ax.axvline(second_reach_angle_rad, color=palette[1], linestyle='--', linewidth=2, zorder=5)
        ax.axvline(compensated_phase1_angle, color=palette[2], linestyle='--', linewidth=2, zorder=5)
        ax.axvline(compensated_phase2_angle, color=palette[3], linestyle='--', linewidth=2, zorder=5)

        plt.tight_layout()

        # Log & close
        if run_phase2 is not None:
            wandb.log({
                "two_target_hists": wandb.Image(fig),
                "second_reach_angle_deg": second_deg,
                "mean_reward_phase2": np.mean(phase2_results['reward_history'])
            })
            wandb.finish()

        plt.savefig(os.path.join(data_save_dir,
                                 f'two_target_action_hist_angle_{int(second_deg)}deg.png'), dpi=300)
        plt.show()
        plt.close(fig)

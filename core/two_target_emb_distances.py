import os
import numpy as np
import pandas as pd
import math
from copy import deepcopy
from core.policy_learning import train_agent
from core.config import config


def normalized_angle(v1, v2):
    """Returns angle in radians and normalized angle (0=parallel, 1=orthogonal)"""
    cos_theta = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
    cos_theta = np.clip(cos_theta, -1, 1)  # numerical safety
    theta = np.arccos(cos_theta)
    normalized = theta / np.pi
    return theta, normalized


def run_two_target_experiment(seed, delta_degrees=30):
    """Run two-target training for a single seed, return angles between policies and embeddings"""

    base_angle = math.pi
    reach_angles_to_train = [base_angle + math.radians(delta_degrees), base_angle]

    policy_means = []
    trained_angle_inds = []

    for cue_ind, reach_angle in enumerate(reach_angles_to_train):
        cfg = deepcopy(config)
        cfg['reach_angle'] = reach_angle
        cfg['seed'] = seed
        cfg['log_to_wandb'] = False  # disable for batch runs

        results = train_agent(cfg)

        agent = results['agent']
        agent.env.reset()
        features = agent.env.get_features(agent.env.current_xy)
        policy_mean = agent.actor(features).detach().numpy()
        policy_means.append(policy_mean)
        trained_angle_inds.append(results['trained_angle_ind'])

    # Get embeddings for the two trained angles
    embeddings = agent.get_action_embeddings_via_g()  # shape [n_actions, 2]

    embedding_0 = embeddings[trained_angle_inds[0]]
    embedding_1 = embeddings[trained_angle_inds[1]]

    policy_mean_0 = policy_means[0]
    policy_mean_1 = policy_means[1]

    # Compute angles
    embedding_angle, embedding_normalized = normalized_angle(embedding_0, embedding_1)
    policy_angle, policy_normalized = normalized_angle(policy_mean_0, policy_mean_1)

    return {
        'seed': seed,
        'delta_degrees': delta_degrees,
        'embedding_angle': embedding_angle,
        'embedding_normalized': embedding_normalized,
        'policy_angle': policy_angle,
        'policy_normalized': policy_normalized,
    }


if __name__ == '__main__':
    seeds = range(10)
    delta_degrees = 165

    results_list = []
    for seed in seeds:
        print(f"Running seed {seed}...")
        result = run_two_target_experiment(seed, delta_degrees=delta_degrees)
        results_list.append(result)
        print(
            f"  Embedding angle: {result['embedding_angle']:.4f} rad, normalized: {result['embedding_normalized']:.4f}")
        print(f"  Policy angle: {result['policy_angle']:.4f} rad, normalized: {result['policy_normalized']:.4f}")

    # Convert to dataframe and save
    df = pd.DataFrame(results_list)
    output_path = os.path.join(config['fig_dir'], f'two_target_angles_delta{delta_degrees}.csv')
    df.to_csv(output_path, index=False)
    print(f"\nSaved to {output_path}")

    # Summary statistics
    print("\n=== Summary ===")
    print(f"Embedding normalized: {df['embedding_normalized'].mean():.4f} +/- {df['embedding_normalized'].std():.4f}")
    print(f"Policy normalized: {df['policy_normalized'].mean():.4f} +/- {df['policy_normalized'].std():.4f}")

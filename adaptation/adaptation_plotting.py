import matplotlib.pyplot as plt
import pandas as pd
import torch
import random
import os
import numpy as np
import math
from scipy.stats import circmean
import seaborn as sns

from core.agent import ACLearningAgentWithEmbedding, softmax_with_temperature, \
    add_noise_to_action_probs
from core.continuous_env import ReachTask
from core.plotting import set_plotting_defaults, find_angle_difference
from adaptation.config import config

def load_test_policies(test_angles, config, outdir, centre_coord, reach_length, grid_size):
    test_policy_means = []
    test_goals = []
    for test_angle in test_angles:
        target_angles = np.linspace(test_angle, 2 * np.pi, 1, endpoint=False)
        target_xs = centre_coord[0] + reach_length * np.cos(target_angles)
        target_ys = centre_coord[1] + reach_length * np.sin(target_angles)
        targets = [(xy[0], xy[1]) for xy in zip(target_xs, target_ys)]
        max_steps = config["max_steps"]
        goal_test = centre_coord if config["random_start_point"] else targets[0]
        test_goals.append(goal_test)
        test_env = ReachTask(goal_test, grid_size_rows=grid_size[0], grid_size_cols=grid_size[1],
                             num_actions=config["num_actions"], max_trial_length=max_steps,
                             basis_type='Fourier', fourier_order=3, adaptation_rotation=0)
        model_filename = 'fully_trained_policy_model_one_target_seed_{}_weight_decay_0.0001_tanh_policy_mean_target_{}_n_actions_{}.pth'.format(
                            2,  round(np.degrees(test_angle)), config["num_actions"])
        test_model_path = os.path.join(outdir, model_filename)
        test_agent = ACLearningAgentWithEmbedding(test_env, grid_size[0], grid_size[1],
                                                  embedding_dim=config["embedding_dim"],
                                                  actor_lr=config["actor_lr"], critic_lr=config["critic_lr"],
                                                  fg_lr=config["fg_lr"], inv_temp=config["inv_temp"],
                                                  policy_std=config["policy_std"],
                                                  full_model_load_path=test_model_path,
                                                  actor_plastic=False, critic_plastic=False, g_plastic=False)
        test_agent.env.reset()
        features = test_agent.env.get_features(test_agent.env.current_xy)
        action_ind, embedding, mean_emb, logstd_emb = test_agent.select_action(features, random_policy=False, noise=True)
        test_policy_means.append(mean_emb)
    return test_policy_means, test_goals
import math
from definitions import initial_learning_fig_dir, full_model_load_path, paper_model_path, paper_fig_dir
import numpy as np
from pathlib import Path


config = {
    "actor_lr": 0.00001,  # actor lr about an order of magnitude smaller than critic lr (see chandak et al 2019)
    "critic_lr": 0.0001,
    "fg_lr": 0.,
    "w_decay_fg": 0,
    "embedding_dim": 2,
    "num_actions": 24,
    "grid_size": (20, 20),
    "reach_length": 1,
    "reach_angle": np.radians(135), #2 * math.pi/4,
    "max_steps": 1,
    "max_episodes": 200000,  #310000 for one target policy, 3.5M for embeddings
    "max_reward_policy_annealing": 0.4,
    "policy_std_max": 0.8,
    "max_reward_target_annealing": 0.5,
    "reward_radius_max": 0.5,
    "reward_radius_min": 0.2,
    "reward_for_hit": 1.0,
    "penalty_for_miss": -0.1,
    "log_to_wandb": False,
    "fourier_order": 3,
    "inv_temp": 0.8,  # 0.8 for original model
    "policy_noise": 0.008,
    "discount_factor": 0.99,
    "use_random_policy": False,
    "log_interval": 10000,
    "seed": 9,
    "policy_std": 0.2,
    "fg_load_path": Path(paper_model_path) / 'action_embedding_model_seed_0_weight_decay_fg_0.0001_n_action_24_fourier_basis.pth',
    "fg_load_dir": Path(paper_model_path),
    "fig_dir": paper_fig_dir,
    "save_model": True,
    "save_model_dir": paper_model_path, #full_model_load_path,
    "save_figs": True,
    "run_for_all_reach_angles": False,
}
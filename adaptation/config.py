import math
from pathlib import Path
from definitions import fig_dir, full_model_load_path, trained_policy_model_fn, paper_model_path


config = {
    "seed": 9,
    "actor_lr": 0.,
    "critic_lr": 0.,
    "fg_lr": 0.0001,  # initial learning: 0.0008
    "w_decay_fg": 0,
    "embedding_dim": 2,
    "num_actions": 24,
    "grid_size": (20, 20),
    "reach_length": 1,
    "reach_angle": math.radians(135),
    "max_steps": 1,
    "max_episodes": 100000, #used to be 40000
    "log_to_wandb": False,
    "num_RBFs_per_row": 5,
    "reward_radius_max": 0.5,
    "reward_radius_min": 0.2,
    "reward_for_hit": 1.0,
    "penalty_for_miss": -0.1,
    "discount_factor": 0.99,
    "inv_temp": 2.5,  # this is higher than for normal learning currently
    "use_random_policy": False,
    "random_start_point": False,
    "log_interval": 5000,
    "fourier_order": 3,
    "policy_std": 0.2,
    "full_model_load_path": full_model_load_path,
    "fig_dir": fig_dir,
    "post_adaptation_model_save_dir": paper_model_path,
    "fg_load_path": Path(paper_model_path) / 'action_embedding_model_seed_2_weight_decay_fg_0.0001.pth',
    "save_model": True,
    'rotation_angle': -30,
    'load_different_fg': False,
    'policy_noise': 0.008,
    'save_figs_locally': False,
    'angle_diff_criterion': 10.,
    'early_stopping_criterion': 30000 * 100000,
    'import_policy_mean': False,
    'policy_mean_to_import': None,
}

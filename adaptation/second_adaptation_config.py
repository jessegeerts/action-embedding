import math
from pathlib import Path
from definitions import fig_dir, full_model_load_path, trained_policy_model_fn, paper_model_path
import os

config = {
    "actor_lr": 0.00001,
    "critic_lr": 0.008,
    "fg_lr": 0.00008, #same as initial learning 0.0008
    "w_decay_fg": 0,
    "embedding_dim": 2,
    "num_actions": 24,
    "grid_size": (10, 10),
    "reach_length": 1,
    "reach_angle": 3 * math.pi/2,
    "max_steps": 1,
    "max_episodes": 200000, #used to be 40000
    "log_to_wandb": False,
    "num_RBFs_per_row": 5,
    "width_RBF": 1.2,
    "inv_temp": 1.5, # this is higher than for normal learning currently
    "use_random_policy": False,
    "random_start_point": False,
    "log_interval": 10000,
    "seed": 5,
    "policy_std": 0.2,
    "full_model_load_path": full_model_load_path,
    "fig_dir": fig_dir,
    "fg_load_path": Path(paper_model_path) / 'post_adaptation_model_seed_0_rotation_30_temp_1.5_weight_decay_0.0001_tanh_policy_mean_n_actions_24.pth',
    "save_model": True,
    'rotation_angle': -30,
    'load_different_fg': True,  # JPG: note, this was set to True. change back when needed
    'policy_noise': 0.008,
    'angle_diff_criterion': 10.,
    'early_stopping_criterion': 30000,
    'save_figs_locally': False,
}
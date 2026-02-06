from adaptation.adaptation_exp import train_adapatation_experiment
from adaptation.second_adaptation_config import config
import wandb
import os
import math
import numpy as np


for seed in range(0, 1):
    config['seed'] = seed

    if config['reach_angle'] == math.pi/2:
        load_seed = 2
    else:
        load_seed = 6

    config['load_full_model_file'] = os.path.join(config['full_model_load_path'], 'fully_trained_policy_model_one_target_seed_{}_weight_decay_0.0001_tanh_policy_mean_target_{}_n_actions_24.pth'.format(load_seed, int(np.degrees(config['reach_angle']))))

    if config['log_to_wandb']:
        wandb.init(
            project='ActionEmbeddingRotationSecondTarget',
            config=config
        )
    agent, action_history, state_history, critic_loss_history, reward_history, target_angles, centre_coord = \
        train_adapatation_experiment(config)
    if config['log_to_wandb']:
        wandb.finish()  # End the current run



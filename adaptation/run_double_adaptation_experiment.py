"""In this file, we run an adaptation experiment with two different targets and opposite rotations.
"""
import math
import wandb
from definitions import data_root  # todo: load/save model from path relative to this


# run_and_log_adaptation_experiment is the new function to run
# train_adaptation_experiment now takes cfg and agent
# note that train_adaptation_experiment now returns a results dict
# todo: change code here to handle new logic

if __name__ == '__main__':
    from adaptation.config import config
    from adaptation.adaptation_exp import train_adapatation_experiment
    import os
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns

    palette = sns.color_palette('Paired')

    if config['log_to_wandb']:
        wandb.init(
            project='ActionEmbeddingRotationGeneralization',
            config=config
        )

    ####################################################################################################################
    ########## RUN EXPERIMENT WITH TARGET AT 90 DEGREES AND ROTATION OF 30 DEGREES #####################################
    ####################################################################################################################
    config['reach_angle'] = math.pi / 2  # radians
    config['rotation_angle'] = 30  # degrees
    config['load_different_fg'] = False  # we use  the action embedding model learned after initial learning
    config['save_model'] = True  # we save the model after adaptation
    config['post_adaptation_model_save_dir'] = os.path.join(data_root, 'savedModels', 'doubleAdaptationExp', 'phase_1')
    # random seeds that learn similar policies (comparable distances from embedding produced by g) for the two targets
    if config['reach_angle'] == math.pi/2:
        load_seed = 2
    else:
        load_seed = 6


    config['load_full_model_file'] = os.path.join(config['full_model_load_path'], 'fully_trained_policy_model_one_target_seed_{}_weight_decay_0.0001_tanh_policy_mean_target_{}_n_actions_24.pth'.format(load_seed, int(np.degrees(config['reach_angle']))))

    agent, action_history, state_history, critic_loss_history, reward_history, target_angles, centre_coord = \
        train_adapatation_experiment(config)

    first_5000_steps = action_history[:5000]  # save the first 5000 steps to compare with the second adaptation
    last_5000_steps = action_history[-5000:]  # save the last 5000 steps to compare with the second adaptation
    ####################################################################################################################
    ########## RUN EXPERIMENT WITH TARGET AT 270 DEGREES AND ROTATION OF -30 DEGREES ###################################
    ####################################################################################################################
    config['reach_angle'] = 3 * math.pi / 2  # radians
    config['rotation_angle'] = -30  # degrees
    config['load_different_fg'] = True  # we use the action embedding model learned after the first adaptation
    config['fg_load_path'] = os.path.join(data_root, 'savedModels', 'doubleAdaptationExp', 'phase_1',
                                          'post_adaptation_model_seed_0_rotation_30_temp_1.5_weight_decay_0.0001_tanh_policy_mean_n_actions_24_target_90.0.pth')
    if config['reach_angle'] == math.pi/2:
        load_seed = 2
    else:
        load_seed = 6
    config['load_full_model_file'] = os.path.join(config['full_model_load_path'], 'fully_trained_policy_model_one_target_seed_{}_weight_decay_0.0001_tanh_policy_mean_target_{}_n_actions_24.pth'.format(load_seed, int(np.degrees(config['reach_angle']))))
    config['post_adaptation_model_save_dir'] = os.path.join(data_root, 'savedModels', 'doubleAdaptationExp', 'phase_2')

    agent2, action_history2, state_history2, critic_loss_history2, reward_history2, target_angles2, centre_coord2 = \
        train_adapatation_experiment(config)

    first_5000_steps2 = action_history2[:5000]  # save the first 5000 steps to compare with the second adaptation
    last_5000_steps2 = action_history2[-5000:]  # save the last 5000 steps to compare with the second adaptation

    action_angles = agent.env.actions  # shape: (24,) with angles in radians

    # Create polar histogram
    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(projection='polar'))

    for i, actions in enumerate([first_5000_steps, first_5000_steps2, last_5000_steps, last_5000_steps2]):

        unique_actions, counts = np.unique(actions, return_counts=True)
        assert len(unique_actions) == len(agent.env.actions), "Not all actions were taken"
        proportions = counts / len(actions)  # Convert to proportions
        taken_angles = action_angles[unique_actions]


        bar_width = 2 * np.pi / len(agent.env.actions)
        bars = ax.bar(taken_angles, proportions, width=bar_width, alpha=0.7, edgecolor='black', linewidth=0.5,
                      color=palette[i])

    ax.set_theta_zero_location('E')  # 0 radians at the top
    ax.set_theta_direction(1)
    ax.set_title('RL Agent Action Distribution\n(Proportion of Total Actions)', pad=20)

    ax.set_ylabel('Proportion', labelpad=30)
    # # Optional: show all possible action directions as reference lines
    # for angle in agent.env.actions:
    #     ax.axvline(angle, color='gray', alpha=0.3, linewidth=0.5)

    legend_elements = [
        plt.Rectangle((0, 0), 1, 1, facecolor=palette[0], label=f'Before +30° adaptation'),
        plt.Rectangle((0, 0), 1, 1, facecolor=palette[1], label=f'Before -30° adaptation'),
        plt.Rectangle((0, 0), 1, 1, facecolor=palette[2], label=f'After +30° adaptation'),
        plt.Rectangle((0, 0), 1, 1, facecolor=palette[3], label=f'After -30° adaptation')
    ]
    ax.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(0.1, 1.1))

    # Increase text sizes
    ax.set_title('RL Agent Action Distribution\n(Proportion of Total Actions)',
                 pad=20, fontsize=16, fontweight='bold')
    ax.set_ylabel('Proportion', labelpad=30, fontsize=14)

    # Increase tick label sizes
    ax.tick_params(axis='both', which='major', labelsize=12)

    plt.tight_layout()
    plt.show()
# show:
# - can learn opposite rotation independently
# - cannot learn same side opposite rotation
# - width of generalization bump determines how close a reach target can still be learned before opposite rotation interferes
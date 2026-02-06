import os
from core.policy_learning import train_agent
from core.config import config  # had its own config before!
from core.plotting import plot_embedding_overlaid_two_cues_post_hoc
from copy import deepcopy
import matplotlib.pyplot as plt
import numpy as np
import math
import wandb

# FIGURE 2
# This version just trains without blocks because it's equivalent and much more interpretable
# this is run with either plus 30 or plus 165 degrees
base_angle = math.pi
reach_angles_to_train = [base_angle + math.radians(30), base_angle]
policy_means = []
policy_means_dict = {}
embedding_histories = []
trained_angle_inds = []
for cue_ind, reach_angle in enumerate(reach_angles_to_train):
    cfg = deepcopy(config)
    cfg['reach_angle'] = reach_angle
    if config['log_to_wandb']:
        wandb.init(
            project='ActionEmbeddingContinuous-StatesTwoTargets',
            config=cfg
        )
    results = train_agent(cfg)

    agent = results['agent']
    agent.env.reset()
    features = agent.env.get_features(agent.env.current_xy)
    policy_mean = agent.actor(features).detach().numpy()
    policy_means.append(policy_mean)
    embedding_histories.append(results['embedding_history'])
    trained_angle_inds.append(results['trained_angle_ind'])
    policy_means_dict[cue_ind] = policy_mean
colors = plt.cm.twilight(np.linspace(0, 1, config['num_actions']))
plotting_colors = colors[trained_angle_inds]
embeddings = agent.get_action_embeddings_via_g()
embedding_fig, embedding_ax = plot_embedding_overlaid_two_cues_post_hoc(embeddings, policy_means_dict,
                                                                               config['policy_std'], agent.env,
                                                                               embedding_histories, colors=plotting_colors)
embedding_fig.savefig(os.path.join(config['fig_dir'], f'two_target_embeddings_{reach_angles_to_train[0]:.2f}_{reach_angles_to_train[1]:.2f}.pdf'))
plt.show()

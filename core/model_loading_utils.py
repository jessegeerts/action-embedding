"""Some functions for making loading trained models easier."""
import os

from core.agent import ACLearningAgentWithEmbedding
from definitions import paper_model_path


def load_trained_full_model_basetask(cfg, env, reach_angle_degrees, seed):
    """Load model that has been trained on the base task (one specific reach angle, no rotation).

    reach_angle_degrees and seed override cfg settings for these parameters.

    :param cfg:
    :param env:
    :param n_acts:
    :param (int) reach_angle_degrees:
    :param seed:
    :return:
    """
    model_weights_dir = paper_model_path
    n_acts = cfg['num_actions']
    model_weights_fn = f'fully_trained_policy_model_one_target_seed_{seed}_weight_decay_0.0001_tanh_policy_mean_target_{reach_angle_degrees}_n_actions_{n_acts}.pth'
    full_model_load_path = os.path.join(model_weights_dir, model_weights_fn)
    cfg['load_full_model_file'] = full_model_load_path
    if cfg['load_different_fg']:
        fg_load_path = cfg['fg_load_path']
    else:
        fg_load_path = None
    agent = ACLearningAgentWithEmbedding(env, cfg,
                                         full_model_load_path=full_model_load_path,
                                         fg_load_path=fg_load_path,
                                         g_plastic=False,
                                         f_plastic=True,
                                         actor_plastic=False,
                                         critic_plastic=False)  # todo: make these options configurable
    return agent

import matplotlib.pyplot as plt
import pandas as pd
import torch
import random
from core.plotting import plot_loss_rolling, plot_angle_diffs_rolling, log_weights_to_wandb, \
    set_plotting_defaults, plot_embeddings_with_gaussian_2d, plot_f_output, \
    find_angle_difference
from core.continuous_env import ReachTask
from adaptation.adaptation_generalization_test import make_generalization_plot, calculate_generalization
import distinctipy
from matplotlib.colors import ListedColormap
import wandb
import os
import os.path as op
from definitions import paper_fig_dir
from core.model_loading_utils import load_trained_full_model_basetask
from adaptation.config import config
import numpy as np
import copy
import argparse


def get_embedding_from_action(one_hot_action, W_f_pseudo_inv, b_f):
    # Subtract the bias and apply the pseudo-inverse of W_f
    embedding = W_f_pseudo_inv @ (one_hot_action - b_f)
    return embedding


def train_adapatation_experiment(cfg, agent, base_agent=None):
    set_plotting_defaults()
    # set random seed
    random.seed(cfg["seed"])
    np.random.seed(cfg["seed"])
    torch.manual_seed(cfg["seed"])
    # make sure save out path exists, else create
    if not os.path.exists(cfg['fig_dir']):
        os.makedirs(cfg['fig_dir'])
    episodes_to_save = [0, int(cfg['max_episodes'] - cfg['log_interval'])]

    colors = distinctipy.get_colors(agent.env.n_actions, pastel_factor=0.7)
    cmap = ListedColormap(colors)
    # Specify different learning rates for actor, critic, and f & g

    episodes = cfg["max_episodes"]
    max_steps = cfg["max_steps"]

    action_history = []
    angle_history = []
    predicted_action_history = []

    critic_loss_history = []
    actor_loss_history = []
    nll_loss_history = []
    reward_history = []
    state_history, steps_history = [], []
    cumulative_reward_across_episodes = 0
    embedding_history = []
    angle_err_history = []
    num_episodes_below_criterion = 0
    for episode in range(episodes):
        agent.env.reset()
        features = agent.env.get_features(agent.env.current_xy)
        done = False
        steps = 0
        total_nll_loss = 0
        total_actor_loss = 0
        total_critic_loss = 0
        episode_actions = []
        episode_angles = []
        episode_a_taken = []
        episode_a_predicted = []
        episode_states = [agent.env.start_xy]
        episode_reward = 0

        random_policy = cfg['use_random_policy']

        while steps < max_steps:
            # Zero the gradients for all modules
            agent.actor_optimizer.zero_grad()
            agent.critic_optimizer.zero_grad()
            agent.f_g_optimizer.zero_grad()

            action_ind, embedding, mean_emb, logstd_emb = agent.select_action(features, random_policy=random_policy,
                                                                              import_policy_mean=cfg[
                                                                                  'import_policy_mean'],
                                                                              policy_mean=cfg[
                                                                                  'policy_mean_to_import'])
            action = agent.env.actions[action_ind]
            angular_error = find_angle_difference(agent.env, action)
            next_state, reward, done = agent.env.act(action)
            next_state_features = agent.env.get_features(next_state)

            # get predicted action from f module
            embs = agent.g(features, next_state_features)
            action_logits = agent.f(embs)
            action_probs = torch.softmax(action_logits, dim=0)
            predicted_act = torch.argmax(action_probs).item()
            nll_loss, actor_loss, critic_loss = agent.update(features, action_ind, embedding, next_state_features,
                                                             reward, done)
            # Accumulate Losses for this episode
            total_actor_loss += actor_loss
            total_critic_loss += critic_loss
            total_nll_loss += nll_loss

            episode_actions.append(action_ind)
            episode_angles.append(action)
            episode_states.append(next_state)
            episode_states.append(next_state)
            episode_a_predicted.append(predicted_act)
            episode_reward += reward
            cumulative_reward_across_episodes += reward
            embedding_history.append(embedding.detach())
            steps += 1

        reward_history.append(episode_reward)
        # calculate average losses for the episode
        avg_critic_loss_ep = total_critic_loss / steps if steps > 0 else 0
        avg_actor_loss_ep = total_actor_loss / steps if steps > 0 else 0
        avg_emb_loss_ep = total_nll_loss / steps if steps > 0 else 0

        action_history.append(episode_actions)
        angle_history.append(episode_angles)
        predicted_action_history.append(episode_a_predicted)
        state_history.append(episode_states)
        critic_loss_history.append(avg_critic_loss_ep)
        actor_loss_history.append(avg_actor_loss_ep)
        nll_loss_history.append(avg_emb_loss_ep)
        steps_history.append(steps)
        angle_err_history.append(angular_error)

        if episode % cfg['log_interval'] == 0:
            print(f'Episode {episode}, Steps Taken: {np.mean(steps_history[-100:])}')
            print(f'Actor loss: {np.mean(actor_loss_history[-100:])},' +
                  f'Critic loss: {np.mean(critic_loss_history[-100:])}, ' +
                  f'Supervised loss: {np.mean(nll_loss_history[-100:])}')

            # calculate recent action prediction accuracy
            recent_actions = action_history[-cfg['log_interval']:]
            recent_pred_actions = predicted_action_history[-cfg['log_interval']:]
            recent_correct = [a[0] == b[0] for a, b in zip(recent_actions, recent_pred_actions)]

            mean_angle_diff = np.mean(angle_err_history[-cfg['log_interval']:])
            if np.abs(mean_angle_diff) < cfg['angle_diff_criterion']:
                num_episodes_below_criterion += cfg['log_interval']
            if cfg["log_to_wandb"]:
                wandb.log({
                    "critic_loss": np.mean(critic_loss_history[-cfg['log_interval']:]),
                    "actor_loss": np.mean(actor_loss_history[-cfg['log_interval']:]),
                    "embedding_loss": np.mean(nll_loss_history[-cfg['log_interval']:]),
                    "episode": episode,
                    "n_steps": steps,
                    "cum_reward": cumulative_reward_across_episodes,
                    "episode_reward": episode_reward,
                    "avg_reward": np.mean(reward_history[-cfg['log_interval']:]),
                    "angle_difference": mean_angle_diff,
                    "abs(angle_difference)": np.abs(mean_angle_diff),
                    "action_pred_accuracy": np.sum(recent_correct) / len(recent_correct),
                })
                embeddings = agent.get_action_embeddings_via_g()
                if not cfg['import_policy_mean']:
                    mean_embedding_last_trial = mean_emb.detach().numpy()
                else:
                    mean_embedding_last_trial = cfg['policy_mean_to_import'].detach().numpy()
                std_embedding = np.exp(logstd_emb.detach().numpy())
                embedding_fig, embedding_ax = plot_embeddings_with_gaussian_2d(embeddings, mean_embedding_last_trial,
                                                                               std_embedding, agent.env,
                                                                               embedding_history, cmap=cmap)

                # Log the image to wandb
                wandb.log({"embedding_plot": wandb.Image(embedding_fig)})
                # Optionally, you can close the plot after logging to free up memory
                plt.close(embedding_fig)

                # plot histogram of actions taken and log to wandb
                fig, axs = plt.subplots(1, 1, figsize=(1.4, 1.4))
                local_action_history = action_history[-cfg['log_interval']:]
                local_action_history = [act for ep in local_action_history for act in ep]
                action_hist = np.array(local_action_history).flatten()
                # Define the bins
                bins = np.arange(0, agent.env.n_actions + 1)
                # Plot the histogram
                n, bins, patches = plt.hist(action_hist, bins=bins, density=True)

                # Apply colors to each bin
                for patch, color in zip(patches, colors):
                    patch.set_facecolor(color)
                axs.set_xlabel('Action Index')
                axs.set_ylabel('Frequency')
                axs.spines['right'].set_visible(False)
                axs.spines['top'].set_visible(False)

                # Find optimal angles (pre and post rotation)
                optimal_action_pre = agent.find_optimal_action_ind(rotation_angle=0)
                optimal_action_post = agent.find_optimal_action_ind()
                plt.axvline(x=optimal_action_pre + .5, color='gray', linestyle='--', lw=.5)
                plt.axvline(x=optimal_action_post + .5, color='black', linestyle='--', lw=.5)

                plt.title('Action Distribution')
                plt.tight_layout()
                if (episode == 10000) or (episode == int(cfg['max_episodes'] - cfg['log_interval'])):
                    plt.tight_layout()
                    if cfg['save_figs_locally']:
                        plt.savefig(os.path.join(cfg['fig_dir'], 'adaptation_action_hist_ep_{}.pdf'.format(episode)))

                if episode > 0:
                    # save out action histogram for later
                    action_hist_fn = 'adaptation_action_hist_seed_{}_target_{}_ep_{}.csv'.format(
                        cfg['seed'], np.round(np.degrees(cfg['reach_angle'])).astype(int), episode)
                    pd.DataFrame({'action_inds': bins[:-1], 'frequency': n}).to_csv(
                        op.join(paper_fig_dir, action_hist_fn))

                wandb.log({"action_hist": wandb.Image(fig), 'episode': episode})
                log_weights_to_wandb(agent.f)
                if episode in episodes_to_save and cfg['save_figs_locally']:
                    save_f_output = True
                else:
                    save_f_output = False
                plot_f_output(agent, agent.env, mean_embedding_last_trial, std_embedding, embedding_history[-1], cmap,
                              episode,
                              cfg['fig_dir'], save=save_f_output)
                plt.close('all')

                # also log generalization plot, if base agent is not None
                if base_agent is not None:
                    generalization_stats = calculate_generalization(agent, base_agent, cfg)
                    generalization_fig, angular_error_fig = make_generalization_plot(generalization_stats,
                                                                                     only_paper_angs=True)
                    wandb.log({"generalization_plot": wandb.Image(generalization_fig),
                               "angular_error_plot": wandb.Image(angular_error_fig),
                               "episode": episode})
                    plt.close(generalization_fig)

        if num_episodes_below_criterion >= cfg['early_stopping_criterion']:
            break

    fig, axs = plt.subplots(1, 1, figsize=(1.5, 1.5))
    plot_loss_rolling(nll_loss_history, axs, color='#A94850', label='embedding loss', window_size=1000)
    # plot_loss_rolling(actor_loss_history, axs, color='#44123F', label='actor loss')
    axs.set_xlabel('Episode')
    axs.set_ylabel('loss')
    axs.spines['right'].set_visible(False)
    axs.spines['top'].set_visible(False)

    plt.legend(frameon=False, fontsize=7)
    plt.tight_layout()
    if cfg['save_figs_locally']:
        plt.savefig(os.path.join(cfg['fig_dir'], 'adaptation_embedding_loss_seed_{}_target_{}.pdf'.format(
            cfg['seed'], np.degrees(cfg['reach_angle'])
        )))
    plt.close('all')

    fig, axs = plt.subplots(1, 1, figsize=(1.5, 1.5))
    plot_loss_rolling(reward_history, axs, color='#A94850', label='reward', window_size=1000)
    # plot_loss_rolling(actor_loss_history, axs, color='#44123F', label='actor loss')
    axs.set_xlabel('Episode')
    axs.set_ylabel('Reward')
    axs.spines['right'].set_visible(False)
    axs.spines['top'].set_visible(False)

    plt.legend(frameon=False, fontsize=7)
    plt.tight_layout()
    if cfg['save_figs_locally']:
        plt.savefig(os.path.join(cfg['fig_dir'], 'adaptation_reward_seed_{}_target_{}.pdf'.format(
            cfg['seed'], np.degrees(cfg['reach_angle'])
        )))
    plt.close('all')

    action_hist_for_loss = [act for ep in angle_history for act in ep]

    fig2, axs2 = plt.subplots(1, 1, figsize=(1.5, 1.5))
    plot_angle_diffs_rolling(angle_err_history, axs2, color='#44123F', label='error', window_size=cfg['max_episodes']//10)
    axs2.set_xlabel('Episode')
    axs2.set_ylabel('Angular error (degrees)')
    axs2.spines['right'].set_visible(False)
    axs2.spines['top'].set_visible(False)
    plt.legend(frameon=False, fontsize=7)
    plt.tight_layout()

    run_log = pd.DataFrame({
        'episode': np.arange(len(reward_history)),
        'reward': reward_history,
        'critic_loss': critic_loss_history,
        'actor_loss': actor_loss_history,
        'nll_loss': nll_loss_history,
        'angle_diff': angle_err_history,
        'action_taken': action_hist_for_loss
    })
    run_log_fn = 'adaptation_run_log_seed_{}_target_{}.csv'.format(cfg['seed'],
                                                                   np.round(np.degrees(cfg['reach_angle'])).astype(int))
    run_log.to_csv(op.join(paper_fig_dir, run_log_fn))

    if cfg['log_to_wandb']:
        wandb.log({"angular_error_over_time_plot": wandb.Image(fig2)})
    if cfg['save_figs_locally']:
        fig2.savefig(os.path.join(cfg['fig_dir'], 'angular_error.pdf'))
    if cfg["save_model"]:
        # save the final model after training
        outdir = cfg['post_adaptation_model_save_dir']
        if not os.path.exists(outdir):
            os.makedirs(outdir)
        torch.save({
            'g_state_dict': agent.g.state_dict(),
            'f_state_dict': agent.f.state_dict(),
            'actor_state_dict': agent.actor.state_dict(),
            'critic_state_dict': agent.critic.state_dict(),
            'params': {
                'state_dim': agent.env.n_features,
                'embedding_dim': 2,  # hard coded for now but can get out of f or g
                'n_actions': agent.env.n_actions
            }
        }, os.path.join(outdir,
                        'post_adaptation_model_seed_{}_rotation_{}_temp_{}_weight_decay_0.0001_tanh_policy_mean_n_actions_{}_target_{}.pth'.format(
                            cfg['seed'],
                            cfg['rotation_angle'], cfg['inv_temp'], cfg['num_actions'],
                            np.degrees(cfg['reach_angle']))))
    plt.close('all')

    return {'agent': agent,
            'action_history': action_history,
            'state_history': state_history,
            'critic_loss_history': critic_loss_history,
            'reward_history': reward_history,
            'embedding_history': embedding_history,
            'angle_history': angle_history,
            'predicted_action_history': predicted_action_history,
            'target_angle': cfg['reach_angle'],
            'rotation_angle': cfg['rotation_angle']}


def run_and_log_adaptation_experiment(cfg):

    seed = cfg['seed']
    reach_angle_degrees = np.round(np.degrees(cfg['reach_angle']).item()).astype(int)
    rotation_angle = cfg['rotation_angle']

    env = ReachTask(cfg, adaptation_rotation=np.radians(rotation_angle).item())
    agent = load_trained_full_model_basetask(cfg, env, reach_angle_degrees, seed)
    base_agent = copy.deepcopy(agent)  # keep copy of the agent before the adaptation experiment to measure adaptation

    if cfg['log_to_wandb']:
        wandb.init(
            project='ActionEmbeddingAdaptationExperiment',
            name=f'adapt_seed_{seed}_target_{reach_angle_degrees}_rotation_{rotation_angle}',
            config=cfg
        )
    print("Starting adaptation experiment with config:", cfg)
    adaptation_results = train_adapatation_experiment(cfg, agent, base_agent)

    generalization_stats = calculate_generalization(adaptation_results['agent'], base_agent, cfg)
    df_angles = generalization_stats[['rotation generalization', 'angle from target']].set_index('angle from target')
    generalization_score_local = df_angles[(df_angles.index >= -45) & (df_angles.index <= 45)][
        'rotation generalization'].mean()
    generalization_score_global = df_angles[(df_angles.index < -45) | (df_angles.index > 45)][
        'rotation generalization'].mean()
    generalization_fig, angular_error_fig = make_generalization_plot(generalization_stats)

    # save generalization stats
    stats_fn = 'generalization_stats_seed_{}_target_{}_rotation_{}.csv'.format(cfg['seed'], reach_angle_degrees,
                                                                               cfg['rotation_angle'])
    generalization_stats.to_csv(op.join(paper_fig_dir, stats_fn))

    if cfg['log_to_wandb']:
        wandb.log({"final_generalization_plot": wandb.Image(generalization_fig)})
        wandb.log({"angular_error_plot": wandb.Image(angular_error_fig)})
        wandb.log({"generalization_local": generalization_score_local,
                   "generalization_global": generalization_score_global,
                   "generalization_diff": generalization_score_local - generalization_score_global})
        wandb.finish()  # End the current run


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=9, help='Random seed for the experiment')
    parser.add_argument('--reach_angle', type=float, default=135., help='Reach angle in degrees')
    args = parser.parse_args()
    config['seed'] = args.seed
    config['reach_angle'] = np.radians(args.reach_angle).item()

    run_and_log_adaptation_experiment(config)

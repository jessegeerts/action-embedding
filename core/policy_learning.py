import matplotlib.pyplot as plt
import torch
import random
import math
import numpy as np
import os
from core.agent import ACLearningAgentWithEmbedding
from core.plotting import find_angle_difference, plot_loss_rolling, set_plotting_defaults, plot_embeddings_one_cue, plot_angle_diffs_rolling
from core.continuous_env import ReachTask
import wandb
from pathlib import Path
import argparse
from definitions import paper_model_path, paper_fig_dir


def reward_based_decay(reward, reward_min, reward_max, std_max, std_min):
    """
    Linearly decays std from std_max to std_min across a known reward range.

    :param reward: Current reward (can be between reward_min and reward_max)
    :param reward_min: Minimum possible reward (e.g. -0.1)
    :param reward_max: Maximum possible reward
    :param std_max: Std when reward == reward_min
    :param std_min: Std when reward == reward_max
    :return: Decayed std
    """
    # Clamp reward to be within [reward_min, reward_max]
    reward = max(min(reward, reward_max), reward_min)
    decay_ratio = (reward - reward_min) / (reward_max - reward_min)
    return std_max - decay_ratio * (std_max - std_min)


def train_agent(config):

    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, help='Set random seed', default=None)
    parser.add_argument('--target', type=float, help='Set reach angle (target, radians)', default=config['reach_angle'])
    args, _ = parser.parse_known_args()

    # set random seed
    seed = args.seed if args.seed is not None else config["seed"]
    config['reach_angle'] = args.target

    print(f'Training for reach angle: {args.target} radians, {np.degrees(args.target)} degrees')

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    reach_angle_degrees = round(np.degrees(config['reach_angle']).item())

    if config['log_to_wandb']:
        wandb.init(
            project='ActionEmbeddingContinuous-States-MultiTarget-debug',
            name=f'one_target_seed_{seed}_angle_{reach_angle_degrees}',
            config=config
        )

    set_plotting_defaults()

    # filename of pretrained encoder-decoder (f and g) for this seed
    fg_weights_fn = f'action_embedding_model_seed_{seed}_weight_decay_fg_0.0001_n_action_24_fourier_basis.pth'
    fg_load_path = Path(config['fg_load_path'])
    if not fg_load_path.exists():
        raise FileNotFoundError(f"FG weights file not found: {fg_load_path}. Please train and save the FG model first.")

    # create the environment
    env = ReachTask(config)
    # create the agent with fixed pre-trained f and g
    agent = ACLearningAgentWithEmbedding(env, config, fg_load_path=config['fg_load_path'],
                                         f_plastic=False, g_plastic=False)

    trained_angle_ind = np.where(np.round(agent.env.actions, 3) == np.round(config['reach_angle'], 3))[0][0]

    episodes = config["max_episodes"]
    max_steps = config["max_steps"]
    avg_reward = -0.1
    reward_avg_window = 100

    action_history = []
    angle_history = []

    critic_loss_history = []
    actor_loss_history = []
    nll_loss_history = []
    reward_history = []
    state_history, steps_history = [], []
    cumulative_reward_across_episodes = 0
    embedding_history = []
    policy_mean_history = []
    angle_diffs_history_for_wandb = []

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
        episode_states = [env.start_xy]
        episode_reward = 0
        if episode >= reward_avg_window:
            avg_reward = np.mean(reward_history[-reward_avg_window:])

        # Anneal policy std and reward window based on average reward
        internal_policy_std = reward_based_decay(avg_reward, -0.1, config['max_reward_policy_annealing'],
                                                       config['policy_std_max'], config['policy_std'])
        agent.set_policy_std(internal_policy_std)

        new_target_radius = reward_based_decay(avg_reward, -0.1, config['max_reward_target_annealing'],
                                               config['reward_radius_max'], config['reward_radius_min'])
        env.set_target_radius(new_target_radius)

        random_policy = config['use_random_policy']
        while steps < max_steps:
            # Zero the gradients for all modules
            agent.actor_optimizer.zero_grad()
            agent.critic_optimizer.zero_grad()
            #agent.f_g_optimizer.zero_grad()

            action_ind, embedding, mean_emb, logstd_emb = agent.select_action(features, random_policy=random_policy, noise=True)
            action = agent.env.actions[action_ind]
            next_state, reward, done = agent.env.act(action)
            next_state_features = agent.env.get_features(next_state)

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
            episode_reward += reward
            cumulative_reward_across_episodes += reward
            embedding_history.append(embedding.detach())
            steps += 1
        policy_mean_history.append(mean_emb.detach().numpy()) #this will only work for 1 step task
        reward_history.append(episode_reward)
        # calculate average losses for the episode
        avg_critic_loss_ep = total_critic_loss / steps if steps > 0 else 0
        avg_actor_loss_ep = total_actor_loss / steps if steps > 0 else 0
        avg_emb_loss_ep = total_nll_loss / steps if steps > 0 else 0

        action_history.append(episode_actions)
        angle_history.append(episode_angles)
        state_history.append(episode_states)
        critic_loss_history.append(avg_critic_loss_ep)
        actor_loss_history.append(avg_actor_loss_ep)
        nll_loss_history.append(avg_emb_loss_ep)
        steps_history.append(steps)

        distance = find_angle_difference(agent.env, agent.env.actions[action_ind])
        angle_diffs_history_for_wandb.append(distance)

        if episode % config['log_interval'] == 0:
            print(f'Episode {episode}, Steps Taken: {np.mean(steps_history[-100:])}')
            print(f'Actor loss: {np.mean(actor_loss_history[-100:])}, Critic loss: {np.mean(critic_loss_history[-100:])}, Supervised loss: {np.mean(nll_loss_history[-100:])}')

            if config["log_to_wandb"]:
                wandb.log({
                    "critic_loss": np.mean(critic_loss_history[-config['log_interval']:]),
                    "actor_loss": np.mean(actor_loss_history[-config['log_interval']:]),
                    "embedding_loss": np.mean(nll_loss_history[-config['log_interval']:]),
                    "episode": episode,
                    "n_steps": steps,
                    "cum_reward": cumulative_reward_across_episodes,
                    "episode_reward": episode_reward,
                    "avg_reward": np.mean(reward_history[-config['log_interval']:]),
                    "angular_error": np.mean(angle_diffs_history_for_wandb[-config['log_interval']:]),
                    "policy_std": agent.internal_policy_std,
                    "reward_window": env.target_radius
                })

            # Generate embedding plot
            embeddings = agent.get_action_embeddings_via_g()
            mean_embedding_last_trial = mean_emb.detach().numpy()
            std_embedding = np.exp(logstd_emb.detach().numpy())
            colors = plt.cm.twilight(np.linspace(0, 1, config['num_actions']))
            embedding_fig, embedding_ax = plot_embeddings_one_cue(embeddings, mean_embedding_last_trial, std_embedding, agent.env, embedding_history, colors[trained_angle_ind])

            if config["log_to_wandb"]:
                wandb.log({"embedding_plot": wandb.Image(embedding_fig)})

            plt.tight_layout()
            if config['save_figs']:
                plt.savefig(os.path.join(config['fig_dir'],
                                         f'embeddings_ep_{episode}_target_{reach_angle_degrees}_seed_{seed}.png'))
            plt.close(embedding_fig)

            # Plot histogram of actions taken
            local_action_history = action_history[-config['log_interval']:]
            local_action_history = [act for ep in local_action_history for act in ep]
            action_hist = np.array(local_action_history).flatten()
            fig, axs = plt.subplots(1,1, figsize=(1.2, 1.2))
            bins = np.arange(0, config['num_actions'] + 1)
            n, bins, patches = plt.hist(action_hist, bins=bins, density=True)

            for patch, color in zip(patches, colors):
                patch.set_facecolor(color)
            axs.set_xlabel('Action Index')
            axs.set_ylabel('Frequency')
            axs.axvline(trained_angle_ind +.5, color='k', linestyle='--')
            axs.spines['right'].set_visible(False)
            axs.spines['top'].set_visible(False)

            # Always save action histogram data
            np.save(os.path.join(paper_fig_dir,
                                 f'action_hist_episode_{episode}_target_{reach_angle_degrees}_seed_{seed}.npy'),
                    action_hist)

            if (episode == 0) or (episode == config['max_episodes'] - 1):
                plt.tight_layout()
                if config['save_figs']:
                    plt.savefig(os.path.join(config['fig_dir'],
                                             f'action_hist_{episode}_target_{reach_angle_degrees}_seed_{seed}.pdf'))
            if config["log_to_wandb"]:
                wandb.log({"action_hist": wandb.Image(fig), 'episode': episode})
            plt.close()

    # save critic loss history for later analysis
    np.save(os.path.join(paper_fig_dir, f'critic_loss_target_{reach_angle_degrees}_seed_{seed}.npy'), np.array(critic_loss_history))

    fig, axs = plt.subplots(1,1, figsize=(1.5, 1.5))
    plot_loss_rolling(critic_loss_history, axs, color='#A94850', label='critic loss')
    #plot_loss_rolling(actor_loss_history, axs, color='#44123F', label='actor loss')
    axs.set_xlabel('Episode')
    axs.set_ylabel('loss')
    axs.spines['right'].set_visible(False)
    axs.spines['top'].set_visible(False)

    plt.legend(frameon=False, fontsize=7)
    plt.tight_layout()
    if config['save_figs']:
        plt.savefig(os.path.join(config['fig_dir'], f'loss_target_{reach_angle_degrees}_seed_{seed}.pdf'))
    plt.close()
    action_hist_for_loss = [act for ep in angle_history for act in ep]
    distances = [find_angle_difference(agent.env, action) for action in action_hist_for_loss] #circular_distance_from_target(action_hist, target=15)
    # save angle differenes for later analysis
    np.save(os.path.join(paper_fig_dir, f'angle_differences_target_{reach_angle_degrees}_seed_{seed}.npy'), np.array(distances))
    fig2, axs2 = plt.subplots(1,1, figsize=(1.5, 1.5))
    plot_angle_diffs_rolling(distances, axs2, color='#44123F', label='error')
    axs2.set_xlabel('Episode')
    axs2.set_ylabel('Angular error (degrees)')
    axs2.spines['right'].set_visible(False)
    axs2.spines['top'].set_visible(False)
    plt.legend(frameon=False, fontsize=7)
    plt.tight_layout()
    if config['save_figs']:
        fig2.savefig(os.path.join(config['fig_dir'], f'angular_error_target_{reach_angle_degrees}_seed_{seed}.pdf'))
    plt.close(fig2)
    if config["save_model"]:
        # save the final model after training
        outdir = config['save_model_dir']
        n_acts = config['num_actions']
        model_weights_fn = f'fully_trained_policy_model_one_target_seed_{seed}_weight_decay_0.0001_tanh_policy_mean_target_{reach_angle_degrees}_n_actions_{n_acts}.pth'
        torch.save({
            'g_state_dict': agent.g.state_dict(),
            'f_state_dict': agent.f.state_dict(),
            'actor_state_dict': agent.actor.state_dict(),
            'critic_state_dict': agent.critic.state_dict(),
            'params': {
                'state_dim': env.n_features,
                'embedding_dim': 2, #hard coded for now but can get out of f or g
                'n_actions': env.n_actions
            }
        }, os.path.join(outdir, model_weights_fn))

    if config['log_to_wandb']:
        wandb.finish()

    return {
        'agent': agent,
        'action_history': action_history,
        'state_history': state_history,
        'critic_loss_history': critic_loss_history,
        'reward_history': reward_history,
        'embedding_history': embedding_history,
        'target_angle': config['reach_angle'],
        'centre_coord': env.start_xy,
        'trained_angle_ind': trained_angle_ind,
        'policy_mean_history': policy_mean_history
    }


if __name__ == '__main__':
    from core.config import config

    if config['run_for_all_reach_angles']:
        num_actions = config['num_actions']
        all_reach_angles = np.linspace(0, 2 * np.pi, num_actions, endpoint=False)
    else:
        all_reach_angles = [config['reach_angle']]

    for reach_angle in all_reach_angles:
        config['reach_angle'] = reach_angle
        # set up some variables for training
        train_results = train_agent(config)

import sys
from pathlib import Path

# Add project root to path so imports work from scripts/ directory
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import numpy.random
import os
import pickle
import torch
import numpy as np
from core.agent import ActionEmbeddingPredictor, ActionMapping
from core.plotting import plot_embeddings_only, plot_embeddings_only_half_step_compatible, \
    set_plotting_defaults, plot_embeddings_only_multi_step_compatible, plot_pca_features
import argparse
import matplotlib.pyplot as plt
from torch import nn
from definitions import paper_model_path, paper_fig_dir
from core.continuous_env import ReachTask
import seaborn as sns
import math


def compute_angular_error(predictions, targets, n_actions):
    """
    Compute angular error in degrees.

    Args:
        predictions: predicted action indices
        targets: true action indices
        n_actions: total number of actions

    Returns:
        Mean angular error in degrees
    """
    errors = []
    for pred, true in zip(predictions, targets):
        # Circular distance (shortest path around the circle)
        diff = min(abs(pred - true), n_actions - abs(pred - true))
        # Convert to degrees
        error_deg = diff * (360 / n_actions)
        errors.append(error_deg)
    return np.mean(errors)


def magnitude_regularizer(embeddings):
    """Encourage embeddings to be away from origin"""
    magnitudes = torch.norm(embeddings)
    return -torch.mean(magnitudes)  # Negative because we want to maximize distance from origin


def get_action_embeddings_via_g(env, g):
    """
    Get embeddings by simulating moving and state transitions pass through g
    """
    env.reset()
    centre_coord = env.current_xy
    embeddings = []
    for a in range(env.n_actions):
        env.current_xy = centre_coord  # Start from a fixed state, or choose a sample state
        features = env.get_features(centre_coord)
        next_state, reward, done = env.act(env.actions[a])
        next_state_features = env.get_features(next_state)

        embedding = g(features, next_state_features).detach().numpy()
        embeddings.append(embedding)
    return np.array(embeddings).squeeze()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, help='Set random seed', default=1)
    parser.add_argument('--learning_rate', type=float, help='Learning rate for the model', default=0.01)
    parser.add_argument('--xavier_scale', type=float, help='Scale for Xavier initialization', default=.1)
    parser.add_argument('--reg_factor', type=float, help='Regularization factor for magnitude loss', default=0.01)
    parser.add_argument('--num_actions', type=int, help='Number of actions in the environment', default=24)
    parser.add_argument('--lecun_scale', type=float, help='Scale for LeCun initialization', default=1.0)
    parser.add_argument('--weight_decay', type=float, help='Weight decay for optimizer', default=0.0001)
    parser.add_argument('--softmax_temp', type=float, help='Softmax temperature', default=.2)
    parser.add_argument('--log_to_wandb', action='store_true', help='Enable Weights & Biases logging')
    args = parser.parse_args()

    set_plotting_defaults()
    outdir = paper_model_path
    config = {
        'seed': args.seed, #35 is a good seed, 1 is a bad seed
        'learning_rate': args.learning_rate,
        'grid_size': (20, 20),
        'num_actions': args.num_actions,
        "reward_radius_max": 0.5,
        "reward_radius_min": 0.2,
        "reward_for_hit": 1.0,
        "penalty_for_miss": -0.1,
        "reach_angle": math.radians(135),
        'n_episodes': 300000,
        'loss_threshold': .01,
        'steps_threshold': 100,
        'fourier_order': 3,
        'log_interval': 10000,
        "w_decay_fg": 0,
        "softmax_temp": args.softmax_temp,
        "drop_p_fg": 0,
        "reg_factor": args.reg_factor,
        "xavier_init": False,
        "xavier_scale": args.xavier_scale,
        "lecun_init": True,
        "lecun_scale": args.lecun_scale,
        "orthogonal_init": True,
        "max_steps": 1,
        "weight_decay": args.weight_decay,
        "save_model": True,
        "save_figures": True,
        "do_nothing_action": False,
        "half_step_actions": False,
        "multi_step_actions": False,
        "tanh_temp": 1,
        "reach_length": 1,
        "log_to_wandb": args.log_to_wandb,
    }
    # set seed to arg
    print('Seed set to {}'.format(config['seed']))
    fig_dir = paper_fig_dir

    # Initialize wandb if enabled
    if config['log_to_wandb']:
        import wandb
        wandb.init(
            project='ActionEmbeddingLearningPaper',
            name='LearnEmbedding_seed_{}_nacts_{}'.format(config['seed'], config['num_actions']),
            config=config
        )

    torch.manual_seed(config['seed'])
    numpy.random.seed(config['seed'])

    n_episodes = config['n_episodes']
    loss_threshold = config['loss_threshold']
    steps_threshold = config['steps_threshold']
    log_interval = config['log_interval']
    learning_rate = config['learning_rate']

    # first we initialize an environment so we can get the state representation
    env = ReachTask(config)

    state_dim = env.n_features
    embedding_dim = 2
    max_steps = config['max_steps']
    g = ActionEmbeddingPredictor(state_dim, embedding_dim, p_drop=config['drop_p_fg'],
                                 xavier=config['xavier_init'], xavier_scale=config['xavier_scale'],
                                 tanh_temp=config['tanh_temp'], lecun_init=config['lecun_init'],
                                 lecun_scale=config['lecun_scale'])

    f = ActionMapping(embedding_dim, env.n_actions)

    # plot initial weights
    initial_weights = g.linear.weight.detach().numpy()
    weights_plot = sns.jointplot(initial_weights.T)
    loss_fn = torch.nn.NLLLoss()
    log_softmax = torch.nn.LogSoftmax(dim=0)

    optimizerg = torch.optim.AdamW(list(g.parameters()), lr=learning_rate, weight_decay=config['weight_decay'], betas=(0.95, 0.999))
    optimizerf = torch.optim.AdamW(list(f.parameters()), lr=learning_rate, weight_decay=config['weight_decay'], betas=(0.95, 0.999))
    actions = env.actions
    action_id = env.action_idx

    action_rng = np.random.RandomState(42)  # Fixed seed for action sampling

    embeddings = get_action_embeddings_via_g(env, g)
    print("Initial embeddings before training:")
    for i, emb in enumerate(embeddings):
        print(f"  Action {i}: [{emb[0]:.4f}, {emb[1]:.4f}]")

    env.reset()
    centre = np.array(env.current_xy)
    centre_features = env.get_features(centre)

    all_features = []
    all_next_features = []
    deltas = []

    for i, action in enumerate(env.actions):
        env.current_xy = centre.copy()
        next_xy, _, _ = env.act(action)
        next_features = env.get_features(next_xy)

        all_features.append(centre_features.numpy())
        all_next_features.append(next_features.numpy())
        deltas.append(next_features.numpy() - centre_features.numpy())

        print(f"Action {i}: delta_xy = {next_xy - centre}")

    deltas = np.array(deltas)
    print(f"\nFeature delta shape: {deltas.shape}")
    print(f"Feature delta norms: {np.linalg.norm(deltas, axis=1)}")

    # Check pairwise distances between feature deltas
    from scipy.spatial.distance import pdist, squareform
    pairwise = squareform(pdist(deltas))
    print(f"\nMin pairwise distance between feature deltas: {pairwise[pairwise > 0].min():.6f}")
    print(f"Mean pairwise distance: {pairwise[pairwise > 0].mean():.6f}")

    # we can now train the model (only the supervised learning of action predictions from state + next state
    losses = []
    accuracies = []
    running_losses = []
    angular_errors = []
    running_correct = []
    running_entropy = []
    running_confidence = []
    running_predictions = []
    running_targets = []
    steps_above_criterion = 0
    for ep in range(n_episodes):
        steps = 0
        optimizerf.zero_grad()
        optimizerg.zero_grad()
        env.reset() # torch.normal(torch.zeros(2), 0.2)
        episode_losses = []
        episode_correct = []
        episode_entropy = []
        episode_confidence = []
        while steps < max_steps:
            features = env.get_features(env.current_xy)
            idx = action_rng.randint(env.n_actions)

            action = env.actions[idx]
            next_xy, reward, done = env.act(action)
            next_features = env.get_features(next_xy)
            embedding = g(features, next_features)
            action_pred = f(embedding)

            # calculate prediction loss
            loss = loss_fn(torch.log_softmax(action_pred / config['softmax_temp'], dim=0).unsqueeze(0),
                           torch.tensor(action_id[action]).unsqueeze(0))

            predicted_action = torch.argmax(action_pred).item()
            correct = predicted_action == action_id[action]

            probs = torch.softmax(action_pred, dim=0)  # Convert logits to probabilities
            entropy = -torch.sum(probs * torch.log(probs + 1e-10)).item()
            confidence = torch.max(probs).item()

            loss.backward()

            # Add this right after your first backward() call in the training loop
            if ep == 0 and steps == 0:  # First step of first episode
                first_step_gradients = {}
                for name, param in g.named_parameters():
                    if param.grad is not None:
                        first_step_gradients[name] = param.grad.clone().detach().numpy()

                # Log key gradient statistics
                weight_grad = first_step_gradients['linear.weight']  # Shape: (2, state_dim)

                grad_stats = {
                    'grad_norm_dim0': np.linalg.norm(weight_grad[0]),
                    'grad_norm_dim1': np.linalg.norm(weight_grad[1]),
                    'grad_correlation': np.corrcoef(weight_grad[0], weight_grad[1])[0, 1],
                    'grad_mean_dim0': np.mean(weight_grad[0]),
                    'grad_mean_dim1': np.mean(weight_grad[1]),
                    'grad_std_dim0': np.std(weight_grad[0]),
                    'grad_std_dim1': np.std(weight_grad[1]),
                    'total_grad_norm': np.linalg.norm(weight_grad)
                }

                if config['log_to_wandb']:
                    import wandb
                    wandb.log(grad_stats)

            optimizerf.step()
            optimizerg.step()

            episode_losses.append(loss.item())
            episode_correct.append(correct)
            episode_entropy.append(entropy)
            episode_confidence.append(confidence)
            running_predictions.append(predicted_action)
            running_targets.append(idx)

            if loss < loss_threshold and steps_above_criterion > steps_threshold:
                break
            elif loss < loss_threshold:
                steps_above_criterion += 1
            else:
                steps_above_criterion = 0
            steps += 1
        running_losses.extend(episode_losses)
        running_correct.extend(episode_correct)
        running_entropy.extend(episode_entropy)
        running_confidence.extend(episode_confidence)
        if ep % log_interval == 0:
            avg_loss = np.mean(running_losses)
            accuracy = np.mean(running_correct)
            avg_entropy = np.mean(running_entropy)
            avg_confidence = np.mean(running_confidence)
            avg_angular_error = compute_angular_error(running_predictions, running_targets, env.n_actions)
            print(f'Iteration {ep}, Loss: {avg_loss}, accuracy: {accuracy}, angular_error: {avg_angular_error:.2f}°')

            running_correct = []
            running_losses = []
            running_entropy = []
            running_confidence = []
            running_predictions = []
            running_targets = []

            losses.append(avg_loss)
            accuracies.append(accuracy)
            angular_errors.append(avg_angular_error)

            if config['log_to_wandb']:
                import wandb
                wandb.log({'loss': avg_loss, 'accuracy': accuracy, 'episode': ep,
                           'entropy': avg_entropy, 'confidence': avg_confidence,
                           'angular_error': avg_angular_error})

                # look at the action embeddings
                embeddings = get_action_embeddings_via_g(env, g)
                fig = plot_embeddings_only(embeddings, env)
                wandb.log({'action_embeddings': wandb.Image(fig), 'episode': ep})

    # After training, run this
    from collections import defaultdict
    confusion = defaultdict(lambda: defaultdict(int))

    env.reset()
    for _ in range(1000):
        idx = np.random.randint(env.n_actions)
        action = env.actions[idx]

        features = env.get_features(env.current_xy)
        next_xy, _, _ = env.act(action)
        next_features = env.get_features(next_xy)

        embedding = g(features, next_features)
        pred = torch.argmax(f(embedding)).item()

        confusion[idx][pred] += 1
        env.reset()

    # Print which actions get confused
    for true_action in range(env.n_actions):
        preds = confusion[true_action]
        if preds[true_action] < 900:  # less than 90% correct
            print(f"Action {true_action}: {dict(preds)}")

    fig, axs = plt.subplots(1,1, figsize=(1.2, 1.2))
    axs.plot(losses, color='#6067B6', label='embedding loss')
    axs.set_xlabel('Episode')
    axs.set_ylabel('loss')
    axs.spines['right'].set_visible(False)
    axs.spines['top'].set_visible(False)

    plt.tight_layout()
    if config["save_figures"]:
        plt.savefig(os.path.join(fig_dir, 'embedding_loss.pdf'))

    embeddings = get_action_embeddings_via_g(env, g)
    print("Embeddings:")
    for i, emb in enumerate(embeddings):
        print(f"  Action {i}: [{emb[0]:.4f}, {emb[1]:.4f}]")

    plt.close()
    # save the final model after training
    if config["save_model"]:
        # Build optional suffixes
        suffixes = []
        if config.get("half_step_actions", False):
            suffixes.append("including_half_steps")
        if config.get("do_nothing_action", False):
            suffixes.append("including_do_nothing_action")
        suffix = ("_" + "_".join(suffixes)) if suffixes else "_fourier_basis"

        # Filename
        filename = (
            f"action_embedding_model_seed_{config['seed']}"
            f"_weight_decay_fg_{config['weight_decay']}"
            f"_n_action_{env.n_actions}{suffix}.pth"
        )

        # Save
        torch.save({
            "g_state_dict": g.state_dict(),
            "f_state_dict": f.state_dict(),
            "params": {
                "state_dim": state_dim,
                "embedding_dim": embedding_dim,
                "n_actions": env.n_actions,
                "half_step": bool(config.get("half_step_actions", False)),
                "do_nothing": bool(config.get("do_nothing_action", False)),
            }
        }, os.path.join(outdir, filename))

        # Save loss and episode data
        loss_data = {
            'episodes': list(range(0, len(losses) * log_interval, log_interval)),
            'losses': losses,
            'accuracies': accuracies,
            'angular_errors': angular_errors,
            'seed': config['seed'],
            'config': config
        }

        # Save to pickle file
        loss_filename = f"loss_data_seed_{config['seed']}.pkl"
        with open(os.path.join(outdir, loss_filename), 'wb') as f:
            pickle.dump(loss_data, f)

        # Also save as numpy arrays for easy loading
        np.save(os.path.join(outdir, f'losses_seed_{config["seed"]}.npy'), np.array(losses))
        np.save(os.path.join(outdir, f'episodes_seed_{config["seed"]}.npy'),
                np.array(list(range(0, len(losses) * log_interval, log_interval))))

    if config['log_to_wandb']:
        import wandb
        wandb.finish()

if __name__ == '__main__':
    main()

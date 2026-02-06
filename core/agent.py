import numpy as np
import torch
import math
from torch import nn as nn, optim as optim
from torch.nn import functional as F
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from core.utils import compute_circular_mean


def softmax_with_temperature(logits, temperature):
    """
    Applies softmax to the input logits with varying temperature.

    Args:
    - logits (torch.Tensor): The input logits.
    - temperature (float): The temperature parameter (lower temperature -> sharper distribution).

    Returns:
    - torch.Tensor: The softmax probabilities.
    """
    if temperature <= 0:
        raise ValueError("Temperature must be positive.")

    # Apply softmax with temperature
    softmax_output = F.softmax(logits / temperature, dim=-1)
    return softmax_output


def add_noise_to_action_probs(action_probs, noise_level=1e-3):
    # Add noise to all actions
    noisy_probs = action_probs + noise_level
    # Normalize to ensure probabilities sum to 1
    noisy_probs /= noisy_probs.sum()
    return noisy_probs


class ActionEmbeddingPredictor(nn.Module):
    def __init__(self, state_dim, embedding_dim, p_drop=0.,
                 xavier=False, xavier_scale=1.0, lecun_init=False, lecun_scale=1., tanh_temp=1.0):
        super(ActionEmbeddingPredictor, self).__init__()

        self.linear = nn.Linear(2 * state_dim, embedding_dim)  # Takes current and next states as input
        self.dropout = nn.Dropout(p=p_drop)
        self.tanh_temp = tanh_temp

        nn.init.normal_(self.linear.weight, mean=0.0, std=1/math.sqrt(state_dim*2))
        nn.init.zeros_(self.linear.bias)

        if xavier:
            nn.init.xavier_uniform_(self.linear.weight, gain=xavier_scale)
            nn.init.zeros_(self.linear.bias)

        if lecun_init:
            fan_in = 2 * state_dim  # Input size
            std = np.sqrt(1.0 / fan_in) * lecun_scale # LeCun initialization
            nn.init.normal_(self.linear.weight, mean=0.0, std=std)
            nn.init.zeros_(self.linear.bias)

    def forward(self, state, next_state):
        state_transition = torch.cat([state, next_state], dim=-1)
        return torch.tanh(self.dropout(self.linear(state_transition/self.tanh_temp)))


class ActionEmbeddingPredictor_only_next_state_input(nn.Module):
    def __init__(self, state_dim, embedding_dim, p_drop=0.,
                 xavier=False, xavier_scale=1.0, lecun_init=False, tanh_temp=1.0):
        super(ActionEmbeddingPredictor_only_next_state_input, self).__init__()

        self.linear = nn.Linear(state_dim, embedding_dim)  # Takes current and next states as input
        self.dropout = nn.Dropout(p=p_drop)
        self.tanh_temp = tanh_temp

        nn.init.normal_(self.linear.weight, mean=0.0, std=1/math.sqrt(state_dim))
        nn.init.zeros_(self.linear.bias)

        if xavier:
            nn.init.xavier_uniform_(self.linear.weight, gain=xavier_scale)
            nn.init.zeros_(self.linear.bias)

        if lecun_init:
            fan_in = state_dim  # Input size
            std = np.sqrt(1.0 / fan_in)  # LeCun initialization
            nn.init.normal_(self.linear.weight, mean=0.0, std=std)
            nn.init.zeros_(self.linear.bias)

    def forward(self, next_state):
        return torch.tanh(self.dropout(self.linear(next_state/self.tanh_temp)))


class ActionEmbeddingPredictorNormed(nn.Module):
    #forces output onto hypersphere
    def __init__(self, state_dim, embedding_dim, p_drop=0., xavier=False):
        super(ActionEmbeddingPredictorNormed, self).__init__()
        self.linear = nn.Linear(2 * state_dim, embedding_dim)  # Takes current and next states as input
        self.dropout = nn.Dropout(p=p_drop)
        if xavier:
            nn.init.xavier_uniform_(self.linear.weight)
            nn.init.zeros_(self.linear.bias)
    def forward(self, state, next_state):
        state_transition = torch.cat([state, next_state], dim=-1)
        embedding = self.dropout(self.linear(state_transition))
        return torch.nn.functional.normalize(embedding, p=2, dim=-1)


class ActionEmbeddingPredictorSigmoid(nn.Module):
    def __init__(self, state_dim, embedding_dim, p_drop=0., xavier=False):
        super(ActionEmbeddingPredictorSigmoid, self).__init__()
        self.linear = nn.Linear(2 * state_dim, embedding_dim)  # Takes current and next states as input
        self.dropout = nn.Dropout(p=p_drop)
        if xavier:
            nn.init.xavier_uniform_(self.linear.weight)
            nn.init.zeros_(self.linear.bias)

    def forward(self, state, next_state):
        state_transition = torch.cat([state, next_state], dim=-1)
        return torch.sigmoid(self.dropout(self.linear(state_transition)))


class ActionMapping(nn.Module):
    def __init__(self, embedding_dim, action_dim):
        super(ActionMapping, self).__init__()
        self.linear = nn.Linear(embedding_dim, action_dim)
        self.prev_weights = None

    def forward(self, action_embedding):
        return self.linear(action_embedding)  # Maps embedding to action logits


class Actor(nn.Module):
    def __init__(self, state_dim, embedding_dim, xavier=False):
        super(Actor, self).__init__()
        self.mean_head = nn.Linear(state_dim, embedding_dim)
        nn.init.normal_(self.mean_head.weight, mean=0.0, std=1e-2)
        if xavier:
            nn.init.xavier_uniform_(self.mean_head.weight)
            nn.init.zeros_(self.mean_head.bias)
        # Bias to zero → ensures output is near zero initially
        nn.init.constant_(self.mean_head.bias, 0.0)

    def forward(self, state):
        # todo: check if we need activation function to make the mean between -1 and 1
        mean = self.mean_head(state)
        return torch.tanh(mean) # Output embedding
        # I think a learnable std might be necessary - it's struggling with exploration currently


class Critic(nn.Module):
    def __init__(self, state_dim, xavier=False):
        super(Critic, self).__init__()
        self.fc = nn.Linear(state_dim, 1)
        if xavier:
            nn.init.xavier_uniform_(self.fc.weight)
            nn.init.zeros_(self.fc.bias)

    def forward(self, state):
        return self.fc(state)  # Output state value


def plot_weight_heatmap(model, title):
    weights = [param.detach().cpu().numpy() for name, param in model.named_parameters() if "weight" in name]

    plt.figure(figsize=(10, 5))

    for i, weight_matrix in enumerate(weights):
        plt.subplot(1, len(weights), i + 1)
        plt.imshow(weight_matrix, cmap="coolwarm", aspect='auto')
        plt.colorbar()
        plt.title(f"{title} Layer {i + 1}")

    plt.show()


class ACLearningAgentWithEmbedding:
    def __init__(self, env, config, fg_load_path=None, full_model_load_path=None,
                 actor_plastic=True, critic_plastic=True, f_plastic=True, g_plastic=True):
        self.env = env
        self.gamma = config["discount_factor"]
        self.actor_lr = config["actor_lr"]
        self.critic_lr = config["critic_lr"]
        self.fg_lr = config["fg_lr"]
        self.target_entropy = 1.5
        self.grid_size_rows = config['grid_size'][0]
        self.grid_size_cols = config['grid_size'][1]
        self.softmax_inv_temp = config["inv_temp"]
        self.actor_plastic = actor_plastic
        self.critic_plastic = critic_plastic
        self.f_plastic = f_plastic
        self.g_plastic = g_plastic
        state_dim = self.env.n_features
        self.internal_policy_std = config["policy_std"]  # note: this is in embedding space (internal policy)
        self.policy_noise = config["policy_noise"]  # this is in action space (overall policy)
        self.embedding_dim = config['embedding_dim']
        # option to load a fully trained model
        if not full_model_load_path:
            # Actor-critic
            self.actor = Actor(state_dim, self.embedding_dim)
            self.critic = Critic(state_dim)
        else:
            checkpoint_fully_trained = torch.load(full_model_load_path)
            # Extract parameters
            state_dim = checkpoint_fully_trained['params']['state_dim']
            embedding_dim = checkpoint_fully_trained['params']['embedding_dim']
            n_actions = checkpoint_fully_trained['params']['n_actions']

            # Reinitialize models
            self.g = ActionEmbeddingPredictor(state_dim, self.embedding_dim)
            self.f = ActionMapping(self.embedding_dim, n_actions)

            # Load weights
            self.g.load_state_dict(checkpoint_fully_trained['g_state_dict'])
            self.f.load_state_dict(checkpoint_fully_trained['f_state_dict'])

            self.actor = Actor(state_dim, self.embedding_dim)
            self.critic = Critic(state_dim)
            self.actor.load_state_dict(checkpoint_fully_trained['actor_state_dict'])
            self.critic.load_state_dict(checkpoint_fully_trained['critic_state_dict'])

            # option to load embeddings
        if not fg_load_path:
            # Define the networks
            if not full_model_load_path:
                self.g = ActionEmbeddingPredictor(state_dim, self.embedding_dim)
                self.f = ActionMapping(self.embedding_dim, self.env.n_actions)
        else:
            checkpoint = torch.load(fg_load_path)

            # Extract parameters
            state_dim = checkpoint['params']['state_dim']
            embedding_dim = checkpoint['params']['embedding_dim']
            n_actions = checkpoint['params']['n_actions']

            # Reinitialize models
            self.g = ActionEmbeddingPredictor(state_dim, self.embedding_dim)
            self.f = ActionMapping(self.embedding_dim, n_actions)

            # Load weights
            self.g.load_state_dict(checkpoint['g_state_dict'])
            self.f.load_state_dict(checkpoint['f_state_dict'])

        if not actor_plastic:
            self.actor_lr = 0
        if not critic_plastic:
            self.critic_lr = 0
        # Optimizers for actor and critic with different learning rates
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=self.actor_lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=self.critic_lr)
        if f_plastic & g_plastic:
            # Optimizer and loss function for f and g (same learning rate for both)
            self.f_g_optimizer = optim.Adam(list(self.g.parameters()) + list(self.f.parameters()), lr=self.fg_lr,
                                            weight_decay=0.0001)
                                        #weight_decay=w_decay_fg)
        elif f_plastic:
            self.f_g_optimizer = optim.AdamW(list(self.f.parameters()), lr=self.fg_lr, weight_decay=0.0001)


        # Loss functions
        self.nll_loss_fn = nn.NLLLoss()

    def state_to_index(self, state_tensor):
        return torch.argmax(state_tensor).item()

    def update_temperature(self, performance_metric, min_temp=5, max_temp=20):
        """
        Update the inverse temperature based on performance.

        Args:
            performance_metric (float): Metric reflecting RL performance (e.g., recent reward).
        """
        self.softmax_inv_temp = min_temp + (max_temp - min_temp) * np.exp(-0.1 * performance_metric)

    def select_action(self, state_tensor, random_policy=False, import_policy_mean=False,
                      policy_mean=None, noise=True, random_embedding=False):
        # Get actor mean and std
        mean_emb = self.actor(state_tensor)
        std_emb = torch.tensor(self.internal_policy_std)

        # Random embedding from uniform distribution between -1 and 1
        if random_embedding:
            embedding = 2.0 * torch.rand(self.embedding_dim) - 1.0
        elif import_policy_mean:
            # Generate embedding from the imported mean and model's std
            embedding = torch.normal(policy_mean, std_emb)
            embedding = torch.tanh(embedding)
        else:
            # Generate embedding from the model’s learned distribution
            embedding = torch.normal(mean_emb, std_emb)
            embedding = torch.tanh(embedding)

        # Use `f` to map the embedding to action logits
        action_logits = self.f(embedding)
        if random_policy:
            temp = 1e12  # High temperature -> uniform distribution
        else:
            temp = self.softmax_inv_temp
        action_probs = softmax_with_temperature(action_logits, temperature=temp)
        if noise:
            action_probs = add_noise_to_action_probs(action_probs, noise_level=self.policy_noise)
        action_idx = torch.multinomial(action_probs, 1).item()
        return action_idx, embedding, mean_emb, torch.log(std_emb)

    def update(self, state_tensor, action_ind, embedding, next_state_tensor, reward, done):

        # Compute the value of the current state and the next state
        value = self.critic(state_tensor)
        mean = self.actor(state_tensor)
        next_value = self.critic(next_state_tensor).detach() if not done else torch.tensor(0.0)
        target = reward + self.gamma * next_value

        # Use TD error to calculate the critic loss (minimize TD error)
        advantage = target - value  # todo: change var name

        # Compute actor loss (policy gradient)
        logstd = torch.log(torch.tensor(self.internal_policy_std))
        log_prob = self.compute_log_prob_for_tanh_squash(embedding, logstd, mean)
        actor_loss = -log_prob * advantage.detach()  # Policy gradient loss using the TD error
        critic_loss = advantage.pow(2)

        # Backpropagate the actor and critic losses
        if self.actor_plastic:
            actor_loss.backward()
            self.actor_optimizer.step()
        if self.critic_plastic:
            critic_loss.backward()
            self.critic_optimizer.step()

        # Get current state embedding
        state_embedding = self.g(state_tensor, next_state_tensor)

        # Update `f` using NLL loss
        action_pred = self.f(state_embedding)
        log_action_probs = torch.log_softmax(action_pred/self.softmax_inv_temp, dim=0) #dividing by temp is new
        nll_loss = self.nll_loss_fn(log_action_probs.unsqueeze(0), torch.tensor([action_ind]))
        #initially predicts totally wrong action from embedding but then starts to predict the right ind, but it doesn't learn to take the right ind for reward
        # Update `f` and `g`

        nll_loss.backward()
        if self.f_plastic or self.g_plastic:
            self.f_g_optimizer.step()

        return nll_loss.item(), actor_loss.item(), critic_loss.item()


    def compute_log_prob(self, embedding, logstd, mean):
        # we need the log probability of the embedding for the internal policy gradient update
        std = torch.exp(logstd)
        variance = std * std
        log_prob = -0.5 * torch.log(2 * torch.tensor(torch.pi) * variance) - (embedding - mean) ** 2 / (2 * variance)
        log_prob = log_prob.sum(dim=-1)
        return log_prob

    def compute_log_prob_for_tanh_squash(self, action, log_std, mean):
        std = torch.exp(log_std)
        pre_tanh_action = 0.5 * (torch.log1p(action) - torch.log1p(-action))  # arctanh

        # Standard Gaussian log-prob
        var = std ** 2
        log_prob = -0.5 * ((pre_tanh_action - mean) ** 2 / var + 2 * log_std + np.log(2 * np.pi))
        log_prob = log_prob.sum(dim=-1)

        # Correction for tanh squashing
        log_det_jacobian = 2 * (np.log(2) - pre_tanh_action - F.softplus(-2 * pre_tanh_action))
        log_det_jacobian = log_det_jacobian.sum(dim=-1)

        return log_prob - log_det_jacobian

    def get_action_embeddings(self):
        """
        Get the embeddings directly from the `f` network's weights.
        """
        with torch.no_grad():
            # Extract the weights of the linear layer in `f`
            embeddings = self.f.linear.weight.detach().numpy()
        return embeddings

    def get_action_embeddings_via_g(self, start_xy=None, return_next_state=False):
        """
        Get embeddings by simulating moving and state transitions pass through g
        """
        if start_xy is None:
            start_xy = (self.grid_size_rows // 2, self.grid_size_cols // 2)
        embeddings = []
        next_states = []
        for a in range(self.env.n_actions):
            self.env.current_xy = start_xy  # Start from a fixed state, or choose a sample state
            features = self.env.get_features(self.env.current_xy)
            next_state, reward, done = self.env.act(self.env.actions[a])
            next_state_features = self.env.get_features(next_state)

            embedding = self.g(features, next_state_features).detach().numpy()
            embeddings.append(embedding)
            next_states.append(next_state)
        if return_next_state:
            return np.array(embeddings).squeeze(), next_states
        else:
            return np.array(embeddings).squeeze()

    @staticmethod
    def calculate_distance_from_target(x, y, target_x, target_y):
        distance = math.sqrt((x - target_x) ** 2 + (y - target_y) ** 2)
        return distance

    def find_optimal_action_ind(self, goal=None, rotation_angle=None):
        actions = self.env.actions
        starting_coord = np.array((self.env.env_shape[0] / 2, self.env.env_shape[1] / 2))
        if rotation_angle is None:
            rotation_angle = self.env.adaptation_rotation

        action_consequences = [(self.env.move_in_direction(starting_coord[0],
                                                      starting_coord[1], a,
                                                      adaptation_rotation=rotation_angle)) for a in
                               actions]
        if goal:
            goal_state = goal
        else:
            goal_state = self.env.target_xy
        distances_from_target = [(self.calculate_distance_from_target(consequence[0], consequence[1], goal_state[0],
                                                                 goal_state[1])) for consequence in action_consequences]
        optimal_action_ind = np.argmin(distances_from_target)
        return optimal_action_ind

    def plot_actions_task_space(self, ax, s=10):
        cmap = plt.get_cmap('twilight')

        # Right plot: Cartesian coordinates of actions
        actions = self.env.actions
        goal_coord = np.array(self.env.target_xy) - self.env.start_xy
        action_coords = np.array([(self.env.move_in_direction(*self.env.start_xy, a)) for a in actions]) - self.env.start_xy
        action_inds = np.arange(actions.shape[0])
        scatter = ax.scatter(*action_coords.T, c=action_inds, cmap=cmap, s=s,
                                label='Cartesian Coordinates')
        plt.draw()  # or fig.canvas.draw()
        colors_from_scatter = scatter.get_facecolors()

        optimal_action_ind = self.find_optimal_action_ind()
        color_for_action = colors_from_scatter[optimal_action_ind]

        ax.set_title('Cartesian Coordinates')
        ax.set_xlabel('Delta X')
        ax.set_ylabel('Delta Y')
        ax.scatter(*goal_coord, marker='*', color=color_for_action, s=s*5)
        return ax

    def plot_embeddings_and_pi_i(self, policy_color='k', cmap=None, ax=None, s=10):
        from matplotlib.patches import Circle
        if ax is None:
            fig, ax = plt.subplots(figsize=(3, 3))
        if cmap is None:
            cmap = plt.get_cmap('twilight')
        # get embeddings for all actions
        embeddings = self.get_action_embeddings_via_g()

        # get sampled action, embedding and policy distribution for current state
        state_feats = self.env.get_features(self.env.current_xy)
        action_ind, sampled_embedding, policy_mean, policy_logstd = self.select_action(state_feats, random_policy=False,
                                                                            noise=True)
        policy_mean = policy_mean.detach().numpy()
        sampled_embedding = sampled_embedding.detach().numpy()
        policy_std = np.exp(policy_logstd.detach().numpy())
        # Reduce to 2D if necessary using PCA
        if embeddings.shape[1] > 2:
            pca = PCA(n_components=2)
            embeddings_2d = pca.fit_transform(embeddings)
        else:
            embeddings_2d = embeddings

        actions = self.env.actions
        action_inds = np.arange(actions.shape[0])

        # Plot the embeddings on the heatmap
        ax.scatter(*embeddings_2d.T, c=action_inds, cmap=cmap, s=s, label='Embeddings')
        ax.scatter(*sampled_embedding, color=policy_color, marker='P', s=s*2)
        ax.scatter(*policy_mean, color=policy_color, s=s*2, marker='X')
        ax.set_aspect('equal', adjustable='box')  # Ensure equal scaling

        ellipse = Circle(policy_mean, policy_std, edgecolor=policy_color, facecolor=policy_color, alpha=0.3)

        ax.add_patch(ellipse)
        # Customize labels and title for left plot
        ax.set_xlabel('Embedding dimension 1')
        ax.set_ylabel('Embedding dimension 2')
        return ax

    def plot_f_output(self, sample_embedding=None, cmap=None, cbar=False, ax=None):
        from matplotlib.patches import Circle

        if cmap is None:
            from matplotlib.colors import ListedColormap
            import distinctipy

            colors = distinctipy.get_colors(self.env.n_actions, pastel_factor=0.7)
            cmap = ListedColormap(colors)

        policy_mean = self.actor(self.env.get_features(self.env.current_xy)).detach().numpy()
        policy_std = self.internal_policy_std

        if sample_embedding is None:
            _, sample_embedding, _, _ = self.select_action(self.env.get_features(self.env.current_xy),
                                                           random_policy=False, noise=True)
            sample_embedding = sample_embedding.detach().numpy()

        # Generate a grid of embeddings
        x_range = np.linspace(-1, 1, 40)
        y_range = np.linspace(-1, 1, 40)
        x, y = np.meshgrid(x_range, y_range)
        embedding_grid = np.vstack([x.ravel(), y.ravel()]).T

        # Compute average action indices
        avg_indices_agent = self.compute_average_action_indices(embedding_grid, n_repeats=20,
                                                                num_actions=self.env.n_actions)

        # Plotting
        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=(3, 2.5))

        scatter2 = ax.scatter(x, y, c=avg_indices_agent, cmap=cmap, s=60)
        ellipse = Circle(policy_mean, policy_std, edgecolor='black', facecolor='none')
        ax.add_patch(ellipse)

        ax.scatter(sample_embedding[0], sample_embedding[1], color='k', marker='*', s=100,
                   label='Last Embedding Sample')

        if cbar:
            plt.colorbar(scatter2, ax=ax)
        ax.set_ylim([-1, 1])
        ax.set_xlim([-1, 1])

    def set_policy_std(self, new_std):
        self.internal_policy_std = new_std

    def compute_average_action_indices(self, embedding_grid, n_repeats=10, num_actions=30):
        averaged_indices = []
        for embedding in embedding_grid:
            indices = []
            for _ in range(n_repeats):
                action_logits = self.f(torch.tensor(embedding, dtype=torch.float32))
                action_probs = torch.softmax(action_logits, dim=0)
                action_idx = torch.multinomial(action_probs, 1).item()
                indices.append(action_idx)
            circular_mean = compute_circular_mean(indices, num_actions)
            averaged_indices.append(circular_mean)
        return np.array(averaged_indices)


class ACLearningAgent:
    """Standard Actor-Critic agent that operates directly in action space."""

    def __init__(self, env, config, full_model_load_path=None,
                 actor_plastic=True, critic_plastic=True):
        self.env = env
        self.gamma = config["discount_factor"]
        self.actor_lr = config["actor_lr"]
        self.critic_lr = config["critic_lr"]
        self.grid_size_rows = config['grid_size'][0]
        self.grid_size_cols = config['grid_size'][1]
        self.softmax_inv_temp = config["inv_temp"]
        self.actor_plastic = actor_plastic
        self.critic_plastic = critic_plastic
        self.policy_noise = config.get("policy_noise", 1e-3)

        state_dim = self.env.n_features
        n_actions = self.env.n_actions

        if full_model_load_path:
            checkpoint = torch.load(full_model_load_path)
            state_dim = checkpoint['params']['state_dim']
            n_actions = checkpoint['params']['n_actions']

            self.actor = ActorDiscrete(state_dim, n_actions)
            self.critic = Critic(state_dim)

            self.actor.load_state_dict(checkpoint['actor_state_dict'])
            self.critic.load_state_dict(checkpoint['critic_state_dict'])
        else:
            self.actor = ActorDiscrete(state_dim, n_actions)
            self.critic = Critic(state_dim)

        if not actor_plastic:
            self.actor_lr = 0
        if not critic_plastic:
            self.critic_lr = 0

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=self.actor_lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=self.critic_lr)

    def select_action(self, state_tensor, random_policy=False, noise=True):
        action_logits = self.actor(state_tensor)

        if random_policy:
            temp = 1e12
        else:
            temp = self.softmax_inv_temp

        action_probs = softmax_with_temperature(action_logits, temperature=temp)

        if noise:
            action_probs = add_noise_to_action_probs(action_probs, noise_level=self.policy_noise)

        action_idx = torch.multinomial(action_probs, 1).item()
        log_prob = torch.log(action_probs[action_idx])

        return action_idx, log_prob

    def update(self, state_tensor, action_ind, log_prob, next_state_tensor, reward, done):
        self.actor_optimizer.zero_grad()
        self.critic_optimizer.zero_grad()

        value = self.critic(state_tensor)
        next_value = self.critic(next_state_tensor).detach() if not done else torch.tensor(0.0)
        target = reward + self.gamma * next_value

        advantage = target - value

        actor_loss = -log_prob * advantage.detach()
        critic_loss = advantage.pow(2)

        if self.actor_plastic:
            actor_loss.backward()
            self.actor_optimizer.step()

        if self.critic_plastic:
            critic_loss.backward()
            self.critic_optimizer.step()

        return actor_loss.item(), critic_loss.item()

    def update_temperature(self, performance_metric, min_temp=5, max_temp=20):
        self.softmax_inv_temp = min_temp + (max_temp - min_temp) * np.exp(-0.1 * performance_metric)

    def find_optimal_action_ind(self, goal=None, rotation_angle=None):
        actions = self.env.actions
        starting_coord = np.array((self.env.env_shape[0] / 2, self.env.env_shape[1] / 2))
        if rotation_angle is None:
            rotation_angle = self.env.adaptation_rotation

        action_consequences = [(self.env.move_in_direction(starting_coord[0],
                                                           starting_coord[1], a,
                                                           adaptation_rotation=rotation_angle)) for a in actions]
        if goal:
            goal_state = goal
        else:
            goal_state = self.env.target_xy

        distances_from_target = [self._calculate_distance_from_target(c[0], c[1], goal_state[0], goal_state[1])
                                 for c in action_consequences]
        return np.argmin(distances_from_target)

    @staticmethod
    def _calculate_distance_from_target(x, y, target_x, target_y):
        return math.sqrt((x - target_x) ** 2 + (y - target_y) ** 2)

    def plot_actions_task_space(self, ax, s=10):
        cmap = plt.get_cmap('twilight')

        actions = self.env.actions
        goal_coord = np.array(self.env.target_xy) - self.env.start_xy
        action_coords = np.array(
            [(self.env.move_in_direction(*self.env.start_xy, a)) for a in actions]) - self.env.start_xy
        action_inds = np.arange(actions.shape[0])

        scatter = ax.scatter(*action_coords.T, c=action_inds, cmap=cmap, s=s, label='Cartesian Coordinates')
        plt.draw()
        colors_from_scatter = scatter.get_facecolors()

        optimal_action_ind = self.find_optimal_action_ind()
        color_for_action = colors_from_scatter[optimal_action_ind]

        ax.set_title('Cartesian Coordinates')
        ax.set_xlabel('Delta X')
        ax.set_ylabel('Delta Y')
        ax.scatter(*goal_coord, marker='*', color=color_for_action, s=s * 5)
        return ax


class ActorDiscrete(nn.Module):
    """Actor network that outputs logits over discrete actions."""

    def __init__(self, state_dim, n_actions, xavier=False):
        super(ActorDiscrete, self).__init__()
        self.linear = nn.Linear(state_dim, n_actions)

        nn.init.normal_(self.linear.weight, mean=0.0, std=1e-2)
        nn.init.constant_(self.linear.bias, 0.0)

        if xavier:
            nn.init.xavier_uniform_(self.linear.weight)
            nn.init.zeros_(self.linear.bias)

    def forward(self, state):
        return self.linear(state)
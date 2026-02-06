import matplotlib.pyplot as plt
import numpy as np
import os
import matplotlib
from matplotlib.patches import Circle
import wandb
import torch
import math
from sklearn.decomposition import PCA
from core.utils import compute_circular_mean


def set_plotting_defaults(font_size=7, font_style='Arial'):
    font = {'size': font_size}
    matplotlib.rc('font', **font)
    matplotlib.rcParams['font.sans-serif'] = font_style
    matplotlib.rcParams['pdf.fonttype'] = 42


def plot_embeddings(embeddings, mean_emb, std_emb, env, embedding_history):

    cmap = plt.get_cmap('twilight')

    if embeddings.shape[1] > 2:
        from sklearn.decomposition import PCA
        pca = PCA(n_components=2)
        embeddings_2d = pca.fit_transform(embeddings)
    else:
        embeddings_2d = embeddings

    actions = env.actions
    starting_coord = np.array((env.env_shape[0] / 2, env.env_shape[1] / 2))
    action_coords = np.array([(env.move_in_direction(starting_coord[0],
                                                     starting_coord[1], a)) for a in actions]) - starting_coord
    action_inds = np.arange(actions.shape[0])
    # Create a grid for the Gaussian distribution
    x = np.linspace(mean_emb[0] - 3 * std_emb, mean_emb[0] + 3 * std_emb, 100)
    y = np.linspace(mean_emb[1] - 3 * std_emb, mean_emb[1] + 3 * std_emb, 100)
    x, y = np.meshgrid(x, y)

    # Calculate the Gaussian values
    z = (1 / (2 * np.pi * std_emb ** 2)) * np.exp(
        -(((x - mean_emb[0]) ** 2 + (y - mean_emb[1]) ** 2) / (2 * std_emb ** 2))
    )

    fig, ax = plt.subplots(1, 2, figsize=(4, 1.65))

    extent = [x.min(), x.max(), y.min(), y.max()]  # Define the extent of the plot

    # Add a color bar for the heatmap

    # Plot the embeddings on the heatmap
    ax[0].scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], c=action_inds, cmap=cmap, s=10, label='Embeddings')
    ax[0].scatter(embedding_history[-1][0], embedding_history[-1][1], color='k', marker='*', s=20)
    ax[0].scatter(mean_emb[0], mean_emb[1], color='#711D4F', s=10)

    ellipse = Circle(mean_emb, std_emb, edgecolor='#711D4F', facecolor='#711D4F', alpha=0.3)

    ax[0].add_patch(ellipse)
    #ax[0].set_title('cue: {}'.format(env.cue))

    # Customize labels and title for left plot
    ax[0].set_xlabel('X')
    ax[0].set_ylabel('Y')

    # Right plot: Cartesian coordinates of actions
    scatter = ax[1].scatter(action_coords[:, 0], action_coords[:, 1], c=action_inds, cmap=cmap, s=10,
                            label='Cartesian Coordinates')
    cbar = plt.colorbar(scatter, ax=ax[1])

    ax[1].set_title('Cartesian Coordinates')
    ax[1].set_xlabel('Delta X')
    ax[1].set_ylabel('Delta Y')
    cbar = plt.colorbar(scatter, ax=ax[0])
    cbar.set_label('Action index')
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color='black')


    plt.tight_layout()
    return fig, ax[0]

def plot_embeddings_only(embeddings, env, ax=None):

    cmap = plt.get_cmap('twilight')

    if embeddings.shape[1] > 2:
        from sklearn.decomposition import PCA
        pca = PCA(n_components=2)
        embeddings_2d = pca.fit_transform(embeddings)
    else:
        embeddings_2d = embeddings

    actions = env.actions
    starting_coord = np.array((env.env_shape[0] / 2, env.env_shape[1] / 2))
    action_coords = np.array([(env.move_in_direction(starting_coord[0],
                                                     starting_coord[1], a)) for a in actions]) - starting_coord
    action_inds = np.arange(actions.shape[0])
    if ax is None:
        fig, ax = plt.subplots(1, 2, figsize=(4, 1.65))

        # Plot the embeddings on the heatmap
        ax[0].scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], c=action_inds, cmap=cmap, s=10, label='Embeddings')

        # Customize labels and title for left plot
        ax[0].set_xlabel('e1')
        ax[0].set_ylabel('e2')
        # Right plot: Cartesian coordinates of actions
        scatter = ax[1].scatter(action_coords[:, 0], action_coords[:, 1], c=action_inds, cmap=cmap, s=10,
                                label='Cartesian Coordinates')
        cbar = plt.colorbar(scatter, ax=ax[1])

        ax[1].set_title('Cartesian Coordinates')
        ax[1].set_xlabel('Delta X')
        ax[1].set_ylabel('Delta Y')

        # set xlim and ylim to -1, 1
        ax[0].set_xlim(-1.1, 1.1)
        ax[0].set_ylim(-1.1, 1.1)
        ax[1].set_xlim(-1.1, 1.1)
        ax[1].set_ylim(-1.1, 1.1)
        cbar = plt.colorbar(scatter, ax=ax[0])
        cbar.set_label('Action index')
        plt.setp(cbar.ax.yaxis.get_ticklabels(), color='black')
        plt.tight_layout()
        return fig

    else:
        # Add a color bar for the heatmap

        # Plot the embeddings on the heatmap
        ax.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], c=action_inds, cmap=cmap, s=10, label='Embeddings')

        # Customize labels and title for left plot
        ax.set_xlabel('e1')
        ax.set_ylabel('e2')

        # set xlim and ylim to -1, 1
        ax.set_xlim(-1.1, 1.1)
        ax.set_ylim(-1.1, 1.1)
        return ax



def plot_embeddings_only_3d_compatible(embeddings, env):
    import matplotlib.pyplot as plt
    import numpy as np

    cmap = plt.get_cmap('twilight')
    d = embeddings.shape[1]

    # Decide dimensionality
    if d == 3:
        emb_plot = embeddings
        fig = plt.figure(figsize=(4.5, 1.8))
        ax0 = fig.add_subplot(1, 2, 1, projection='3d')
        ax1 = fig.add_subplot(1, 2, 2)
    elif d > 3:
        from sklearn.decomposition import PCA
        emb_plot = PCA(n_components=2).fit_transform(embeddings)
        fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(4.5, 1.8))
    else:  # 2D
        emb_plot = embeddings
        fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(4.5, 1.8))

    actions = env.actions
    starting_coord = np.array((env.env_shape[0] / 2, env.env_shape[1] / 2))
    action_coords = np.array([
        env.move_in_direction(starting_coord[0], starting_coord[1], a)
        for a in actions
    ]) - starting_coord
    action_inds = np.arange(actions.shape[0])

    # --- Left: embeddings ---
    if d == 3:
        scatter0 = ax0.scatter(emb_plot[:, 0], emb_plot[:, 1], emb_plot[:, 2],
                               c=action_inds, cmap=cmap, s=10)
        ax0.set_xlabel('X'); ax0.set_ylabel('Y'); ax0.set_zlabel('Z')
        ax0.set_xlim(-1.1, 1.1); ax0.set_ylim(-1.1, 1.1); ax0.set_zlim(-1.1, 1.1)
    else:
        scatter0 = ax0.scatter(emb_plot[:, 0], emb_plot[:, 1],
                               c=action_inds, cmap=cmap, s=10)
        ax0.set_xlabel('X'); ax0.set_ylabel('Y')
        ax0.set_xlim(-1.1, 1.1); ax0.set_ylim(-1.1, 1.1)

    # --- Right: action deltas ---
    scatter1 = ax1.scatter(action_coords[:, 0], action_coords[:, 1],
                           c=action_inds, cmap=cmap, s=10)
    ax1.set_title('Cartesian Coordinates')
    ax1.set_xlabel('Delta X'); ax1.set_ylabel('Delta Y')
    ax1.set_xlim(-1.1, 1.1); ax1.set_ylim(-1.1, 1.1)

    # Shared colorbar
    cbar = fig.colorbar(scatter1, ax=[ax0, ax1], fraction=0.046, pad=0.04)
    cbar.set_label('Action index')
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color='black')

    plt.tight_layout()
    return fig

def plot_embeddings_only_half_step_compatible(embeddings, env):
    # Cyclic colormaps
    cmap_full = plt.get_cmap('twilight')
    cmap_half = plt.get_cmap('hsv')

    if embeddings.shape[1] > 3:
        from sklearn.decomposition import PCA
        embeddings_2d = PCA(n_components=2).fit_transform(embeddings)
    else:
        embeddings_2d = embeddings

    actions = env.actions
    start = np.array((env.env_shape[0] / 2, env.env_shape[1] / 2), dtype=float)

    def token_to_delta(tok):
        x0, y0 = start
        if isinstance(tok, tuple):
            theta, dist = tok
            if theta is None:   # do-nothing action
                return np.array([0.0, 0.0]), False
            x1, y1 = env.move_in_direction(
                x0, y0, theta, distance=dist, adaptation_rotation=env.adaptation_rotation
            )
            return np.array([x1 - x0, y1 - y0]), (abs(dist - 0.5) < 1e-6)
        else:
            theta = float(tok)
            x1, y1 = env.move_in_direction(
                x0, y0, theta, distance=1.0, adaptation_rotation=env.adaptation_rotation
            )
            return np.array([x1 - x0, y1 - y0]), False

    coords, is_half = [], []
    for tok in actions:
        delta, half = token_to_delta(tok)
        coords.append(delta)
        is_half.append(half)
    coords = np.vstack(coords)
    is_half = np.array(is_half)

    fig, ax = plt.subplots(1, 2, figsize=(5.5, 2))

    idxs_full = np.where(~is_half)[0]
    idxs_half = np.where(is_half)[0]

    # Left: embeddings
    ax[0].scatter(embeddings_2d[idxs_full, 0], embeddings_2d[idxs_full, 1],
                  c=idxs_full, cmap=cmap_full, s=10)
    if len(idxs_half) > 0:
        ax[0].scatter(embeddings_2d[idxs_half, 0], embeddings_2d[idxs_half, 1],
                      c=idxs_half, cmap=cmap_half, s=10)
    ax[0].set_xlabel('X')
    ax[0].set_ylabel('Y')

    # Right: action deltas
    ax[1].scatter(coords[idxs_full, 0], coords[idxs_full, 1],
                  c=idxs_full, cmap=cmap_full, s=10)
    if len(idxs_half) > 0:
        ax[1].scatter(coords[idxs_half, 0], coords[idxs_half, 1],
                      c=idxs_half, cmap=cmap_half, s=10)
    ax[1].set_title('Cartesian Coordinates')
    ax[1].set_xlabel('Delta X')
    ax[1].set_ylabel('Delta Y')

    for a in ax:
        a.set_xlim(-1.1, 1.1)
        a.set_ylim(-1.1, 1.1)

    plt.tight_layout()
    return fig


def plot_embeddings_only_multi_step_compatible(embeddings, env):
    # Multiple colormaps for different step sizes
    colormaps = ['viridis', 'plasma', 'inferno', 'cividis']

    # Reduce to 2D if needed
    if embeddings.shape[1] > 2:
        from sklearn.decomposition import PCA
        embeddings_2d = PCA(n_components=2).fit_transform(embeddings)
    else:
        embeddings_2d = embeddings

    actions = env.actions
    start = np.array((env.env_shape[0] / 2, env.env_shape[1] / 2), dtype=float)

    def token_to_delta_and_step_size(tok):
        x0, y0 = start
        if isinstance(tok, tuple):
            theta, dist = tok
            if theta is None:  # do-nothing action
                return np.array([0.0, 0.0]), dist
            x1, y1 = env.move_in_direction(
                x0, y0, theta, distance=dist,
                adaptation_rotation=env.adaptation_rotation
            )
            return np.array([x1 - x0, y1 - y0]), dist
        else:
            theta = float(tok)
            x1, y1 = env.move_in_direction(
                x0, y0, theta, distance=1.0,
                adaptation_rotation=env.adaptation_rotation
            )
            return np.array([x1 - x0, y1 - y0]), 1.0

    coords, step_sizes = [], []
    for tok in actions:
        delta, step_size = token_to_delta_and_step_size(tok)
        coords.append(delta)
        step_sizes.append(step_size)

    coords = np.vstack(coords)
    step_sizes = np.array(step_sizes)

    # Get unique step sizes and assign colors
    unique_step_sizes = np.unique(step_sizes)

    fig, ax = plt.subplots(1, 2, figsize=(5.5, 2))

    # Plot each step size with different colormap
    for i, step_size in enumerate(unique_step_sizes):
        mask = step_sizes == step_size
        indices = np.where(mask)[0]
        cmap = plt.get_cmap(colormaps[i % len(colormaps)])

        # Left: embeddings
        ax[0].scatter(embeddings_2d[mask, 0], embeddings_2d[mask, 1],
                      c=indices, cmap=cmap, s=10,
                      label=f'{step_size}x step')

        # Right: action deltas
        ax[1].scatter(coords[mask, 0], coords[mask, 1],
                      c=indices, cmap=cmap, s=10,
                      label=f'{step_size}x step')

    ax[0].set_xlabel('Embedding X')
    ax[0].set_ylabel('Embedding Y')
    ax[0].set_title('Learned Embeddings')
    ax[0].legend(bbox_to_anchor=(1.05, 1), loc='upper left')

    ax[1].set_title('Cartesian Coordinates')
    ax[1].set_xlabel('Delta X')
    ax[1].set_ylabel('Delta Y')

    for a in ax:
        a.set_xlim(-1.1, 1.1)
        a.set_ylim(-1.1, 1.1)

    plt.tight_layout()
    return fig


def plot_embeddings_and_mean(embeddings, mean_emb, std_emb, env, embedding_history, color, cmap, ax):
    if embeddings.shape[1] > 2:
        pca = PCA(n_components=2)
        embeddings_2d = pca.fit_transform(embeddings)
    else:
        embeddings_2d = embeddings

    actions = env.actions
    starting_coord = np.array((env.env_shape[0] / 2, env.env_shape[1] / 2))
    action_coords = np.array([(env.move_in_direction(starting_coord[0],
                                                     starting_coord[1], a)) for a in actions]) - starting_coord
    action_inds = np.arange(actions.shape[0])

    # Plot the embeddings on the heatmap
    ax.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], c=action_inds, cmap=cmap, s=10, label='Embeddings')
    ax.scatter(embedding_history[-1][0], embedding_history[-1][1], color=color, marker='P', s=20)
    ax.scatter(mean_emb[0], mean_emb[1], color=color, s=20, marker='X')
    ax.set_aspect('equal', adjustable='box')  # Ensure equal scaling

    ellipse = Circle(mean_emb, std_emb, edgecolor=color, facecolor=color, alpha=0.3)

    ax.add_patch(ellipse)
    # Customize labels and title for left plot
    ax.set_xlabel('X')
    ax.set_ylabel('Y')


def plot_embeddings_one_cue(embeddings, mean_emb, std_emb, env, embedding_history, color):
    cmap = plt.get_cmap('twilight')
    fig, ax = plt.subplots(1, 2, figsize=(4, 1.65))
    plot_embeddings_and_mean(embeddings, mean_emb, std_emb, env, embedding_history, color, cmap, ax[0])
    # Right plot: Cartesian coordinates of actions
    actions = env.actions
    starting_coord = np.array((env.env_shape[0] / 2, env.env_shape[1] / 2))
    goal_coord = np.array(env.target_xy) - starting_coord
    action_coords = np.array([(env.move_in_direction(starting_coord[0],
                                                     starting_coord[1], a)) for a in actions]) - starting_coord
    action_inds = np.arange(actions.shape[0])
    scatter = ax[1].scatter(action_coords[:, 0], action_coords[:, 1], c=action_inds, cmap=cmap, s=10,
                            label='Cartesian Coordinates')

    cbar = plt.colorbar(scatter, ax=ax[1])

    optimal_action_ind = find_optimal_action_ind(env)
    norm = plt.Normalize(vmin=action_inds.min(), vmax=action_inds.max())
    color_for_action = cmap(norm(optimal_action_ind))

    ax[1].set_title('Cartesian Coordinates')
    ax[1].set_xlabel('Delta X')
    ax[1].set_ylabel('Delta Y')
    ax[1].scatter(*goal_coord, marker='*', color=color_for_action, s=50)
    cbar = plt.colorbar(scatter, ax=ax[0])
    cbar.set_label('Action index')
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color='black')

    plt.tight_layout()
    return fig, ax[0]

def plot_embedding_overlaid_two_cues(embeddings, mean_embs, std_embs, env, embedding_histories, block_num, colors):
    cmap = plt.get_cmap('twilight')
    fig, ax = plt.subplots(1, 2, figsize=(4, 1.65))
    if block_num >=2:
        for cue_ind, mean_emb_hist in mean_embs.items():
            mean_emb = mean_emb_hist[-1]
            std_emb = std_embs[cue_ind][-1]
            embedding_history = embedding_histories[cue_ind]
            color = colors[cue_ind]
            plot_embeddings_and_mean(embeddings, mean_emb, std_emb, env, embedding_history, color, cmap, ax[0])
    else:
        cue_ind = 0
        mean_emb = mean_embs[cue_ind][-1]
        std_emb = std_embs[cue_ind][-1]
        embedding_history = embedding_histories[cue_ind]
        color = colors[cue_ind]
        plot_embeddings_and_mean(embeddings, mean_emb, std_emb, env, embedding_history, color, cmap, ax[0])


    # Right plot: Cartesian coordinates of actions
    actions = env.actions
    starting_coord = np.array((env.env_shape[0] / 2, env.env_shape[1] / 2))
    action_coords = np.array([(env.move_in_direction(starting_coord[0],
                                                     starting_coord[1], a)) for a in actions]) - starting_coord
    action_inds = np.arange(actions.shape[0])
    scatter = ax[1].scatter(action_coords[:, 0], action_coords[:, 1], c=action_inds, cmap=cmap, s=10,
                            label='Cartesian Coordinates')
    cbar = plt.colorbar(scatter, ax=ax[1])
    ax[1].set_title('Cartesian Coordinates')
    ax[1].set_xlabel('Delta X')
    ax[1].set_ylabel('Delta Y')
    cbar = plt.colorbar(scatter, ax=ax[0])
    cbar.set_label('Action index')
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color='black')

    plt.tight_layout()
    return fig, ax[0]

def plot_embedding_overlaid_two_cues_post_hoc(embeddings, mean_embs, std_emb, env, embedding_histories, colors):
    cmap = plt.get_cmap('twilight')
    fig, ax = plt.subplots(1, 2, figsize=(4, 1.65))

    for cue_ind, mean_emb in mean_embs.items():
        std_emb = std_emb
        embedding_history = embedding_histories[cue_ind]
        color = colors[cue_ind]
        plot_embeddings_and_mean(embeddings, mean_emb, std_emb, env, embedding_history, color, cmap, ax[0])

    # Right plot: Cartesian coordinates of actions
    actions = env.actions
    starting_coord = np.array((env.env_shape[0] / 2, env.env_shape[1] / 2))
    action_coords = np.array([(env.move_in_direction(starting_coord[0],
                                                     starting_coord[1], a)) for a in actions]) - starting_coord
    action_inds = np.arange(actions.shape[0])
    scatter = ax[1].scatter(action_coords[:, 0], action_coords[:, 1], c=action_inds, cmap=cmap, s=10,
                            label='Cartesian Coordinates')
    cbar = plt.colorbar(scatter, ax=ax[1])
    ax[1].set_title('Cartesian Coordinates')
    ax[1].set_xlabel('Delta X')
    ax[1].set_ylabel('Delta Y')
    cbar = plt.colorbar(scatter, ax=ax[0])
    cbar.set_label('Action index')
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color='black')

    plt.tight_layout()
    return fig, ax[0]

def plot_embeddings_with_gaussian_3d(embeddings, mean_emb, std_emb, env, embedding_history):
    cmap = plt.get_cmap('viridis')

    if embeddings.shape[1] > 2:
        from sklearn.decomposition import PCA
        pca = PCA(n_components=2)
        embeddings_2d = pca.fit_transform(embeddings)
    else:
        embeddings_2d = embeddings

    actions = env.actions
    starting_coord = np.array((env.env_shape[0] / 2, env.env_shape[1] / 2))
    action_coords = np.array([(env.move_in_direction(starting_coord[0],
                                                     starting_coord[1], a)) for a in actions]) - starting_coord

    # Normalize actions for color mapping
    norm = Normalize(vmin=actions.min(), vmax=actions.max())
    colors = cmap(norm(actions))

    # Create a grid for the Gaussian distribution
    x = np.linspace(mean_emb[0] - 3 * std_emb, mean_emb[0] + 3 * std_emb, 100)
    y = np.linspace(mean_emb[1] - 3 * std_emb, mean_emb[1] + 3 * std_emb, 100)
    x, y = np.meshgrid(x, y)

    # Calculate the Gaussian z-values
    z = (1 / (2 * np.pi * std_emb**2)) * np.exp(
        -(((x - mean_emb[0]) ** 2 + (y - mean_emb[1]) ** 2) / (2 * std_emb**2))
    )

    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Plot the Gaussian surface
    ax.plot_surface(x, y, z, cmap=plt.cm.magma, edgecolor='none', alpha=0.8)

    # Plot the embeddings on the plane z=0
    ax.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], 0, c='cyan', s=50, depthshade=True, label='Embeddings')

    # Highlight the most recent embedding point
    ax.scatter(embedding_history[-1][0], embedding_history[-1][1], 0, color='red', s=100, label='Last Embedding')

    # Plot the mean embedding as a magenta point
    ax.scatter(mean_emb[0], mean_emb[1], 0, color='magenta', s=100, label='Mean Embedding')

    # Set pane background to black
    ax.zaxis.pane.set_facecolor((0, 0, 0, 1))
    ax.zaxis.pane.set_alpha(1)
    ax.grid(False)
    ax.yaxis.pane.fill = False
    ax.xaxis.pane.fill = False

    # Customize labels and title
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Density')
    ax.set_title('3D Embeddings with Gaussian Distribution')

    ax.legend()
    return fig, ax

def plot_embeddings_with_gaussian_2d(embeddings, mean_emb, std_emb, env, embedding_history, cmap=None):
    if not cmap:
        cmap = plt.get_cmap('twilight')

    if embeddings.shape[1] > 2:
        from sklearn.decomposition import PCA
        pca = PCA(n_components=2)
        embeddings_2d = pca.fit_transform(embeddings)
    else:
        embeddings_2d = embeddings

    actions = env.actions
    starting_coord = np.array((env.env_shape[0] / 2, env.env_shape[1] / 2))
    action_coords = np.array([(env.move_in_direction(starting_coord[0],
                                                     starting_coord[1], a, adaptation_rotation=env.adaptation_rotation)) for a in actions]) - starting_coord


    # Create a grid for the Gaussian distribution
    x = np.linspace(mean_emb[0] - 3 * std_emb, mean_emb[0] + 3 * std_emb, 100)
    y = np.linspace(mean_emb[1] - 3 * std_emb, mean_emb[1] + 3 * std_emb, 100)
    x, y = np.meshgrid(x, y)

    # Calculate the Gaussian values
    z = (1 / (2 * np.pi * std_emb**2)) * np.exp(
        -(((x - mean_emb[0]) ** 2 + (y - mean_emb[1]) ** 2) / (2 * std_emb**2))
    )

    fig, ax = plt.subplots(1, 2, figsize=(4, 1.65))

    # Left plot: Gaussian heatmap with embeddings
    # Left plot: Gaussian heatmap with embeddings using imshow
    ax[0].set_facecolor('black')
    extent = [x.min(), x.max(), y.min(), y.max()]  # Define the extent of the plot
    heatmap = ax[0].imshow(z, extent=extent, origin='lower', cmap=plt.cm.magma, aspect='auto')

    # Add a color bar for the heatmap
    cbar = plt.colorbar(heatmap, ax=ax[0])
    cbar.set_label('Density', color='black')
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color='black')

    # Plot the embeddings on the heatmap
    ax[0].scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], c=actions, cmap=cmap, s=10, label='Embeddings')
    ax[0].scatter(embedding_history[-1][0], embedding_history[-1][1], color='white', marker='+', s=20, label='Last Embedding')

    # Customize labels and title for left plot
    ax[0].set_xlabel('X')
    ax[0].set_ylabel('Y')
    ax[0].set_title('2D Embeddings with Policy Heatmap')

    # Right plot: Cartesian coordinates of actions
    action_inds = np.arange(actions.shape[0])
    scatter = ax[1].scatter(action_coords[:, 0], action_coords[:, 1], c=action_inds, cmap=cmap, s=10, label='Cartesian Coordinates')
    # plot the reward zone
    start_state = (env.env_shape[0] / 2, env.env_shape[1] / 2)
    goal_state = env.target_xy
    target_displacement = (goal_state[0] - start_state[0], goal_state[1] - start_state[1])
    ax[1].scatter(target_displacement[0], target_displacement[1], c='k', marker ='+', s=15)
    #ellipse = Circle(target_displacement, env.reward_window, edgecolor='white', facecolor='none', linestyle='--')

    #ax[1].add_patch(ellipse)
    cbar = plt.colorbar(scatter, ax=ax[1])
    cbar.set_label('Action index')
    ax[1].set_facecolor('black')  # Set background for right plot
    ax[1].set_title('Cartesian Coordinates')
    ax[1].set_xlabel('Delta X')
    ax[1].set_ylabel('Delta Y')

    plt.tight_layout()

    return fig, ax



def get_ellipse_properties(sigma_x, sigma_y, rho):
    """
    Calculate the width, height, and angle of an ellipse from sigma_x, sigma_y, and rho.

    Parameters:
    - sigma_x (float): Standard deviation along the x-axis.
    - sigma_y (float): Standard deviation along the y-axis.
    - rho (float): Correlation coefficient (between -1 and 1).

    Returns:
    - width (float): Full width of the ellipse (major axis).
    - height (float): Full height of the ellipse (minor axis).
    - angle (float): Angle of rotation in degrees (counterclockwise from the x-axis).
    """
    # Covariance matrix
    cov_matrix = np.array([
        [sigma_x ** 2, rho * sigma_x * sigma_y],
        [rho * sigma_x * sigma_y, sigma_y ** 2]
    ])

    # Eigen decomposition
    eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)

    # Eigenvalues correspond to the lengths of the semi-major and semi-minor axes
    # Width and height are twice the square root of the eigenvalues
    width = 2 * np.sqrt(eigenvalues[1])  # Larger eigenvalue -> major axis
    height = 2 * np.sqrt(eigenvalues[0])  # Smaller eigenvalue -> minor axis

    # Angle of the ellipse in radians, converted to degrees
    angle = np.arctan2(eigenvectors[1, 1], eigenvectors[0, 1])  # Rotation based on eigenvector
    angle_degrees = np.degrees(angle)

    return width, height, angle_degrees

def plot_loss_rolling(losses, ax, window_size=10000, color='k', label='loss'):
    losses = np.array(losses)

    rolling_mean = np.convolve(losses, np.ones(window_size) / window_size, mode='valid')

    ax.plot(rolling_mean, color=color, linestyle='-', label=label)

def plot_angle_diffs_rolling(losses, ax, window_size=10000, color='k', label='loss'):
    losses = np.radians(np.array(losses))  # Convert degrees to radians
    losses = np.array(losses)  # Already in degrees

    rolling_means = [np.mean(losses[i:i + window_size]) for i in range(len(losses) - window_size + 1)]

    ax.plot(rolling_means, color=color, linestyle='-', label=label)



def plot_loss_binned(losses, ax, window_size=10000, color='k', label='loss'):
    losses = np.array(losses)

    # Compute the number of bins
    num_bins = len(losses) // window_size
    if num_bins == 0:
        raise ValueError("Window size is too large compared to the number of losses.")

    # Compute mean loss per bin
    binned_means = [np.mean(losses[i * window_size:(i + 1) * window_size]) for i in range(num_bins)]

    # X-axis: bin centers
    bin_positions = np.arange(num_bins) * window_size + window_size // 2

    ax.plot(bin_positions, binned_means, color=color, linestyle='-', label=label)


def circular_distance_from_target(numbers, target=15, period=30):
    distances = []
    for x in numbers:
        diff = abs(x - target)
        circular_dist = min(diff, period - diff)
        distances.append(circular_dist)
    return distances


def plot_weights_imshow(model):
    weights = model.linear.weight.detach().cpu().numpy()  # Extract weights
    plt.figure(figsize=(10, 6))
    plt.imshow(weights, cmap='viridis', aspect='auto')
    plt.colorbar(label='Weight Magnitude')
    plt.xlabel("Embedding Dimension")
    plt.ylabel("Action Dimension")
    plt.title("ActionMapping Weight Visualization")

    # Log the image to wandb
    wandb.log({"weight_plot": wandb.Image(plt.gcf())})


def plot_weights(model):
    weights = model.linear.weight.detach().cpu().numpy()  # Extract weights
    action_dim = weights.shape[0]

    plt.figure(figsize=(10, 6))
    plt.plot(range(action_dim), weights[:, 0], label='Dim 1', color='blue')
    plt.plot(range(action_dim), weights[:, 1], label='Dim 2', color='red')
    plt.xlabel("Action Dimension")
    plt.ylabel("Weight Value")
    plt.title("ActionMapping Weight Visualization")
    plt.legend()

    # Log the image to wandb
    wandb.log({"weight_plot": wandb.Image(plt.gcf())})


def plot_weight_changes(model):
    current_weights = model.linear.weight.detach().cpu().numpy()
    if model.prev_weights is None:
        model.prev_weights = model.linear.weight.detach().cpu().numpy()
    weight_diff = current_weights - model.prev_weights
    action_dim = weight_diff.shape[0]

    plt.figure(figsize=(10, 6))
    plt.plot(range(action_dim), weight_diff[:, 0], label='Dim 1 Change', color='blue')
    plt.plot(range(action_dim), weight_diff[:, 1], label='Dim 2 Change', color='red')
    plt.xlabel("Action Dimension")
    plt.ylabel("Weight Change")
    plt.title("Change in ActionMapping Weights Since Last Log Interval")
    plt.legend()

    # Log the weight changes to wandb
    wandb.log({"weight_change_plot": wandb.Image(plt.gcf())})
    model.prev_weights = current_weights.copy()# Update stored weights for next comparison


def log_weights_to_wandb(model):
    plot_weights(model)
    plot_weight_changes(model)


def compute_average_action_indices(agent, embedding_grid, n_repeats=10, num_actions=30):
    averaged_indices = []
    for embedding in embedding_grid:
        indices = []
        for _ in range(n_repeats):
            action_logits = agent.f(torch.tensor(embedding, dtype=torch.float32))
            action_probs = torch.softmax(action_logits, dim=0)
            action_idx = torch.multinomial(action_probs, 1).item()
            indices.append(action_idx)
        circular_mean = compute_circular_mean(indices, num_actions)
        averaged_indices.append(circular_mean)
    return np.array(averaged_indices)


def plot_f_output(agent, env, mean_emb, std_emb, last_embedding, cmap, episode_num, fig_dir, save=False, log_wandb=True):
    # Generate a grid of embeddings
    x_range = np.linspace(-1, 1, 40)
    y_range = np.linspace(-1, 1, 40)
    x, y = np.meshgrid(x_range, y_range)
    embedding_grid = np.vstack([x.ravel(), y.ravel()]).T

    # Compute average action indices
    avg_indices_agent = compute_average_action_indices(agent, embedding_grid, n_repeats=20, num_actions=env.n_actions)

    # Plotting
    fig, axs2 = plt.subplots(1, 1, figsize=(3, 2.5))
    scatter2 = axs2.scatter(x, y, c=avg_indices_agent, cmap=cmap, s=60)
    ellipse = Circle(mean_emb, std_emb, edgecolor='black', facecolor='none')
    axs2.add_patch(ellipse)
    axs2.scatter(last_embedding[0], last_embedding[1], color='k', marker='*', s=100,
                  label='Last Embedding')

    plt.colorbar(scatter2, ax=axs2)
    plt.tight_layout()
    plt.axis('equal')
    axs2.set_ylim([-1, 1])
    axs2.set_xlim([-1, 1])

    if log_wandb:
        wandb.log({"f_output": wandb.Image(plt.gcf())})
    if save:
        fig.savefig(os.path.join(fig_dir, 'adaptation_f_output_with_policy_ep_{}_rainbow.pdf'.format(episode_num)))


def calculate_distance_from_target(x, y, target_x, target_y):
    distance = math.sqrt((x - target_x) ** 2 + (y - target_y) ** 2)
    return distance

def find_optimal_action_ind(env, goal=None):
    actions = env.actions
    starting_coord = np.array((env.env_shape[0] / 2, env.env_shape[1] / 2))
    action_consequences = [(env.move_in_direction(env.start_xy[0],
                                     env.start_xy[1], a, adaptation_rotation=env.adaptation_rotation)) for a in
              actions]
    if goal:
        goal_state = goal
    else:
        goal_state = env.target_xy
    distances_from_target = [(calculate_distance_from_target(consequence[0], consequence[1], goal_state[0],
                                                             goal_state[1])) for consequence in action_consequences]
    optimal_action_ind = np.argmin(distances_from_target)
    return optimal_action_ind


def find_angle_difference(env, action, goal=None):
    """
    Finds the shortest angular difference between a given action and the optimal action.

    Parameters:
        env: The environment object containing action information.
        action: The action (angle in radians) for which the difference is computed.

    Returns:
        Angle difference in degrees
    """
    # Get the index of the optimal action
    optimal_action_ind = find_optimal_action_ind(env, goal=goal)

    # Extract the angles
    optimal_angle = env.actions[optimal_action_ind]  # Optimal action's angle
    action_angle = action  # The given action's angle

    # Compute shortest angular difference using atan2
    angle_diff = np.degrees(np.arctan2(np.sin(optimal_angle - action_angle), np.cos(optimal_angle - action_angle)))

    return angle_diff


def plot_pca_features(env):
    # --- collect features for all actions ---
    env.reset()
    centre = env.current_xy
    feats, endpoints = [], []
    for i in range(env.n_actions):
        env.current_xy = centre
        a = env.actions[i]
        _, _, _ = env.act(a)              # step to get endpoint
        f = env.get_features(env.current_xy)
        feats.append(f.detach().cpu().numpy())
        endpoints.append(env.current_xy)
    F = np.array(feats)     # (A, n_features)
    XY = np.array(endpoints)

    # --- z-score across actions ---
    F_std = (F - F.mean(axis=0)) / (F.std(axis=0) + 1e-12)

    # --- PCA projection ---
    Z = PCA(n_components=2).fit_transform(F_std)

    # --- angle for coloring ---
    v = XY - np.asarray(centre)
    ang = np.arctan2(v[:,1], v[:,0])  # [-pi,pi]

    # --- plot ---
    plt.figure(figsize=(5,5))
    sc = plt.scatter(Z[:,0], Z[:,1], c=ang, cmap="twilight", s=60, edgecolor="k")
    plt.axis("equal")
    plt.title("PCA of z-scored FEATURES (twilight colormap)")
    plt.colorbar(sc, label="Angle (radians)")
    plt.tight_layout()
    plt.show()

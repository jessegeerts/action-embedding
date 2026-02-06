import numpy as np
import math
import torch
import matplotlib.pyplot as plt


class ReachTask:
    def __init__(self, config, adaptation_rotation=0):
        self.env_shape = config["grid_size"]
        self.n_actions = config["num_actions"]
        self.n_states = np.prod(self.env_shape)

        self.fourier_order = config["fourier_order"]
        self.adaptation_rotation = adaptation_rotation
        self.target_radius = config["reward_radius_min"]
        self.reward_value = config["reward_for_hit"]
        self.penalty_for_miss = config["penalty_for_miss"]
        self.current_xy = (self.env_shape[0] / 2, self.env_shape[1] / 2)
        self.start_xy = self.current_xy

        self.reach_angle = config["reach_angle"]
        self.target_xy = (self.start_xy[0] + config["reach_length"] * np.cos(config["reach_angle"]),
                          self.start_xy[1] + config["reach_length"] * np.sin(config["reach_angle"]))

        print('initial loc: {}, target:{}'.format(self.current_xy, self.target_xy))
        self.timer = 0
        self.max_trial_length = config["max_steps"]

        self.basis_type = "Fourier"  # note: we haven't implemented any other bases

        self.frequencies = [
            (0, 1),  # cos(πy), sin(πy) - horizontal stripes
            (1, 0),  # cos(πx), sin(πx) - vertical stripes
            (1, 1),  # cos(π(x+y)), sin(π(x+y)) - diagonal bands
            (1, -1),  # cos(π(x-y)), sin(π(x-y)) - opposite diagonal
            (2, 0),  # cos(2πx), sin(2πx) - finer vertical stripes
            (0, 2),  # cos(2πy), sin(2πy) - finer horizontal stripes
            (2, 2),  # cos(2π(x+y)), sin(2π(x+y)) - fine diagonal
        ]
        self.n_features = 2 * len(self.frequencies)  # cos+sin per frequency

        # Actions init
        self.actions = np.linspace(0, 2 * np.pi, self.n_actions, endpoint=False)
        self.action_idx = {a: idx for a, idx in zip(self.actions, range(self.n_actions))}

    def set_visuomotor_rotation(self, rotation):
        self.adaptation_rotation = rotation

    def outside_env(self, x, y):
        return (x >= self.env_shape[0]) or (x < 0) or (y >= self.env_shape[1]) or (y < 0)

    def is_point_within_circle(self, x, y, target_x, target_y):
        distance = math.sqrt((x - target_x) ** 2 + (y - target_y) ** 2)
        return distance <= self.target_radius

    def set_target_radius(self, radius):
        self.target_radius = radius

    def get_features(self, xy):
        if self.basis_type == "Fourier":
            # Compute Fourier basis features (up to specified order)
            x, y = xy
            x /= self.env_shape[0]
            y /= self.env_shape[1]

            features = []  # bias term make this [1] if you want a bias term
            for f1, f2 in self.frequencies:
                phase = np.pi * (f1 * x + f2 * y)
                features.append(np.cos(phase))
                features.append(np.sin(phase))
            return torch.tensor(features, dtype=torch.float32)
        else:
            raise NotImplementedError(f"Basis type '{self.basis_type}' not implemented.")

    def _freq_norm(self, f1, f2):
        # Euclidean norm in frequency space
        return float(np.hypot(f1, f2))

    def _freq_weight(self, f1, f2):
        r = self._freq_norm(f1, f2)
        # ---- choose ONE scheme below ----
        mode = 'power'
        if mode == "power":
            p = getattr(self, "lowpass_p", 1.0)  # stronger: 1.0–2.0
            return 1.0 / (r ** p) if r > 0 else 1.0
        elif mode == "exp":
            alpha = getattr(self, "lowpass_alpha", 1.0)  # 0.5–1.5 typical
            return float(np.exp(-alpha * max(0.0, r - 1.0)))
        elif mode == "gaussian":
            sigma = getattr(self, "lowpass_sigma", 0.7)  # 0.5–1.0 typical
            return float(np.exp(-((r - 1.0) ** 2) / (2.0 * sigma ** 2)))
        else:
            return 1.0
    def get_features_zscore(self, xy):
        if self.basis_type == "RBF":
            population_rate = [f.pdf((xy)) / f.pdf(f.mean) for f in self.RBF]
            complete_features = np.zeros([self.n_features])
            complete_features[:len(self.RBF)] = np.array(population_rate)
            f = torch.tensor(complete_features, dtype=torch.float32)

        elif self.basis_type == "Fourier":
            x, y = xy
            ex, ey = float(self.env_shape[0]), float(self.env_shape[1])

            # Center around the true grid centre
            x_c = (x / ex) - 0.5
            y_c = (y / ey) - 0.5

            # Size-invariant phase scaling (so env size doesn’t change “wiggliness”)
            size_factor = min(ex, ey)  # or np.sqrt(ex * ey)
            gamma = self.phase_scale * size_factor  # effective phase shrink

            features = []

            for f1, f2 in self.frequencies:
                phase = 2.0 * np.pi * gamma * (f1 * x_c + f2 * y_c)
                # mild low-pass (axes get 1, diagonals ~1/√2 if enabled)
                r = np.hypot(f1, f2)
                w = self._freq_weight(f1, f2)
                features.append(w * np.cos(phase))
                features.append(w * np.sin(phase))

            # Append raw, centered coords (strong anti-fold cue; still linear features)
            #if self.use_raw_xy_channels:
            #    features.append(x_c)
            #    features.append(y_c)

            f = torch.tensor(features, dtype=torch.float32)

        # z-score if stats are available
        if hasattr(self, "feature_mean") and self.feature_mean is not None:
            f = (f - torch.tensor(self.feature_mean, dtype=torch.float32)) / \
                torch.tensor(self.feature_std, dtype=torch.float32)
        return f

    @staticmethod
    def move_in_direction(x, y, theta, distance=1, adaptation_rotation=0):
        true_angle = theta + adaptation_rotation
        x1 = x + distance * np.cos(true_angle)
        y1 = y + distance * np.sin(true_angle)
        return x1, y1

    def get_next_xy(self, a):
        x, y = self.current_xy
        next_x, next_y = self.move_in_direction(x, y, a, adaptation_rotation=self.adaptation_rotation)
        if self.outside_env(next_x, next_y):
            return (x, y)
        return (next_x, next_y)

    def act(self, a):
        next_xy = self.get_next_xy(a)
        reward = self.get_reward(next_xy)
        self.current_xy = next_xy
        done = self.in_terminal_state()
        return next_xy, reward, done

    def get_reward(self, next_xy):
        next_x, next_y = next_xy
        x_r = self.target_xy[0]
        y_r = self.target_xy[1]
        return self.reward_value if self.is_point_within_circle(next_x, next_y, x_r, y_r) else self.penalty_for_miss

    def reset(self):
        self.current_xy = (self.env_shape[0] / 2, self.env_shape[1] / 2)

    def in_terminal_state(self):
        x, y = self.current_xy
        x_r = self.target_xy[0]
        y_r = self.target_xy[1]
        return self.is_point_within_circle(x, y, x_r, y_r)

    def plot_feature_heatmaps(self, resolution=50, use_zscore=True, figsize=None):
        """
        Plot each feature as a heatmap on separate subplots.

        Parameters:
        -----------
        resolution : int
            Grid resolution for the heatmap (default: 50)
        use_zscore : bool
            Whether to use z-scored features (default: True)
        figsize : tuple or None
            Figure size (width, height). If None, automatically determined.
        """
        # Create coordinate grids
        x_coords = np.linspace(0, self.env_shape[0], resolution)
        y_coords = np.linspace(0, self.env_shape[1], resolution)
        X, Y = np.meshgrid(x_coords, y_coords)

        # Compute features for each grid point
        feature_maps = []
        for i in range(resolution):
            row_features = []
            for j in range(resolution):
                xy = (X[i, j], Y[i, j])
                if use_zscore:
                    features = self.get_features_zscore(xy)
                else:
                    features = self.get_features(xy)
                row_features.append(features.numpy())
            feature_maps.append(row_features)

        feature_maps = np.array(feature_maps)  # Shape: (resolution, resolution, n_features)

        # Determine subplot layout
        n_features = self.n_features
        n_cols = int(np.ceil(np.sqrt(n_features)))
        n_rows = int(np.ceil(n_features / n_cols))

        # Set figure size if not provided
        if figsize is None:
            figsize = (4 * n_cols, 3 * n_rows)

        fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)

        # Handle case where there's only one subplot
        if n_features == 1:
            axes = [axes]
        elif n_rows == 1 or n_cols == 1:
            axes = axes.flatten()
        else:
            axes = axes.flatten()

        # Plot each feature
        for feat_idx in range(n_features):
            ax = axes[feat_idx]

            # Extract feature map for this feature
            feature_map = feature_maps[:, :, feat_idx]

            # Create heatmap
            im = ax.imshow(feature_map, cmap='viridis', origin='lower',
                           extent=[0, self.env_shape[1], 0, self.env_shape[0]])

            # Add colorbar
            plt.colorbar(im, ax=ax, shrink=0.8)

            # Set title and labels
            if self.basis_type == "Fourier" and feat_idx < len(self.frequencies) * 2:
                freq_idx = feat_idx // 2
                func_type = "cos" if feat_idx % 2 == 0 else "sin"
                f1, f2 = self.frequencies[freq_idx]
                ax.set_title(f'{func_type}(π({f1}x + {f2}y))')
            else:
                ax.set_title(f'Feature {feat_idx}')

            ax.set_xlabel('Y coordinate')
            ax.set_ylabel('X coordinate')

            # Mark goal location if available
            if hasattr(self, 'target_xy') and self.target_xy is not None:
                goal_x, goal_y = self.target_xy
                ax.scatter(goal_y, goal_x, c='red', s=100, marker='*',
                           edgecolors='white', linewidth=2, label='Goal')
                ax.legend()

        # Hide unused subplots
        for idx in range(n_features, len(axes)):
            axes[idx].set_visible(False)

        plt.tight_layout()
        plt.show()

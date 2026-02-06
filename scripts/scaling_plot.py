import sys
from pathlib import Path

# Add project root to path so imports work from scripts/ directory
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import matplotlib.pyplot as plt
import numpy as np
from core.plotting import set_plotting_defaults

set_plotting_defaults(font_size=13)

state_dim = 14  # your Fourier features
emb_dims = [2, 8]
num_actions_list = np.array([8, 16, 24, 32, 48, 64, 128, 256, 512, 1024, 2048])

# Critic: state_dim -> 1 (same for both)
critic_params = state_dim + 1

# Standard AC: state_dim -> n_actions (linear layer)
standard_ac_actor = state_dim * num_actions_list + num_actions_list
standard_ac_total = standard_ac_actor + critic_params

fig, axes = plt.subplots(1, 2, figsize=(10, 4))

# Left plot: RL-trainable actor parameters
ax = axes[0]
ax.plot(num_actions_list, standard_ac_actor, marker='o', color='#A94850', label='Standard Actor-Critic')

emb_colors = ['#6067B6', '#2E8B57']
for embedding_dim, color in zip(emb_dims, emb_colors):
    embedding_ac_params = np.full_like(num_actions_list, state_dim * embedding_dim + embedding_dim)
    ax.plot(num_actions_list, embedding_ac_params, marker='s', color=color, label=f'Embedding AC (emb_dim={embedding_dim})')

ax.set_xlabel('Action space dimensionality')
ax.set_ylabel('# RL-trainable actor parameters')
ax.set_xscale('log')
ax.set_yscale('log')
ax.set_title('RL-trainable parameters')
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)

# Right plot: Total parameters (actor + critic + f + g)
ax = axes[1]
ax.plot(num_actions_list, standard_ac_total, marker='o', color='#A94850')

for embedding_dim, color in zip(emb_dims, emb_colors):
    # Actor: state_dim -> embedding_dim
    actor_params = state_dim * embedding_dim + embedding_dim
    # g network: (2 * state_dim) -> embedding_dim
    g_params = 2 * state_dim * embedding_dim + embedding_dim
    # f network: embedding_dim -> n_actions
    f_params = embedding_dim * num_actions_list + num_actions_list
    # Total (actor + critic + f + g)
    total_params = actor_params + critic_params + g_params + f_params
    ax.plot(num_actions_list, total_params, marker='s', color=color)

ax.set_xlabel('Action space dimensionality')
ax.set_ylabel('# Total parameters')
ax.set_xscale('log')
ax.set_yscale('log')
ax.set_title('Total parameters')
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)

# Shared legend at the bottom
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc='lower center', ncol=3, frameon=False, bbox_to_anchor=(0.5, -0.05))

plt.tight_layout()
plt.subplots_adjust(bottom=0.2)
plt.savefig('parameter_scaling_comparison.pdf', bbox_inches='tight')
plt.show()

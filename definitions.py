import os
from pathlib import Path

# Project root is the directory containing this file
PROJECT_ROOT = Path(__file__).parent.resolve()

# All paths are now relative to project root
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
FIGURES_DIR = PROJECT_ROOT / "figures"

# Backwards-compatible aliases (used throughout codebase)
data_root = DATA_DIR
full_model_load_path = MODELS_DIR
paper_model_path = MODELS_DIR / "paper"
fig_dir = FIGURES_DIR
initial_learning_fig_dir = FIGURES_DIR
paper_fig_dir = FIGURES_DIR / "paper"

# Create directories if missing
for p in [DATA_DIR, MODELS_DIR, FIGURES_DIR, paper_model_path, paper_fig_dir]:
    p.mkdir(parents=True, exist_ok=True)

# Example model filename (for reference)
trained_policy_model_fn = (
    'fully_trained_policy_model_one_target_seed_5_weight_decay_0.0001_'
    'tanh_policy_mean_target_270_n_actions_24.pth'
)

# Optional W&B API key from environment
WANDB_API_KEY = os.environ.get("WANDB_API_KEY")

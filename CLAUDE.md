# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Reinforcement learning research codebase investigating action embeddings for policy learning, adaptation, and generalization in a continuous reaching task. The core hypothesis: learning low-dimensional action embeddings that capture action consequences enables faster policy adaptation to perturbations (visuomotor rotations).

## Project Structure

```
action-embedding-rl/
├── core/                # Core agent, environment, and training code
├── adaptation/          # Adaptation experiments
├── scripts/             # Training and plotting scripts
├── notebooks/           # Jupyter notebooks for paper figures
├── models/              # Pre-trained model checkpoints (created on first run)
│   └── paper/          # Models for paper figures
├── figures/            # Generated figures
│   └── paper/          # Paper figure outputs
└── data/               # Generated data files
```

## Key Commands

### Setup
```bash
pip install -r requirements.txt
```

### Embedding Learning (Stage 1)
```bash
python scripts/embedding_learning.py --seed 0
```
Trains the f (ActionMapping) and g (ActionEmbeddingPredictor) networks. Must complete before policy training.

### Policy Learning (Stage 2)
```bash
python -m core.actor_critic_with_embedding_continuous_states --seed 0 --target 2.356
```
Trains actor-critic policy using pre-trained embeddings. Target is in radians (2.356 ≈ 135°).

### Batch Experiments
```bash
./scripts/run_seeds_embedding.sh          # Train embeddings for seeds 0-9
./scripts/run_seeds_policy_learning.sh    # Train policies for seeds 0-9 × 24 target angles
```

### Adaptation Experiments
```bash
python -m adaptation.adaptation_exp --seed 0 --reach_angle 135
python -m adaptation.double_adaptation_interleaved
```

## Architecture

### Core Components (core/)

**Agent Networks** (`agent.py`):
- `ActionEmbeddingPredictor` (g): Maps state transitions (s, s') → 2D embedding
- `ActionMapping` (f): Maps embedding → action logits
- `Actor`: Maps state → embedding (policy mean)
- `Critic`: Maps state → value estimate

**Environment** (`continuous_env.py`):
- `ReachTask`: 20×20 grid, agent reaches toward target angle
- 24 discrete actions (uniformly distributed 0-2π)
- Fourier basis state features (order 3 → 14 features)
- Supports visuomotor rotation for adaptation experiments

**Training Loop** (`actor_critic_with_embedding_continuous_states.py`):
- Actor-critic with fixed (fg_lr=0) or plastic embeddings
- W&B logging (optional, set `log_to_wandb: False` in config to disable)

### Experiment Modules

- `adaptation/`: Visuomotor rotation adaptation, double adaptation
- `core/blockwise_two_cues/`: Two-cue blocking paradigm

## Configuration

Each module has a `config.py` with hyperparameters. Key defaults:
```python
actor_lr: 0.00001      # Slow actor learning
critic_lr: 0.0001      # 10x faster critic
fg_lr: 0.0             # Embeddings frozen during policy learning
embedding_dim: 2       # 2D embedding space
num_actions: 24        # Discrete action count
log_to_wandb: False    # W&B logging disabled by default
```

Override via command line: `--seed`, `--target`, `--num_actions`

## Path Configuration

All paths are project-relative (defined in `definitions.py`):
- `models/` - Model checkpoints
- `models/paper/` - Paper-specific models
- `figures/` - Generated figures
- `data/` - Generated data files

Model filename patterns:
- Embeddings: `action_embedding_model_seed_{X}_weight_decay_fg_0.0001_n_action_24_fourier_basis.pth`
- Policies: `fully_trained_policy_model_one_target_seed_{X}_..._target_{Y}_n_actions_24.pth`

## Dependencies

Install via: `pip install -r requirements.txt`

Core: `torch`, `numpy`, `scipy`, `matplotlib`, `seaborn`, `pandas`, `scikit-learn`

Optional: `wandb` (for experiment tracking - set `WANDB_API_KEY` env variable)

## Paper Figures

Jupyter notebooks in `notebooks/`:
- `plot_fig1.ipynb`: Figure 1 - Model architecture and training
- `plot_fig2.ipynb`: Figure 2 - Policy angles
- `plot_fig3.ipynb`: Figure 3 - Adaptation effects
- `plot_fig4.ipynb`: Figure 4 - Double adaptation and interference

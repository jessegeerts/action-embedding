# Action Embeddings for Reinforcement Learning

This repository contains the code to reproduce results from the paper *"Why motor learning involves multiple systems: an algorithmic perspective"* by **Francesca Greenstreet**, **Jesse Geerts**, Juan Gallego and Claudia Clopath [[bioRxiv]](https://www.biorxiv.org/content/10.64898/2025.12.19.695526v1)

## Installation

```bash
# Clone the repository
git clone https://github.com/jessegeerts/action-embedding-rl.git
cd action-embedding-rl

# Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Project Structure

```
action-embedding-rl/
├── core/                    # Core agent, environment, and training code
├── adaptation/              # Adaptation experiments
├── scripts/                 # Training and plotting scripts
├── notebooks/               # Jupyter notebooks for figures
├── models/                  # Trained model checkpoints (created during training)
│   └── paper/              # Models for paper figures
├── figures/                 # Generated figures
│   └── paper/              # Paper figure outputs
└── data/                    # Generated data files
```

## Reproducing Paper Results

Training consists of two stages: (1) learning action embeddings via supervised learning, then (2) training RL policies using those embeddings. The models are shallow networks and train quickly.

### Step 1: Train Action Embeddings

Train the encoder (g) and decoder (f) networks that learn to predict actions from state transitions:

```bash
# Train for a single seed
python scripts/embedding_learning.py --seed 0

# Or train all seeds (0-9) using the batch script
./scripts/run_seeds_embedding.sh
```

Models are saved to `models/paper/`.

### Step 2: Train Policies

Train actor-critic policies using the pre-trained embeddings:

```bash
# Train for a single seed and target angle (in radians)
python -m core.policy_learning --seed 0 --target 2.356

# Or train all seeds × all target angles using the batch script
./scripts/run_seeds_policy_learning.sh
```

Common target angles (in radians):
- 0° = 0.0
- 45° = 0.785
- 90° = 1.571
- 135° = 2.356
- 180° = 3.142

### Step 3: Run Adaptation Experiments (for Figures 3 & 4)

After training base policies, run the adaptation experiments.

**For Figure 3** (single adaptation, requires 10 seeds × 24 target angles = 240 runs):

```bash
# Single run
python -m adaptation.adaptation_exp --seed 0 --reach_angle 135

# Or run all combinations using the batch script
./scripts/run_adaptation_experiments.sh
```

**For Figure 4** (double adaptation, requires 10 seeds × 7 angular separations = 70 runs):

```bash
# Single run (phase2_rel_angle is the angular separation between targets)
python -m adaptation.double_adaptation_interleaved_logging --seed 0 --phase2_rel_angle 180

# Or run all combinations using the batch script
./scripts/run_double_adaptation_experiments.sh
```

### Step 4: Generate Figures

Once training is complete, generate the paper figures using the Jupyter notebooks:

```bash
cd notebooks
jupyter notebook
```

**Note:** Notebooks must be run from the `notebooks/` directory. They automatically add the project root to the Python path.

| Figure | Notebook | Prerequisites |
|--------|----------|---------------|
| Figure 1 | `plot_fig1.ipynb` | Steps 1-2 (embedding + policy training) |
| Figure 2 | `plot_fig2.ipynb` | Step 2 (policy training) |
| Figure 3 | `plot_fig3.ipynb` | Steps 1-3 (all training + adaptation) |
| Figure 4 | `plot_fig4.ipynb` | Steps 1-3 (all training + double adaptation) |

### Generated Data Files

Training scripts save intermediate data files that notebooks use:
- `models/paper/`: Model checkpoints (`.pth` files)
- `figures/paper/`: Training curves and metrics (`.npy`, `.csv` files)

## Citation

If you use this code, please cite:

```bibtex
@article{greenstreet2025motor,
  title={Why motor learning involves multiple systems: an algorithmic perspective},
  author={Greenstreet, Francesca and Geerts, Jesse P and Gallego, Juan A and Clopath, Claudia},
  journal={bioRxiv},
  pages={2025--12},
  year={2025},
  publisher={Cold Spring Harbor Laboratory}
}
```

#!/bin/bash

# Run from project root: ./scripts/run_seeds_embedding.sh

# Array of seeds to run
seeds=(0 1 2 3 4 5 6 7 8 9)

# Run each seed
for seed in "${seeds[@]}"; do
    echo "Running seed $seed"
    python scripts/embedding_learning.py --seed $seed
    echo "Completed seed $seed"
done

echo "All runs completed!"

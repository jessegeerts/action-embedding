#!/bin/bash

# Run from project root: ./scripts/run_adaptation_experiments.sh
# This runs adaptation experiments for Figure 3 (10 seeds × 24 target angles)

seeds=(0 1 2 3 4 5 6 7 8 9)
target_angles=(0 15 30 45 60 75 90 105 120 135 150 165 180 195 210 225 240 255 270 285 300 315 330 345)

for seed in "${seeds[@]}"; do
    for target in "${target_angles[@]}"; do
        echo "Running adaptation: seed $seed, target $target degrees"
        python -m adaptation.adaptation_exp --seed $seed --reach_angle $target
        echo "Completed seed $seed, target $target"
    done
done

echo "All adaptation experiments completed!"

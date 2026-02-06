#!/bin/bash

# Run from project root: ./scripts/run_double_adaptation_experiments.sh
# This runs double adaptation experiments for Figure 4 (10 seeds × 7 angular separations)

seeds=(0 1 2 3 4 5 6 7 8 9)
angular_separations=(0 30 60 90 120 150 180)

for seed in "${seeds[@]}"; do
    for sep in "${angular_separations[@]}"; do
        echo "Running double adaptation: seed $seed, angular separation $sep degrees"
        python -m adaptation.double_adaptation_interleaved_logging \
            --seed $seed \
            --phase2_rel_angle $sep
        echo "Completed seed $seed, separation $sep"
    done
done

echo "All double adaptation experiments completed!"

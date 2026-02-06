#!/bin/bash

# Run from project root: ./scripts/run_seeds_policy_learning.sh

seeds=(0 1 2 3 4 5 6 7 8 9)
target_angles=(0.0 0.26179939 0.52359878 0.78539816 1.04719755 1.30899694 1.57079633 1.83259571 2.0943951 2.35619449 2.61799388 2.87979327 3.14159265 3.40339204 3.66519143 3.92699082 4.1887902 4.45058959 4.71238898 4.97418837 5.23598776 5.49778714 5.75958653 6.02138592)

for seed in "${seeds[@]}"; do
    for target_angle in "${target_angles[@]}"; do
        echo "Running seed $seed with target angle $target_angle"
        python -m core.policy_learning \
            --seed $seed \
            --target $target_angle
        echo "Completed seed $seed with target angle $target_angle"
    done
done

echo "All runs completed!"

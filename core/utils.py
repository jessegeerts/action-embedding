import math

import numpy as np
from scipy.stats import circmean


def calc_circular_shift(a1, a2):
    return np.arctan2(np.sin(a2 - a1), np.cos(a2 - a1))


def circular_shift_pre_post(pre_samples, post_samples):
    """
    Compute the circular shift between two sets of samples in a circular variable.

    Args:
        pre_samples (array-like): Pre-samples (values between 0 and range_max).
        post_samples (array-like): Post-samples (values between 0 and range_max).
        range_max (int): Maximum value of the circular variable (default: 30).

    Returns:
        float: Signed circular shift in the range [-range_max/2, range_max/2].
    """

    # Compute circular means
    pre_mean = circmean(pre_samples, high=2 * np.pi, low=0)
    post_mean = circmean(post_samples, high=2 * np.pi, low=0)

    # Compute circular shift in radians
    shift_rad = calc_circular_shift(pre_mean, post_mean)
    return shift_rad, pre_mean, post_mean


def calculate_distance_from_target(x, y, target_x, target_y):
    distance = math.sqrt((x - target_x) ** 2 + (y - target_y) ** 2)
    return distance


def compute_circular_mean(indices, num_values):
    """
    Compute the circular mean of indices on a circle with num_values positions.
    """
    angles = np.array(indices) * (2 * np.pi / num_values)  # Convert indices to angles
    sin_sum = np.sum(np.sin(angles))
    cos_sum = np.sum(np.cos(angles))
    mean_angle = np.arctan2(sin_sum, cos_sum)  # Circular mean in radians
    if mean_angle < 0:
        mean_angle += 2 * np.pi  # Ensure the angle is in [0, 2*pi)
    return mean_angle * (num_values / (2 * np.pi))  # Convert back to circular index



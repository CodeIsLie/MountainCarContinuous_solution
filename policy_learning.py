import gym
import numpy as np


def init_policy(vel_bins, state_bins):
    # rows - velocity, cols - coordinate
    policy = np.zeros((vel_bins, state_bins))
    for vel in range(vel_bins // 2):
        policy[vel] = np.full(state_bins, -1.0)

    for vel in range(vel_bins // 2, vel_bins):
        policy[vel] = np.full(state_bins, 1.0)

    return policy.astype(np.int)

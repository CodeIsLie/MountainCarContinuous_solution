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


def calc_velocity(old_observation, new_observation):
    sign = 1 if new_observation[0] - old_observation[0] > 0 else -1
    speed = np.sqrt((new_observation[0] - old_observation[0]) ** 2 + (new_observation[1] - old_observation[1]) ** 2)
    return sign * speed

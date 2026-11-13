"""Shared fixed-angle task settings for training and evaluation."""
import gymnasium as gym
import numpy as np

TARGET_ANGLE = np.deg2rad(45.0)
TOLERANCE = 0.1
# Previous experiment used MAX_TORQUE = 5.0; outputs are preserved.
MAX_TORQUE = 2.0


def make_env(**kwargs):
    env = gym.make('Pendulum-v1', **kwargs)
    env.unwrapped.max_torque = MAX_TORQUE
    env.unwrapped.action_space = gym.spaces.Box(
        low=-MAX_TORQUE, high=MAX_TORQUE, shape=(1,), dtype=np.float32)
    return env


def error(obs):
    theta = np.arctan2(obs[1], obs[0])
    delta = theta - TARGET_ANGLE
    return np.arctan2(np.sin(delta), np.cos(delta))


def task_reward(obs, torque):
    return -float(error(obs)**2 + 0.1 * obs[2]**2 + 0.001 * np.asarray(torque).reshape(-1)[0]**2)

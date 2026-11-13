"""Train DDPG on Pendulum-v1 (swing the pendulum up and keep it upright).

  python3 -u main.py [N_EPISODES] > train.log     # default 250 (as in Phil's Pendulum example)

Hyperparameters follow Phil's TensorFlow2 Pendulum example: actor lr 1e-3,
critic lr 2e-3, tau 5e-3, Gaussian noise 0.1, layers 400/300, batch 64.
Unlike that example, actions ARE scaled to the full torque range [-2, 2].
"""
import os
import sys

import gymnasium as gym
import numpy as np

from ddpg_torch import Agent
from utils import plotLearning

os.makedirs('tmp/ddpg', exist_ok=True)

n_games = int(sys.argv[1]) if len(sys.argv) > 1 else 250

# --- Reward shaping: extra reward for being (almost) exactly upright ----------
UPRIGHT_ANGLE = 0.1   # rad (~5.7 deg) from vertical counts as "upright"
UPRIGHT_BONUS = 1.0   # added to the reward on every upright step
# --- Exploration: Gaussian action noise (DDPG's counterpart of DQN's epsilon) -
NOISE = 0.05          # was 0.1 in run 1

env = gym.make('Pendulum-v1')
max_action = float(env.action_space.high[0])  # 2.0

agent = Agent(alpha=0.001, beta=0.002, input_dims=env.observation_space.shape, tau=0.005,
              env=env, batch_size=64, layer1_size=400, layer2_size=300,
              n_actions=env.action_space.shape[0], noise=NOISE)


def angle(obs):
    """obs = [cos th, sin th, th_dot] -> th in (-pi, pi], 0 = upright."""
    return np.arctan2(obs[1], obs[0])


score_history = []
for i in range(n_games):
    obs, _ = env.reset()
    done = False
    score = 0          # ORIGINAL env reward: comparable with run 1 and the -150 target
    upright = 0        # steps spent within UPRIGHT_ANGLE of vertical
    while not done:
        act = agent.choose_action(obs)                     # in [-1, 1]
        new_state, reward, terminated, truncated, info = env.step(act * max_action)
        done = terminated or truncated
        is_up = abs(angle(new_state)) < UPRIGHT_ANGLE
        shaped = reward + (UPRIGHT_BONUS if is_up else 0.0)   # what the agent learns from
        # Pendulum never terminates, it only times out (truncated) after 200 steps,
        # so the bootstrap must NOT be cut off: store `terminated`, not `done`.
        agent.remember(obs, act, shaped, new_state, int(terminated))
        agent.learn()
        score += reward
        upright += is_up
        obs = new_state
    score_history.append(score)

    if i % 10 == 0:
        agent.save_models()

    print('episode ', i, 'score %.2f' % score, 'upright %d/200' % upright,
          'trailing 100 games avg %.3f' % np.mean(score_history[-100:]), flush=True)

agent.save_models()
os.makedirs('plots', exist_ok=True)
plotLearning(score_history, 'plots/pendulum.png', window=100)

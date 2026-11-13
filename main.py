"""Train DDPG on Pendulum-v1 (swing the pendulum up and keep it upright).

  python3 -u main.py                                   # run-3 settings, 250 episodes
  python3 -u main.py --bonus 0 --penalty 0 --seed 1 --out runs/A_s1

Each run writes <out>/train.log and checkpoints to <out>/ckpt/.

Hyperparameters follow Phil's TensorFlow2 Pendulum example: actor lr 1e-3,
critic lr 2e-3, tau 5e-3, layers 400/300, batch 64. Unlike that example,
actions ARE scaled to the full torque range [-2, 2].
"""
import argparse
import os
import random

import gymnasium as gym
import numpy as np
import torch as T

from ddpg_torch import Agent
from target_control import make_env, error, task_reward

p = argparse.ArgumentParser()
p.add_argument('--episodes', type=int, default=250)
p.add_argument('--bonus', type=float, default=1.0, help='reward added on every upright step')
p.add_argument('--penalty', type=float, default=1.0, help='reward subtracted on every NON-upright step')
p.add_argument('--noise', type=float, default=0.05, help='Gaussian exploration noise (DDPG counterpart of epsilon)')
p.add_argument('--seed', type=int, default=0)
# Old upright default: --out .
p.add_argument('--out', default='runs/angle45_torque2_default', help='folder for train.log and ckpt/')
args = p.parse_args()

UPRIGHT_ANGLE = 0.1   # rad (~5.7 deg) from vertical tolerance around the 45-degree target

# Seed everything so a run can be repeated exactly and seeds can be compared
random.seed(args.seed)
np.random.seed(args.seed)      # replay-buffer sampling
T.manual_seed(args.seed)       # network init + exploration noise

ckpt_dir = os.path.join(args.out, 'ckpt')
os.makedirs(ckpt_dir, exist_ok=True)
log = open(os.path.join(args.out, 'train.log'), 'w')

# Old upright environment: env = gym.make('Pendulum-v1')
env = make_env()
max_action = float(env.action_space.high[0])  # 2.0: original actuator limit

agent = Agent(alpha=0.001, beta=0.002, input_dims=env.observation_space.shape, tau=0.005,
              env=env, batch_size=64, layer1_size=400, layer2_size=300,
              n_actions=env.action_space.shape[0], noise=args.noise, chkpt_dir=ckpt_dir)

print(f'# bonus={args.bonus} penalty={args.penalty} noise={args.noise} seed={args.seed}', file=log, flush=True)


def angle(obs):
    """obs = [cos th, sin th, th_dot] -> th in (-pi, pi], 0 = upright."""
    return np.arctan2(obs[1], obs[0])


score_history = []
for i in range(args.episodes):
    obs, _ = env.reset(seed=args.seed * 10_000 + i)
    done = False
    score = 0          # Unshaped 45-degree task reward
    upright = 0        # steps within UPRIGHT_ANGLE of vertical
    reach = None       # first step at which the pendulum got upright (swing-up time)
    effort = 0.0       # control effort: sum of torque^2
    losses, qs = [], []
    errors, velocities, torques = [], [], []
    t = 0
    while not done:
        act = agent.choose_action(obs)                     # in [-1, 1]
        torque = act * max_action
        new_state, reward, terminated, truncated, info = env.step(torque)
        done = terminated or truncated
        t += 1
        # Old upright test: is_up = abs(angle(new_state)) < UPRIGHT_ANGLE
        is_up = abs(error(new_state)) < UPRIGHT_ANGLE
        if is_up and reach is None:
            reach = t
        # what the agent learns from: +bonus when upright, -penalty for every step it isn't
        # Old upright reward: shaped = reward + (args.bonus if is_up else -args.penalty)
        reward = task_reward(obs, torque)
        shaped = reward + (args.bonus if is_up else -args.penalty)
        # Pendulum never terminates, it only times out (truncated) after 200 steps,
        # so the bootstrap must NOT be cut off: store `terminated`, not `done`.
        agent.remember(obs, act, shaped, new_state, int(terminated))
        agent.learn()
        if agent.last_critic_loss is not None:
            losses.append(agent.last_critic_loss)
            qs.append(agent.last_q)
        score += reward
        upright += is_up
        effort += float(torque[0] ** 2)
        errors.append(float(np.degrees(error(new_state))))
        velocities.append(float(new_state[2]))
        torques.append(float(torque[0]))
        obs = new_state
    score_history.append(score)

    if i % 10 == 0:
        agent.save_models()

    print('episode ', i, 'score %.2f' % score, 'near_target %d/200' % upright,
          'reach %d' % (reach if reach is not None else 200),
          'effort %.1f' % effort,
          'closs %.3f' % (np.mean(losses) if losses else 0.0),
          'q %.2f' % (np.mean(qs) if qs else 0.0),
          'error_deg %.3f' % np.mean(np.abs(errors[-100:])),
          'speed %.3f' % np.mean(np.abs(velocities[-100:])),
          'mean_torque %.3f' % np.mean(torques[-100:]),
          'trailing 100 games avg %.3f' % np.mean(score_history[-100:]), file=log, flush=True)

agent.save_models()

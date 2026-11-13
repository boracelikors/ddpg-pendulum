"""Random-policy baseline on Pendulum-v1: what "no learning" looks like.

  python3 random_baseline.py            # mean score over 10 seeded episodes
  python3 random_baseline.py video      # also save random_episode.mp4
"""
import sys

import gymnasium as gym
import imageio
import numpy as np

record = len(sys.argv) > 1 and sys.argv[1] == 'video'
env = gym.make('Pendulum-v1', render_mode='rgb_array' if record else None)
env.action_space.seed(0)

scores, frames = [], []
for seed in range(10):
    obs, _ = env.reset(seed=seed)
    done, score = False, 0.0
    while not done:
        if record and seed == 0:
            frames.append(env.render())
        obs, r, term, trunc, _ = env.step(env.action_space.sample())
        score += r
        done = term or trunc
    scores.append(score)

print('random policy: mean %.1f  (min %.1f, max %.1f)' % (np.mean(scores), min(scores), max(scores)))
if record:
    imageio.mimsave('random_episode.mp4', frames, fps=30, macro_block_size=1)
    print('saved random_episode.mp4')

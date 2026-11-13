"""Evaluate the saved actor without noise, and/or record one episode to MP4.

  python3 record_episode.py eval  [CKPT_DIR] [N]           # mean score over N seeds
  python3 record_episode.py video [CKPT_DIR] [SEED] [OUT]  # save an MP4
  python3 record_episode.py live  [CKPT_DIR]               # window; reloads newest checkpoint
"""
import os
import sys

import gymnasium as gym
import numpy as np
import torch as T

from ddpg_torch import ActorNetwork

T.set_num_threads(1)  # single-sample inference: extra threads only steal CPU from training
mode = sys.argv[1]
ckpt_dir = sys.argv[2] if len(sys.argv) > 2 else 'tmp/ddpg'
MAX_ACTION = 2.0

actor = ActorNetwork(0.001, [3], 400, 300, n_actions=1, name='Actor', chkpt_dir=ckpt_dir)


def load():
    actor.load_state_dict(T.load(actor.checkpoint_file, map_location='cpu'))
    actor.eval()


def run(env, seed=None, frames=None):
    obs, _ = env.reset(seed=seed)
    done, score = False, 0.0
    while not done:
        if frames is not None:
            frames.append(env.render())
        with T.no_grad():
            a = actor(T.tensor(obs, dtype=T.float)).numpy() * MAX_ACTION
        obs, r, term, trunc, _ = env.step(a)
        score += r
        run.upright += abs(np.arctan2(obs[1], obs[0])) < 0.1
        done = term or trunc
    return score


run.upright = 0


if mode == 'eval':
    load()
    n = int(sys.argv[3]) if len(sys.argv) > 3 else 10
    env = gym.make('Pendulum-v1')
    scores = [run(env, s) for s in range(n)]
    print(ckpt_dir, 'mean %.1f' % np.mean(scores), '| upright %.0f%% of steps' % (100 * run.upright / (200 * n)),
          '|', ' '.join('%.0f' % x for x in scores))
elif mode == 'video':
    import imageio
    load()
    seed = int(sys.argv[3]) if len(sys.argv) > 3 else 0
    out = sys.argv[4] if len(sys.argv) > 4 else 'episode.mp4'
    env = gym.make('Pendulum-v1', render_mode='rgb_array')
    frames = []
    score = run(env, seed, frames)
    imageio.mimsave(out, frames, fps=30, macro_block_size=1)
    print(f'seed {seed}: score {score:.1f} -> {out}')
else:  # live
    env = gym.make('Pendulum-v1', render_mode='human')
    last, ep = None, 0
    while True:
        if os.path.exists(actor.checkpoint_file) and os.path.getmtime(actor.checkpoint_file) != last:
            try:
                load(); last = os.path.getmtime(actor.checkpoint_file)
                print('>> new checkpoint loaded', flush=True)
            except Exception:
                pass
        if last is None:
            run(env)  # untrained actor until the first checkpoint exists
        else:
            ep += 1
            print(f'watch episode {ep}: score {run(env):.1f}', flush=True)

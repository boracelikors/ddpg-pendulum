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
from target_control import make_env, error, task_reward, MAX_TORQUE

T.set_num_threads(1)  # single-sample inference: extra threads only steal CPU from training
mode = sys.argv[1]
ckpt_dir = sys.argv[2] if len(sys.argv) > 2 else 'runs/angle45_torque2_default/ckpt'
# Old upright scaling: MAX_ACTION = 2.0
MAX_ACTION = MAX_TORQUE

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
        # Old score used the upright environment reward.
        r = task_reward(obs, a)
        obs, _, term, trunc, _ = env.step(a)
        score += r
        # Old upright test: abs(np.arctan2(obs[1], obs[0])) < 0.1
        run.upright += abs(error(obs)) < 0.1
        done = term or trunc
    return score


run.upright = 0


if mode == 'eval':
    load()
    n = int(sys.argv[3]) if len(sys.argv) > 3 else 10
    env = make_env()
    scores = [run(env, s) for s in range(n)]
    print(ckpt_dir, 'mean %.1f' % np.mean(scores), '| near target %.0f%% of steps' % (100 * run.upright / (200 * n)),
          '|', ' '.join('%.0f' % x for x in scores))
elif mode == 'video':
    import imageio
    load()
    seed = int(sys.argv[3]) if len(sys.argv) > 3 else 0
    out = sys.argv[4] if len(sys.argv) > 4 else 'episode.mp4'
    env = make_env(render_mode='rgb_array')
    frames = []
    score = run(env, seed, frames)
    imageio.mimsave(out, frames, fps=30, macro_block_size=1)
    print(f'seed {seed}: score {score:.1f} -> {out}')
else:  # live
    env = make_env(render_mode='human')
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

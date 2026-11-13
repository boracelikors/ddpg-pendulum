"""Noise-free comparison of saved runs on the same 20 seeds, plus learning-curve plot.

  python3 compare_runs.py run1 run2 run3      # models in tmp/<run>, logs in results/logs/results_<run>.log
"""
import re
import sys

import gymnasium as gym
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch as T

from ddpg_torch import ActorNetwork

T.set_num_threads(1)
runs = sys.argv[1:]
N_SEEDS, UP = 20, 0.1
LABELS = {'run1': 'Run 1: original reward', 'run2': 'Run 2: +1 upright bonus',
          'run3': 'Run 3: + -1 per non-upright step'}
COLORS = {'run1': 'tab:blue', 'run2': 'tab:purple', 'run3': 'tab:red'}

print(f'{"run":6} {"score":>8} {"upright%":>9} {"reach mean":>11} {"reach max":>10} {"never":>6} {"tilt deg":>9}')
for run in runs:
    actor = ActorNetwork(0.001, [3], 400, 300, n_actions=1, name='Actor', chkpt_dir=f'tmp/{run}')
    actor.load_state_dict(T.load(actor.checkpoint_file, map_location='cpu'))
    actor.eval()
    env = gym.make('Pendulum-v1')
    scores, ups, reaches, tilts = [], 0, [], []
    for seed in range(N_SEEDS):
        obs, _ = env.reset(seed=seed)
        score, reach = 0.0, None
        for t in range(200):
            with T.no_grad():
                a = actor(T.tensor(obs, dtype=T.float)).numpy() * 2.0
            obs, r, *_ = env.step(a)
            score += r
            th = np.arctan2(obs[1], obs[0])
            ups += abs(th) < UP
            if reach is None and abs(th) < UP:
                reach = t + 1
            if t >= 100:
                tilts.append(abs(np.degrees(th)))
        scores.append(score)
        reaches.append(reach)
    got = [x for x in reaches if x is not None]
    print(f'{run:6} {np.mean(scores):8.1f} {100 * ups / (200 * N_SEEDS):8.0f}% {np.mean(got):11.1f} '
          f'{max(got):10d} {reaches.count(None):6d} {np.median(tilts):9.2f}')


def load(f):
    E, S, U, R = [], [], [], []
    for line in open(f):
        m = re.match(r'episode\s+(\d+)\s+score\s+(-?[\d.]+)(?:\s+upright\s+(\d+)/200)?(?:\s+reach\s+(\d+))?', line)
        if m:
            E.append(int(m[1])); S.append(float(m[2]))
            U.append(float(m[3]) if m[3] else np.nan); R.append(float(m[4]) if m[4] else np.nan)
    S = np.array(S)
    A = np.array([S[max(0, i - 99):i + 1].mean() for i in range(len(S))])
    return np.array(E), S, A, np.array(U), np.array(R)


def smooth(x, k=10):
    return np.convolve(x, np.ones(k) / k, 'valid')


fig, axes = plt.subplots(1, 3, figsize=(18, 5))
for run in runs:
    e, s, a, u, r = load(f'results/logs/results_{run}.log')
    c, lab = COLORS.get(run, None), LABELS.get(run, run)
    axes[0].plot(e, a, color=c, lw=2.5, label=lab)
    if not np.isnan(u).all():
        axes[1].plot(e[9:], smooth(u), color=c, lw=2.5, label=lab)
    if not np.isnan(r).all():
        axes[2].plot(e[9:], smooth(r), color=c, lw=2.5, label=lab)
axes[0].axhline(-150, color='g', ls='--', label='Target (-150)')
axes[0].set(title='Original env score (100-ep average)', xlabel='Episode', ylim=(-1800, 50))
axes[1].set(title='Upright steps of 200 (10-ep avg)', xlabel='Episode', ylim=(0, 200))
axes[2].set(title='Steps until first upright (10-ep avg, lower = faster)', xlabel='Episode', ylim=(0, 200))
for ax in axes:
    ax.grid(alpha=.3); ax.legend(loc='best')
plt.tight_layout()
plt.savefig('results/plots/results_compare.png', dpi=100)
print('saved results/plots/results_compare.png')

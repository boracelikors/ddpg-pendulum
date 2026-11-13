"""Multi-seed comparison of reward configurations.

  python3 compare_seeds.py            # reads runs/<CONFIG>_s<SEED>/{train.log,ckpt/}

Evaluation: each trained actor is run noise-free on the same 20 held-out start
states. Every metric is reported as mean ± std over the training seeds.
"""
import glob
import os
import re
from collections import defaultdict

import gymnasium as gym
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch as T

from ddpg_torch import ActorNetwork

T.set_num_threads(1)
UP, EVAL_SEEDS, THRESH = 0.1, [100_000 + k for k in range(20)], 150
CONFIGS = {
    'A_original':      ('A: original reward',            'tab:blue'),
    'B_bonus':         ('B: +1 upright bonus',           'tab:purple'),
    'C_bonus_penalty': ('C: +1 bonus, -1 non-upright',   'tab:red'),
}


def evaluate(ckpt_dir):
    actor = ActorNetwork(0.001, [3], 400, 300, n_actions=1, name='Actor', chkpt_dir=ckpt_dir)
    actor.load_state_dict(T.load(actor.checkpoint_file, map_location='cpu'))
    actor.eval()
    env = gym.make('Pendulum-v1')
    m = defaultdict(list)
    for seed in EVAL_SEEDS:
        obs, _ = env.reset(seed=seed)
        score, reach, ups, effort, ths, us = 0.0, None, 0, 0.0, [], []
        for t in range(200):
            with T.no_grad():
                u = float(actor(T.tensor(obs, dtype=T.float)).item() * 2.0)
            obs, r, *_ = env.step([u])
            th = np.degrees(np.arctan2(obs[1], obs[0]))
            score += r; effort += u * u; ths.append(th); us.append(u)
            ups += abs(th) < np.degrees(UP)
            if reach is None and abs(th) < np.degrees(UP):
                reach = t + 1
        ths, us = np.array(ths), np.array(us)
        m['score'].append(score)
        m['upright %'].append(100 * ups / 200)
        m['reach'].append(reach if reach else 200)
        m['never'].append(reach is None)
        m['tilt °'].append(np.median(np.abs(ths[100:])))                 # steady-state error
        m['overshoot °'].append(np.abs(ths[reach:]).max() if reach else 180)  # worst angle after reaching
        m['effort'].append(effort)                                       # sum torque^2 per episode
        m['chatter'].append(np.abs(np.diff(us[100:])).mean())             # mean |Δtorque| while balancing
    out = {k: float(np.mean(v)) for k, v in m.items()}
    out['never'] = int(np.sum(m['never']))
    return out


def load_log(path):
    cols = defaultdict(list)
    pat = re.compile(r'episode\s+(\d+)\s+score\s+(\S+)\s+upright\s+(\d+)/200\s+reach\s+(\d+)\s+'
                     r'effort\s+(\S+)\s+closs\s+(\S+)\s+q\s+(\S+)')
    for line in open(path):
        m = pat.search(line)
        if m:
            for k, v in zip(('ep', 'score', 'upright', 'reach', 'effort', 'closs', 'q'), m.groups()):
                cols[k].append(float(v))
    return {k: np.array(v) for k, v in cols.items()}


def smooth(x, k=10):
    return np.convolve(x, np.ones(k) / k, 'valid')


results, logs = defaultdict(list), defaultdict(list)
for d in sorted(glob.glob('runs/*_s*')):
    cfg = os.path.basename(d).rsplit('_s', 1)[0]
    lg = load_log(os.path.join(d, 'train.log'))
    if len(lg.get('ep', [])) == 0:
        continue
    logs[cfg].append(lg)
    ev = evaluate(os.path.join(d, 'ckpt'))
    sm = smooth(lg['upright'])
    hit = np.where(sm >= THRESH)[0]
    ev['learn ep'] = float(hit[0] + 9) if len(hit) else np.nan   # first ep with 10-ep avg upright >= 150
    ev['final closs'] = float(lg['closs'][-10:].mean())
    ev['final Q'] = float(lg['q'][-10:].mean())
    results[cfg].append(ev)
    print(f'{os.path.basename(d):22} score {ev["score"]:7.1f}  upright {ev["upright %"]:4.0f}%  '
          f'reach {ev["reach"]:5.1f}  tilt {ev["tilt °"]:5.2f}°  learn-ep {ev["learn ep"]}', flush=True)

keys = ['score', 'upright %', 'reach', 'never', 'tilt °', 'overshoot °', 'effort', 'chatter',
        'learn ep', 'final closs', 'final Q']
lines = ['| Metric | ' + ' | '.join(CONFIGS[c][0] for c in CONFIGS if c in results) + ' |',
         '|---|' + '---|' * sum(c in results for c in CONFIGS)]
for k in keys:
    row = []
    for c in CONFIGS:
        if c not in results:
            continue
        v = np.array([r[k] for r in results[c]], dtype=float)
        row.append(f'{np.nanmean(v):.2f} ± {np.nanstd(v):.2f}' if k != 'never' else f'{int(np.nansum(v))} / {20 * len(v)}')
    lines.append(f'| {k} | ' + ' | '.join(row) + ' |')
table = '\n'.join(lines)
print('\n' + table)
open('results_seeds.md', 'w').write(f'Noise-free evaluation on 20 held-out start states; mean ± std over '
                                     f'{len(next(iter(results.values())))} training seeds.\n\n' + table + '\n')

panels = [('score', 'Original env score (100-ep avg)', None), ('upright', 'Upright steps of 200', (0, 200)),
          ('reach', 'Steps until first upright', (0, 200)), ('effort', 'Control effort Σtorque²', None),
          ('closs', 'Critic loss', None), ('q', 'Mean Q estimate', None)]
fig, axes = plt.subplots(2, 3, figsize=(18, 9))
for ax, (k, title, ylim) in zip(axes.flat, panels):
    for c, (lab, col) in CONFIGS.items():
        if c not in logs:
            continue
        n = min(len(lg['ep']) for lg in logs[c])
        if n < 10:
            continue
        if k == 'score':
            curves = np.array([[lg['score'][max(0, i - 99):i + 1].mean() for i in range(n)] for lg in logs[c]])
            x = np.arange(n)
        else:
            curves = np.array([smooth(lg[k][:n]) for lg in logs[c]])
            x = np.arange(9, n)
        ax.plot(x, curves.mean(0), color=col, lw=2.5, label=f'{lab} (n={len(curves)})')
        ax.fill_between(x, curves.min(0), curves.max(0), color=col, alpha=0.15)
    ax.set(title=title, xlabel='Episode')
    if ylim:
        ax.set_ylim(*ylim)
    if k == 'closs':
        ax.set_yscale('log')
    ax.grid(alpha=.3)
axes[0, 0].legend(loc='lower right')
fig.suptitle('Mean over seeds (line) and min-max range (band)', fontsize=12)
plt.tight_layout()
plt.savefig('results_seeds.png', dpi=100)
print('saved results_seeds.png, results_seeds.md')

"""Live learning curves for every run in runs/*/train.log (refreshes every 5 s).

  python3 live_monitor.py
"""
import glob
import os
import re

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation

COLORS = {'A_original': 'tab:blue', 'B_bonus': 'tab:purple', 'C_bonus_penalty': 'tab:red'}
pat = re.compile(r'episode\s+(\d+)\s+score\s+(-?[\d.]+)\s+near_target\s+(\d+)/200\s+reach\s+(\d+)')

fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(17, 5))
fig.canvas.manager.set_window_title('DDPG Pendulum - live training curves')


def update(_):
    for ax in (ax1, ax2, ax3):
        ax.clear()
    progress, seen = [], set()
    for path in sorted(glob.glob('runs/angle45/*_s*/train.log')):
        run = os.path.basename(os.path.dirname(path))
        cfg = run.rsplit('_s', 1)[0]
        rows = [tuple(map(float, m.groups())) for m in map(pat.search, open(path)) if m]
        if not rows:
            continue
        e, s, u, r = map(np.array, zip(*rows))
        avg = np.array([s[max(0, i - 99):i + 1].mean() for i in range(len(s))])
        col = COLORS.get(cfg, 'gray')
        lab = cfg if cfg not in seen else None
        seen.add(cfg)
        ax1.plot(e, avg, color=col, lw=1.8, label=lab)
        if len(u) >= 10:
            k = np.ones(10) / 10
            ax2.plot(e[9:], np.convolve(u, k, 'valid'), color=col, lw=1.5, label=lab)
            ax3.plot(e[9:], np.convolve(r, k, 'valid'), color=col, lw=1.5, label=lab)
        progress.append(int(e[-1]) + 1)
    ax1.axhline(-150, color='g', ls='--')
    ax1.set(title='45-degree score (100-ep avg)', xlabel='Episode', ylim=(-1800, 50))
    ax2.set(title='Steps near 45 degrees of 200 (10-ep avg)', xlabel='Episode', ylim=(0, 200))
    ax3.set(title='Steps until first near_target (10-ep avg)', xlabel='Episode', ylim=(0, 200))
    for ax in (ax1, ax2, ax3):
        ax.grid(alpha=.3)
    ax1.legend(loc='lower right')
    if progress:
        fig.suptitle(f'{len(progress)} runs  |  episode {min(progress)}-{max(progress)} of 250', fontsize=12)


anim = FuncAnimation(fig, update, interval=5000, cache_frame_data=False)
update(0)
plt.show()

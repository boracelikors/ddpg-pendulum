"""Live learning curve: re-reads train.log every few seconds and redraws."""
import re

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

LOGS = {
    'DDPG Pendulum': ('train.log', 'tab:purple'),
}
# parses lines like "episode 0 score -295.17 trailing 100 games avg -295.174"
pat = re.compile(r'episode\s+(\d+)\s+score\s+(-?[\d.]+).*\s(-?[\d.]+)\s*$')

fig, ax = plt.subplots(figsize=(10, 5.5))
fig.canvas.manager.set_window_title('DDPG Pendulum - live training curve')


def read(path):
    eps, scores, avgs = [], [], []
    try:
        with open(path) as f:
            for line in f:
                m = pat.search(line)
                if m:
                    eps.append(int(m[1])); scores.append(float(m[2])); avgs.append(float(m[3]))
    except FileNotFoundError:
        pass
    return eps, scores, avgs


def update(_):
    ax.clear()
    titles = []
    for name, (path, color) in LOGS.items():
        eps, scores, avgs = read(path)
        if not eps:
            continue
        ax.plot(eps, scores, color=color, alpha=0.15)
        ax.plot(eps, avgs, color=color, linewidth=2.5, label=f'{name} (100-ep avg)')
        titles.append(f'{name}: episode {eps[-1]}, avg {avgs[-1]:.0f}')
    ax.axhline(-200, color='green', linestyle='--', label='Good (-200)')
    ax.axhline(0, color='gray', linewidth=0.8)
    ax.set_xlabel('Episode'); ax.set_ylabel('Score')
    ax.set_title('  |  '.join(titles) or 'waiting...', fontsize=10)
    ax.set_ylim(-1800, 50)
    ax.legend(loc='lower right')
    ax.grid(alpha=0.3)


anim = FuncAnimation(fig, update, interval=5000, cache_frame_data=False)
update(0)
plt.show()

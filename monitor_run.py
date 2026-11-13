"""Live learning and control metrics for one training run."""
import sys
from pathlib import Path
import re
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

path = Path(sys.argv[1]) / 'train.log'
fig, axes = plt.subplots(3, 3, figsize=(14, 9))
fig.canvas.manager.set_window_title('DDPG +45 degrees: learning and control')
metrics = [('score', 'Task return'), ('closs', 'Critic loss'), ('q', 'Mean Q (shaped return)'),
           ('near_target', 'Steps near target / 200'), ('reach', 'First target crossing (200 = none)'),
           ('effort', 'Total squared torque'), ('error_deg', 'Final 100 steps: mean absolute error (deg)'),
           ('speed', 'Final 100 steps: mean absolute speed (rad/s)'),
           ('mean_torque', 'Final 100 steps: mean torque (Nm)')]

def update(_):
    rows = []
    if path.exists():
        for line in path.read_text().splitlines():
            if line.startswith('episode'):
                row = dict(re.findall(r'([a-z_]+)\s+(-?[0-9.]+)', line))
                if all(k in row for k, _ in metrics):
                    rows.append(row)
    for ax, (key, title) in zip(axes.flat, metrics):
        ax.clear()
        if rows:
            ax.plot([float(r['episode']) for r in rows], [float(r[key]) for r in rows])
        ax.set(title=title, xlabel='Episode')
        ax.grid(alpha=.3)
    if rows:
        fig.suptitle(f'+45 degree target | episode {int(float(rows[-1]["episode"])) + 1}/250 | {path.parent.name}')
    fig.tight_layout(rect=(0, 0, 1, .95))

anim = FuncAnimation(fig, update, interval=3000, cache_frame_data=False)
update(0)
plt.show()

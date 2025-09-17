import numpy as np
import matplotlib.pyplot as plt
import glob
import os
import time

plt.ion()  # 开启交互模式

npy_files = sorted(glob.glob("*.npy"))

fig, ax = plt.subplots()
line, = ax.plot([])

for fname in npy_files:
    data = np.load(fname)
    line.set_ydata(np.abs(data))
    line.set_xdata(np.arange(len(data)))
    ax.relim()
    ax.autoscale_view()
    ax.set_title(os.path.basename(fname))
    ax.set_xlabel("Sample")
    ax.set_ylabel("Value")
    fig.canvas.draw()
    fig.canvas.flush_events()
    time.sleep(0.04)  # 每个文件显示1秒

plt.ioff()
plt.show()
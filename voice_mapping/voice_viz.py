import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from numpy.lib.stride_tricks import sliding_window_view
from playsound import playsound
from collections import deque
from datetime import date
import librosa.display
import librosa
import os
import subprocess
import glob
import math
from tqdm import tqdm
from itertools import cycle


FR = 128
# FP = 'test.m4a'
FP = 'bhavik.mp3'
BINS = 512
VID_NAME = f'{date.today()}.mov'
SPEED = 5
DIRECTION = 'left'
COLORS = [
    # "#fff700",
    "#00ffee",
    "#26ff00",
    # "#ff00ee",
]

# Use agg if don't want to show img
matplotlib.use('Agg')

# Load file
data_np, sr = librosa.load(FP)

# Setup step
step = -SPEED if DIRECTION == 'left' else SPEED

# Chunk
chunk_size = int(sr / FR)
print(f'Chunk size: {chunk_size:,}')
# Form into windows
chunks = sliding_window_view(data_np, window_shape=BINS)[::chunk_size]
# Format each window to be 0 on the far left
l_adjust = (chunks[:, 0] * np.ones_like(chunks).T).T
chunks = chunks - l_adjust
# Amend each item proportionally along axis 1 to make far right 0
r_adjust = (chunks[:, -1] * np.ones_like(chunks).T).T
r_adjust = r_adjust * np.linspace(0, 1, BINS)
chunks = chunks - r_adjust

# Colors
colors = COLORS
if len(colors) == 0:
    colors = ['#ffffff']
colors = deque(colors)

# Build parts for colored plot
x_size = math.ceil(BINS / (len(colors)))
j = 0
st, en = -x_size, 0
parts = [[st, en]]
while en <= BINS:
    # Increase st and en
    st, en = (j*x_size), ((j+1)*x_size)
    parts.append([st, en])
    j += 1
# parts.append([st, BINS])
parts = np.array(parts)

# Setup initial plot
fig, ax = plt.subplots(1, figsize=(15, 7))
x = np.arange(0, BINS)
ax.set_ylim(data_np.min(), data_np.max())
ax.set_xlim(0, BINS)
plt.setp(ax, xticks=[0, BINS])
plt.show(block=False)

# Start playing sound
# playsound(FP, False)
ax.set_axis_off()
# Loop over chunks and in each loop replot
for i, chunk in tqdm(enumerate(chunks), total=len(chunks), desc='Plotting'):
    # Clear the chart
    for l in ax.lines:
        l.remove()
    # Plot each part
    parts_ = np.clip(parts, 0, BINS)
    for (st, en), c in zip(parts_, cycle(colors)):
        x_part = np.arange(st, en)
        ax.plot(x_part, chunk[st:en], '-', lw=2, color=c)
    fig.canvas.draw()
    fig.canvas.flush_events()
    # Export frame for video
    plt.savefig("./out/file%02d.png" % i, transparent=True)

    # Move parts along
    parts += step
    if abs(parts[1, 0]) >= x_size:
        # Return to zero
        parts += -parts[1, 0]
        # Rotate colors
        colors.rotate(1 if step > 0 else -1)


# Generate video
try:
    os.chdir("./out")
    # Remove any existing file
    if os.path.exists(VID_NAME):
        os.remove(VID_NAME)
    subprocess.call([
        'ffmpeg', '-framerate', str(FR),
        '-i', 'file%02d.png', '-r', str(FR),
        '-i', f'../{FP}', '-ar', str(sr),
        '-pix_fmt', 'yuva420p',
        '-vcodec', 'png',
        f'{VID_NAME}'
    ])
finally:
    # Remove images
    for file_name in glob.glob("*.png"):
        os.remove(file_name)
    os.chdir("../")

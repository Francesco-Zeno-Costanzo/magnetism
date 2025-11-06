"""
code to visualize the spin configurations from CSV files
using a 3D quiver plot and animate over time steps.
"""
import glob
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


# Data
N     = 16  
files = sorted(glob.glob('img/spins_step*.csv'))

# Load data from files
data_list = []
for fname in files:
    data = np.loadtxt(fname, delimiter=',')
    S = np.zeros((N, N, 3))
    for row in data:
        i, j, x, y, z = map(float, row)
        S[int(i), int(j), :] = [x, y, z]
    data_list.append(S)

# Create meshgrid for quiver plot
x = np.arange(N)
y = np.arange(N)
X, Y = np.meshgrid(x, y)
Z = np.zeros_like(X)

# Animation
fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(111, projection='3d')
ax.set_zlim(-1, 1)
ax.set_box_aspect((1, 1, 0.5))
ax.set_xlabel("i")
ax.set_ylabel("j")
ax.set_zlabel("S_z")


S0      = data_list[0]
U, V, W = S0[:, :, 0], S0[:, :, 1], S0[:, :, 2]
quiver  = ax.quiver(X, Y, Z, U, V, W, normalize=True, color='b')

def update(frame):
    ''' Update function for animation
    '''
    global quiver
    quiver.remove() # Remove previous quiver

    S       = data_list[frame]
    U, V, W = S[:, :, 0], S[:, :, 1], S[:, :, 2]
    quiver  = ax.quiver(X, Y, Z, U, V, W, normalize=True, color='b')

    ax.set_title(f"Step {frame}")

    return quiver


ani = FuncAnimation(fig, update, frames=len(data_list), interval=120, blit=False)

plt.show()

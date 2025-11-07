"""
Spin lattice visualization script.
"""

import glob
import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


def load_data(N):
    '''
    Load CSV spin snapshots from ./img directory

    Parameters
    ----------
    N : int
        Size of the lattice
    
    Returns
    -------
    data : list
        List of all configurations
    '''
    files = sorted(glob.glob('img/spins_step*.csv'))
    if not files:
        raise FileNotFoundError("No CSV files found in ./img/")
    
    data = []

    for fname in files:
        raw_data = np.loadtxt(fname, delimiter=',')
        S = np.zeros((N, N, 3))
        for row in raw_data:
            i, j, x, y, z = map(float, row)
            if int(i) < N and int(j) < N:
                S[int(i), int(j), :] = [x, y, z]
        data.append(S)
    
    print("Data Loaded")
    return data


def plot_2d(data, N, save=False):
    '''
    2D quiver animation with m_z as color

    Parameters
    ----------
    data : list
        List of all configurations
    N : int
        Size of the lattice
    '''

    x    = np.arange(N)
    y    = np.arange(N)
    X, Y = np.meshgrid(x, y)

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_xlim(-0.5, N - 0.5)
    ax.set_ylim(-0.5, N - 0.5)
    ax.set_aspect('equal')
    ax.set_xlabel('i')
    ax.set_ylabel('j')

    S0 = data[0]
    U, V, W = S0[:, :, 0], S0[:, :, 1], S0[:, :, 2]
    colors = plt.cm.coolwarm((W + 1) / 2).reshape(-1, 4)
    quiver = ax.quiver(X, Y, U, V, color=colors, pivot='middle', scale=30, headwidth=5, headlength=7)

    def update(frame):
        S = data[frame]
        U, V, W = S[:, :, 0], S[:, :, 1], S[:, :, 2]
        colors = plt.cm.coolwarm((W + 1) / 2).reshape(-1, 4)
        quiver.set_UVC(U, V)
        quiver.set_color(colors)
        ax.set_title(f"Step {frame}")
        return quiver,

    ani = FuncAnimation(fig, update, frames=len(data), blit=False, interval=100)

    if save:
        print("Saving animation to animation_2d.mp4 ...")
        ani.save("animation_2d.mp4", fps=20, dpi=150)
    else:
        plt.show()
    

def plot_3d(data, N, save=False):
    '''
    3D quiver animation

    Parameters
    ----------
    data : list
        List of all configurations
    N : int
        Size of the lattice
    '''
    x    = np.arange(N)
    y    = np.arange(N)
    X, Y = np.meshgrid(x, y)
    Z    = np.zeros_like(X)

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_zlim(-1, 1)
    ax.set_box_aspect((1, 1, 0.5))
    ax.set_xlabel("i")
    ax.set_ylabel("j")
    ax.set_zlabel("S_z")

    S0 = data[0]
    U, V, W = S0[:, :, 0], S0[:, :, 1], S0[:, :, 2]
    quiver = ax.quiver(X, Y, Z, U, V, W, normalize=True, color='b')

    def update(frame):
        nonlocal quiver
        quiver.remove()
        S = data[frame]
        U, V, W = S[:, :, 0], S[:, :, 1], S[:, :, 2]
        quiver = ax.quiver(X, Y, Z, U, V, W, normalize=True, color='b')
        ax.set_title(f"Step {frame}")
        return quiver,

    ani = FuncAnimation(fig, update, frames=len(data), blit=False)
    
    if save:
        print("Saving animation to animation_3d.mp4 ...")
        ani.save("animation_3d.mp4", fps=20, dpi=150)
    else:
        plt.show()

def compare_frames(data, N):
    '''
    Function to visualize a comparison between the initial and
    the final configuration of the spin lattice

    Parameters
    ----------
    data : list
        List of all configurations
    N : int
        Size of the lattice
    '''
    S0 = data[0]
    S1 = data[-1]

    fig, axes = plt.subplots(1, 2, figsize=(14, 9))
    for ax, S, title in zip(axes, [S0, S1], ["Initial", "Final"]):
        U, V, W = S[:, :, 0], S[:, :, 1], S[:, :, 2]
        x    = np.arange(N)
        y    = np.arange(N)
        X, Y = np.meshgrid(x, y)

        colors = plt.cm.coolwarm((W + 1) / 2).reshape(-1, 4)
        
        ax.quiver(X, Y, U, V, color=colors, pivot='middle', scale=30, headwidth=5, headlength=7)
        ax.set_aspect('equal')
        ax.set_title(title)
        ax.set_xlim(-0.5, N - 0.5)
        ax.set_ylim(-0.5, N - 0.5)

    plt.tight_layout()
    plt.show()


def main():

    parser = argparse.ArgumentParser(
        prog="plot_spins",
        description="Visualize spin configurations from CSV files.\n"
                    "Example usage:\n"
                    "  python plot_spins.py -n 32 -m 2d\n"
                    "  python plot_spins.py -n 16 -m 3d\n"
                    "  python plot_spins.py -n 32 -c",
        formatter_class=argparse.RawTextHelpFormatter
    )

    parser.add_argument("-n", "--N", type=int, default=16,
                        help="Grid size (default: 16)")
    
    parser.add_argument("-m", "--mode", choices=["2d", "3d"], default="none",
                        help="Visualization mode: 2d or 3d")
    
    parser.add_argument("-s", "--save", action="store_true",
                    help="Save the animation instead of showing it")
    
    parser.add_argument("-c", "--compare", action="store_true",
                        help="Show comparison of first and last frames (2D only)")

    args = parser.parse_args()

    data_list = load_data(args.N)

    if args.mode == "2d":
        plot_2d(data_list, args.N, save=args.save)
    elif args.mode == "3d":
        plot_3d(data_list, args.N, save=args.save)
    
    if args.compare:
        compare_frames(data_list, args.N)


if __name__ == "__main__":
    main()

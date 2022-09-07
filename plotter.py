import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
from matplotlib import cm
import numpy as np

from data_generator import *


def plot_data(data):
    x = data[:, 0].astype('float')
    z = data[:, 1].astype('float')
    y = data[:, 2].astype('float')
    plt.plot(x, y)
    plt.plot(z, y)
    plt.show()


def plot_3d_data(data):
    xs = data[:, 0].astype('float')
    ys = data[:, 1].astype('float')
    zs = data[:, 2].astype('float')
    ax3d = plt.axes(projection="3d")
    x, z = np.meshgrid(xs, ys)
    ax3d.plot3D(xs, ys, zs)

    plt.show()


if __name__ == '__main__':
    plot_3d_data(read_data("data/noisy_sine_2D.dat"))

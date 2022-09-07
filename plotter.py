import matplotlib.pyplot as plt
import numpy as np
from data_generator import *


def plot_data(data):
    x = data[:, 0].astype('float')
    z = data[:, 1].astype('float')
    y = data[:, 2].astype('float')
    plt.plot(x, y)
    plt.plot(z, y)


def plot_3d_data(data):
    xs = data[:, 0].astype('float')
    ys = data[:, 1].astype('float')
    zs = data[:, 2].astype('float')
    ax3d = plt.axes(projection="3d")
    ax3d.plot3D(xs, ys, zs, "kx")


def plot_3d_surface(features, labels):
    x_plot = features[:, 0].reshape(-1, 1)
    y_plot = features[:, 1].reshape(-1, 1)
    z_plot = labels

    ax3d = plt.axes(projection="3d")
    ax3d.plot_surface(x_plot, y_plot, z_plot)


if __name__ == '__main__':
    DATA_LENGTH = 500
    AMPLITUDE = 1
    SIGNAL_NOISE_RATIO = 0

    # x_linspace = np.linspace(0, 2 * np.pi, DATA_LENGTH)
    x, y = read_data("data/linear_sine_2D.dat", 2)
    # plot_3d_surface(x, y)

    plot_3d_data(np.concatenate((x, y), axis=1))
    plt.show()

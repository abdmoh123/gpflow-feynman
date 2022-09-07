import numpy as np
from matplotlib import pyplot as plt


# generates a sine wave with optional noise
def generate_sine_wave(x, amplitude=1, s_n_ratio=0):
    noise = gen_noise(x.shape, s_n_ratio)
    return (amplitude * np.sin(x)) + noise


def generate_cos_wave(x, amplitude=1, s_n_ratio=0):
    return generate_sine_wave(x + (3/2 * np.pi), amplitude=amplitude, s_n_ratio=s_n_ratio)


# generates a square wave with optional noise
def generate_square_wave(x, squareness=1000, amplitude=1, s_n_ratio=0):
    y = temp = x
    for _ in range(squareness):
        y = generate_sine_wave(temp, amplitude=1)
        temp = y
    # scales wave to have the correct amplitude
    y = (y / np.max(y)) * amplitude
    noise = gen_noise(x.size, s_n_ratio)
    return y + noise


# generates random array between 1 and -1 multiplied by noise strength
def gen_noise(array_shape, s_n_ratio):
    if s_n_ratio <= 0:
        noise = np.zeros(array_shape)
    else:
        noise = (2 * np.random.random(array_shape) - 1) / s_n_ratio
    return noise


def write_data(data, file_name):
    np.set_printoptions(suppress=True)
    # writes the data into a csv file
    np.savetxt("./data/" + file_name + ".dat", data, delimiter=' ')


def read_data(directory):
    return np.genfromtxt(directory, delimiter=' ', dtype='<U')


if __name__ == '__main__':
    DATA_LENGTH = 200
    AMPLITUDE = 1
    SIGNAL_NOISE_RATIO = 10

    axis = np.linspace(0, 2 * np.pi, DATA_LENGTH)
    # sine_wave = generate_sine_wave(x_axis, AMPLITUDE, SIGNAL_NOISE_RATIO)
    # square_wave = generate_square_wave(x_axis, amplitude=AMPLITUDE, s_n_ratio=SIGNAL_NOISE_RATIO)
    #
    # write_data(x_axis, sine_wave, "noisy_sine")
    # write_data(x_axis, square_wave, "noisy_square")
    X, Y = np.meshgrid(axis, axis)
    input_axis = (X + Y) / 2
    Z = generate_sine_wave(input_axis, AMPLITUDE, SIGNAL_NOISE_RATIO)

    ax3d = plt.axes(projection="3d")
    ax3d.plot_surface(X, Y, Z)
    plt.show()

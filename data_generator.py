import numpy as np


# generates a sine wave with optional noise
def generate_sine_wave(x, amplitude=1, s_n_ratio=0):
    noise = gen_noise(x.size, s_n_ratio)
    return (amplitude * np.sin(x)) + noise


def generate_2D_sine(x0, x1, amplitude=1, s_n_ratio=0):
    return amplitude * np.sin(x0, x1)


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
def gen_noise(array_size, s_n_ratio):
    if s_n_ratio <= 0:
        noise = np.zeros(array_size)
    else:
        noise = (2 * np.random.random(array_size) - 1) / s_n_ratio
    return noise


def write_data(features, labels, file_name):
    np.set_printoptions(suppress=True)
    # combines features and labels into 1 matrix
    data = np.concatenate((features, labels)).reshape((-1, 2), order='F')
    # writes the data into a csv file
    np.savetxt("./data/" + file_name + ".dat", data, delimiter=' ')


def read_data(directory):
    return np.genfromtxt(directory, delimiter=' ', dtype='<U')


if __name__ == '__main__':
    DATA_LENGTH = 500
    AMPLITUDE = 1
    SIGNAL_NOISE_RATIO = 10

    x_axis = np.linspace(0, 2 * np.pi, DATA_LENGTH)
    sine_wave = generate_sine_wave(x_axis, AMPLITUDE, SIGNAL_NOISE_RATIO)
    square_wave = generate_square_wave(x_axis, amplitude=AMPLITUDE, s_n_ratio=SIGNAL_NOISE_RATIO)

    write_data(x_axis, sine_wave, "noisy_sine")
    write_data(x_axis, square_wave, "noisy_square")

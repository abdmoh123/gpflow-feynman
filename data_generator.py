import numpy as np


# generates a sine wave with optional noise
def generate_sine_wave(x, amplitude=1, noise_amplitude=0):
    noise = gen_noise(x.shape, noise_amplitude)
    return (amplitude * np.sin(x)) + noise


def generate_cos_wave(x, amplitude=1, noise_amplitude=0):
    return generate_sine_wave(x + (3/2 * np.pi), amplitude, noise_amplitude)


# generates a square wave with optional noise
def generate_square_wave(x, squareness=1000, amplitude=1, noise_amplitude=0):
    y = temp = x
    for _ in range(squareness):
        y = generate_sine_wave(temp, amplitude=1)
        temp = y
    # scales wave to have the correct amplitude
    y = (y / np.max(y)) * amplitude
    noise = gen_noise(x.shape, noise_amplitude)
    return y + noise


# generates random array between 1 and -1 multiplied by noise strength
def gen_noise(array_shape, noise_amplitude):
    return noise_amplitude * (2 * np.random.random(array_shape) - 1)


# generates data growing linearly in the 0-th dimension and oscillating in the 1-st dimension
def additive_sine_2d(amplitude=1, noise_amplitude=0, data_length=20):
    x0, x1 = np.meshgrid(np.linspace(0, 2 * np.pi, data_length), np.linspace(0, 2 * np.pi, data_length))
    y = np.sqrt(x0) + 0.5 * generate_sine_wave(x1, amplitude=amplitude) + gen_noise(x0.shape, noise_amplitude)

    def as_column(arr: np.ndarray) -> np.ndarray:
        # returns array as a column vector
        return arr.reshape((-1, 1))

    return np.concatenate((as_column(x0), as_column(x1), as_column(y)), axis=1)


def write_data(data, file_name, delimiter=' '):
    np.set_printoptions(suppress=True)
    # writes the data into a csv file
    np.savetxt("./data/" + file_name + ".dat", data, delimiter=delimiter)


# reads data (as type float) and separates features and labels
def read_data(directory, num_features=1):
    data = read_file(directory)
    X = data[:, :num_features].astype(float)
    Y = data[:, num_features:].astype(float)
    return X, Y


def read_file(directory, delimiter=' '):
    return np.genfromtxt(directory, delimiter=delimiter, dtype='<U')


if __name__ == '__main__':
    # constants to control experiment
    DATA_LENGTH = 20
    AMPLITUDE = 1
    NOISE_AMPLITUDE = 0.2

    # x_0 = np.linspace(0, 2 * np.pi, DATA_LENGTH)
    # x_1 = (x_0 * 2) - np.mean(x_0)
    # sine_wave = generate_sine_wave(x_0 + x_1, AMPLITUDE, SIGNAL_NOISE_RATIO)
    # data_to_write = np.stack((x_0, x_1, sine_wave), axis=1)
    # write_data(data_to_write, "noisy_sine_2D")
    # square_wave = generate_square_wave(x_axis, amplitude=AMPLITUDE, s_n_ratio=SIGNAL_NOISE_RATIO)
    # write_data(x_axis, square_wave, "noisy_square")
    lin_sine_data = additive_sine_2d(AMPLITUDE, NOISE_AMPLITUDE, DATA_LENGTH)
    write_data(lin_sine_data, "linear_sine_2D")

# Press Ctrl+F8 to toggle the breakpoint.
import numpy as np
from aifeynman import run_aifeynman
import torch.cuda
from data_generator import *
import matplotlib.pyplot as plt
import tensorflow as tf
import feynman_mean_function
import gpflow
from gpflow.utilities import print_summary
from sklearn.metrics import mean_squared_error, r2_score


def main():
    # constants to control experiment
    DATA_LENGTH = 500
    AMPLITUDE = 1
    SIGNAL_NOISE_RATIO = 10
    TRAIN_TEST_RATIO = 0.2
    NUM_SAMPLES = 10
    # FILE_NAME = "noisy_sine"
    FILE_NAME = "noisy_square"

    # disables PyTorch GPU support
    torch.cuda.is_available = lambda: False

    np.random.seed(1)  # for reproducibility
    # generates data into a np array
    X = np.linspace(0, 2 * np.pi, DATA_LENGTH)
    if FILE_NAME == "noisy_sine":
        Y = generate_sine_wave(X, amplitude=AMPLITUDE, s_n_ratio=SIGNAL_NOISE_RATIO)
    elif FILE_NAME == "noisy_square":
        Y = generate_square_wave(X, amplitude=AMPLITUDE, s_n_ratio=SIGNAL_NOISE_RATIO)
    else:
        Y = read_data("./data/"+FILE_NAME)[1]
    # randomly samples from the dataset
    data = np.random.permutation(np.concatenate((X, Y)).reshape((-1, 2), order='F'))
    training_data = data[:np.ceil(TRAIN_TEST_RATIO*DATA_LENGTH).astype(int), :]
    training_data = training_data[training_data[:, 0].argsort()]
    X_TRAIN = training_data[:, 0].reshape(-1, 1)
    Y_TRAIN = training_data[:, 1].reshape(-1, 1)

    # saves data to a csv file so AI feynman can run
    write_data(X_TRAIN, Y_TRAIN, FILE_NAME)
    # runs AI feynman on same data
    # run_aifeynman("./data/", FILE_NAME+".dat", 30, "14ops.txt", polyfit_deg=3, NN_epochs=100)

    # reads and converts the AI feynman solution/equation to an evaluable format
    feynman_solution_array = read_data("./results/solution_"+FILE_NAME+".dat")
    shape = np.array(feynman_solution_array.shape) - 1
    feynman_solution = convert_feynman(str(feynman_solution_array[shape[0], shape[1]]))
    print(feynman_solution)

    # sets up the kernel and GP model
    kernel = gpflow.kernels.RBF()
    kernel2 = gpflow.kernels.RBF()
    mean_function = feynman_mean_function.FeynmanMean(feynman_solution)
    model_with_feynman = gpflow.models.GPR(data=(X_TRAIN, Y_TRAIN), kernel=kernel, mean_function=mean_function)
    model_without_feynman = gpflow.models.GPR(data=(X_TRAIN, Y_TRAIN), kernel=kernel2, mean_function=None)
    print("Before optimisation\n============================================\nWith AI Feynman as mean function:")
    print_summary(model_with_feynman)
    print("With NO mean function:")
    print_summary(model_without_feynman)

    # optimises the kernel and GP model
    optimiser = gpflow.optimizers.Scipy()
    optimiser.minimize(model_with_feynman.training_loss, model_with_feynman.trainable_variables)
    optimiser.minimize(model_without_feynman.training_loss, model_without_feynman.trainable_variables)
    print("\nAfter optimisation\n============================================\nWith AI Feynman as mean function:")
    print_summary(model_with_feynman)
    print("With NO mean function:")
    print_summary(model_without_feynman)

    # calculates the mean and variance of the GP as well as a few samples
    X_TEST = np.linspace(-1 * np.pi, 3 * np.pi, DATA_LENGTH * 2).reshape(-1, 1)
    # GPR with AI feynman solution as mean function
    mean_ai, var_ai = model_with_feynman.predict_y(X_TEST)
    samples_ai = model_with_feynman.predict_f_samples(X_TEST, NUM_SAMPLES)
    # calculates Y values using AI feynman's solution
    feynman_prediction = eval(feynman_solution)
    # GPR without AI feynman
    mean, var = model_without_feynman.predict_y(X_TEST)
    samples = model_without_feynman.predict_f_samples(X_TEST, NUM_SAMPLES)

    # generates the original data without noise
    if FILE_NAME == "noisy_sine":
        ground_truth = generate_sine_wave(X_TEST.reshape(1, -1)[0], amplitude=AMPLITUDE)
    elif FILE_NAME == "noisy_square":
        ground_truth = generate_square_wave(X_TEST.reshape(1, -1)[0], amplitude=AMPLITUDE)
    else:
        print("Error: Invalid file name!")
        exit()
    feyn_rmse = mean_squared_error(ground_truth, feynman_prediction)
    feyn_R2 = r2_score(ground_truth, feynman_prediction)
    GPRAI_rmse = mean_squared_error(ground_truth, mean_ai)
    GPRAI_R2 = r2_score(ground_truth, mean_ai)
    GPR_rmse = mean_squared_error(ground_truth, mean)
    GPR_R2 = r2_score(ground_truth, mean)
    print("\nMetrics (RMS error | R2 score)\n============================================")
    print("AI Feynman:\n" + str(feyn_rmse), "|", str(100 * feyn_R2) + "%")
    print("GPR aided by AI Feynman:\n" + str(GPRAI_rmse), "|", str(100 * GPRAI_R2) + "%")
    print("GPR without using AI Feynman:\n" + str(GPR_rmse), "|", str(100 * GPR_R2) + "%")

    # plots training data, predicted output, margin of error (2 std) and some GP samples
    fig = plt.figure()
    fig.suptitle("Graphs comparing GP regression with and without AI Feynman as mean function", fontweight="bold")
    ax = fig.add_subplot(111)
    ax.spines['top'].set_color('none')
    ax.spines['bottom'].set_color('none')
    ax.spines['left'].set_color('none')
    ax.spines['right'].set_color('none')
    ax.tick_params(labelcolor='w', top=False, bottom=False, left=False, right=False)
    ax.set_xlabel("X axis (radians based)")
    ax.set_ylabel("Y axis")
    axes = fig.subplots(2, 2, sharex=True, sharey=False)
    # GP regression with AI feynman
    axes[0, 0].plot(X_TRAIN, Y_TRAIN, "kx", label="Training data")
    axes[0, 0].fill_between(
        X_TEST[:, 0],
        mean_ai[:, 0] - (2 * np.sqrt(var_ai[:, 0])),
        mean_ai[:, 0] + (2 * np.sqrt(var_ai[:, 0])),
        alpha=0.2,
        label="2 std confidence"
    )
    axes[0, 0].plot(X_TEST, samples_ai[:, :, 0].numpy().T, linewidth=0.5)
    axes[0, 0].plot(X_TEST, mean_ai, color="tab:cyan", lw=2, label="Mean aided by AI Feynman")
    # AI feynman on its own
    axes[0, 1].plot(X_TEST, feynman_prediction, color="tab:cyan", label="AI Feynman solution")
    # GP regression without AI feynman
    axes[1, 0].plot(X_TRAIN, Y_TRAIN, "kx", label="Training data")
    axes[1, 0].fill_between(
        X_TEST[:, 0],
        mean[:, 0] - (2 * np.sqrt(var[:, 0])),
        mean[:, 0] + (2 * np.sqrt(var[:, 0])),
        alpha=0.2,
        label="2 std confidence"
    )
    axes[1, 0].plot(X_TEST, samples[:, :, 0].numpy().T, linewidth=0.5)
    axes[1, 0].plot(X_TEST, mean, color="tab:cyan", lw=2, label="Mean without AI Feynman")
    # Original data
    axes[1, 1].plot(X.reshape(DATA_LENGTH, 1), Y.reshape(DATA_LENGTH, 1), color="tab:cyan", label="Original data")
    for i in range(len(axes)):
        for j in range(len(axes[i])):
            axes[i, j].legend()
    plt.show()


# converts the equation created by AI feynman to an evaluable format
def convert_feynman(solution):
    solution = solution.replace("sin", "np.sin")
    solution = solution.replace("cos", "np.cos")
    solution = solution.replace("tan", "np.tan")
    solution = solution.replace("pi", "np.pi")
    solution = solution.replace("log", "np.log")
    solution = solution.replace("x0", "X_TEST")
    return solution


if __name__ == '__main__':
    main()

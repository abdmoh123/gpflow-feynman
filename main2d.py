# Press Ctrl+F8 to toggle the breakpoint.
import numpy as np
from aifeynman import run_aifeynman
import torch.cuda

import plotter
from data_generator import *
import matplotlib.pyplot as plt
import tensorflow as tf
import feynman_mean_function
import gpflow
from gpflow.utilities import print_summary
from sklearn.metrics import mean_squared_error, r2_score


def sample_data(x, y, percentage):
    np.random.seed(1)  # for reproducibility

    # randomises the order of each data point (so it can be randomly sampled)
    data = np.concatenate((x, y), axis=1)
    randomised_data = np.random.permutation(data)
    data_length = randomised_data.shape[0]

    # samples training data and puts it back in the original order (ascending)
    training_data = randomised_data[:np.ceil(percentage * data_length).astype(int), :]
    training_data = training_data[training_data[:, 0].argsort()]
    X_TR = training_data[:, :x.shape[1]].reshape(-1, x.shape[1])
    Y_TR = training_data[:, x.shape[1]:].reshape(-1, y.shape[1])
    return X_TR, Y_TR


def get_feynman_solution(FILE_NAME):
    # runs AI feynman on same data
    # run_aifeynman("./data/", FILE_NAME+"train.dat", 30, "14ops.txt", polyfit_deg=3, NN_epochs=100)

    # reads and converts the AI feynman solution/equation to an evaluable format
    feynman_solution_array = read_file("./results/solution_" + FILE_NAME + "train.dat")
    shape = np.array(feynman_solution_array.shape) - 1

    # function to convert the equation created by AI feynman to an evaluable format
    def convert_feynman(solution):
        solution = solution.replace("x0", "temp_X[:, 0]")
        solution = solution.replace("x1", "temp_X[:, 1]")
        solution = solution.replace("sin", "np.sin")
        solution = solution.replace("asin", "np.arcsin")
        solution = solution.replace("cos", "np.cos")
        solution = solution.replace("acos", "np.arccos")
        solution = solution.replace("tan", "np.tan")
        solution = solution.replace("atan", "np.arctan")
        solution = solution.replace("pi", "np.pi")
        solution = solution.replace("log", "np.log")
        solution = solution.replace("sqrt", "np.sqrt")
        solution = solution.replace("exp", "np.exp")
        return solution

    return convert_feynman(str(feynman_solution_array[shape[0], shape[1]]))


def main():
    NUM_FEATURES = 2
    TRAIN_PERCENTAGE = 0.2
    NUM_SAMPLES = 10
    FILE_NAME = "linear_sine_2D"
    # FILE_NAME = "noisy_square_2D"

    # disables PyTorch GPU support
    torch.cuda.is_available = lambda: False

    # randomly samples from the dataset
    X, Y = read_data("data/"+FILE_NAME+".dat", NUM_FEATURES)
    X_TR, Y_TR = sample_data(X, Y, TRAIN_PERCENTAGE)
    training_data = np.concatenate((X_TR, Y_TR), axis=1)

    # saves data to a csv file so AI feynman can run
    write_data(training_data, FILE_NAME+"train")
    # finds and prints the solution produced by AI Feynman
    feynman_solution = get_feynman_solution(FILE_NAME)
    print("AI Feynman solution:\n============================================\n" + feynman_solution)

    # sets up the kernel and GP model
    kernel = gpflow.kernels.RBF()
    kernel2 = gpflow.kernels.RBF()
    mean_function = feynman_mean_function.FeynmanMean(feynman_solution)
    model_with_feynman = gpflow.models.GPR(data=(X_TR, Y_TR), kernel=kernel, mean_function=mean_function)
    model_without_feynman = gpflow.models.GPR(data=(X_TR, Y_TR), kernel=kernel2, mean_function=None)
    print("\nBefore optimisation\n============================================\nWith AI Feynman as mean function:")
    print_summary(model_with_feynman)
    print("With NO mean function:")
    print_summary(model_without_feynman)

    # optimises the kernel and GP model
    optimiser = gpflow.optimizers.Scipy()
    optimiser.minimize(model_with_feynman.training_loss, model_with_feynman.trainable_variables, options=dict(maxiter=1000))
    optimiser.minimize(model_without_feynman.training_loss, model_without_feynman.trainable_variables, options=dict(maxiter=1000))
    print("\nAfter optimisation\n============================================\nWith AI Feynman as mean function:")
    print_summary(model_with_feynman)
    print("With NO mean function:")
    print_summary(model_without_feynman)

    # calculates the mean and variance of the GP as well as a few samples
    AMPLITUDE = 1
    TEST_DATA_LENGTH = 20
    # GPR with AI feynman solution as mean function
    mean_ai, var_ai = model_with_feynman.predict_y(X)
    samples_ai = model_with_feynman.predict_f_samples(X, NUM_SAMPLES)
    # calculates Y values using AI feynman's solution
    temp_X = X
    feynman_prediction = eval(feynman_solution)
    # GPR without AI feynman
    mean, var = model_without_feynman.predict_y(X)
    samples = model_without_feynman.predict_f_samples(X, NUM_SAMPLES)

    # generates the original data without noise
    ground_truth = additive_sine_2d(AMPLITUDE, data_length=TEST_DATA_LENGTH)[:, 2]
    # calculates metrics for evaluating and comparing the models
    feyn_rmse = mean_squared_error(ground_truth, feynman_prediction)
    feyn_R2 = r2_score(ground_truth, feynman_prediction)
    GPRAI_rmse = mean_squared_error(ground_truth, mean_ai)
    GPRAI_R2 = r2_score(ground_truth, mean_ai)
    GPR_rmse = mean_squared_error(ground_truth, mean)
    GPR_R2 = r2_score(ground_truth, mean)
    # prints calculated metrics
    print("\nMetrics (RMS error | R2 score)\n============================================")
    print("AI Feynman:\n" + str(feyn_rmse), "|", str(100 * feyn_R2) + "%")
    print("GPR aided by AI Feynman:\n" + str(GPRAI_rmse), "|", str(100 * GPRAI_R2) + "%")
    print("GPR without using AI Feynman:\n" + str(GPR_rmse), "|", str(100 * GPR_R2) + "%")

    # plots data
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
    ax3d1 = fig.add_subplot(221, projection="3d")
    ax3d2 = fig.add_subplot(222, projection="3d")
    ax3d3 = fig.add_subplot(223, projection="3d")
    ax3d4 = fig.add_subplot(224, projection="3d")

    ax3d1.plot3D(X[:, 0], X[:, 1], mean_ai[:, 0], "kx", label="GP with Feynman")
    ax3d2.plot3D(X[:, 0], X[:, 1], mean[:, 0], "kx", label="GP without Feynman")
    ax3d3.plot3D(X[:, 0], X[:, 1], feynman_prediction, "kx", label="Feynman solution")
    ax3d4.plot3D(X[:, 0], X[:, 1], Y[:, 0], "kx", label="Original data")
    ax3d1.legend()
    ax3d2.legend()
    ax3d3.legend()
    ax3d4.legend()
    plt.show()


if __name__ == '__main__':
    main()

from gpflow.mean_functions import MeanFunction
import tensorflow as tf
from gpflow.base import Module, Parameter, TensorType
import numpy as np


# uses AI Feynman's solution as a mean function for GPR and other similar models
class FeynmanMean(MeanFunction):
    def __init__(self, equation):
        MeanFunction.__init__(self)
        self.equation = equation

    def __call__(self, X: TensorType) -> tf.Tensor:
        X_TEST = np.array(X)
        return eval(self.equation)

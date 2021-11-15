import math

import numpy as np


# Here's a good way to visualize the math functions in a jupyter notebook
# %matplotlib inline
# import matplotlib.pyplot as plt
# plt.style.use('seaborn-whitegrid')
#
# f = linear
#
# x = np.linspace(0, 1, 1000)
# plt.plot(x, f(x));

def make_scaled_sigmoid(slope):
    """slope at 1/2 (prior to being scaled) - recommended slope > 5"""

    slope_2 = slope * 2

    def sigmoid(x):
        return (2 / (1 + math.exp(-slope_2 * (x - 0.5)))) - 1

    factor = 1 / sigmoid(1)
    factor_2 = factor * 2

    @np.vectorize
    def sigmoid(x):
        return (factor_2 / (1 + math.exp(-slope_2 * (x - 0.5)))) - factor

    return sigmoid


def scaled_double_sigmoid(slope):
    """slope at 1/3 and 2/3 (prior to being scaled) - recommended slope > 30"""

    def double_sigmoid(x):
        return (1 / (1 + math.exp(-slope * (x - 0.666667)))) + (1 / (1 + math.exp(-slope * (x - 0.33333)))) - 1

    factor = 1 / double_sigmoid(1)

    @np.vectorize
    def double_sigmoid(x):
        return (factor / (1 + math.exp(-slope * (x - 0.666667)))) + (
                    factor / (1 + math.exp(-slope * (x - 0.33333)))) - factor

    return double_sigmoid


@np.vectorize
def discrete(x):
    if x > 0.5:
        return 1
    else:
        return -1


@np.vectorize
def double_discrete(x):
    if x > 0.666666666667:
        return 1
    elif x < 0.333333333333:
        return -1
    else:
        return 0
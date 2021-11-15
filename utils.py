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


class SynapseConnectionFunctions:
    """\
    SynapseConnectionFunctions are applied to the synapses as a last step before the synaptic_connection matrix is returned.

    SynapseConnectionFunctions all must have a domain of 0..1 that maps to a range of -1..1 and must monotonically increase.
    """
    @classmethod
    def make_scaled_sigmoid(cls, slope):
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

    @classmethod
    def make_scaled_double_sigmoid(cls, slope):
        """slope at 1/3 and 2/3 (prior to being scaled) - recommended slope > 30"""

        def double_sigmoid(x):
            return (1 / (1 + math.exp(-slope * (x - 0.666667)))) + (1 / (1 + math.exp(-slope * (x - 0.33333)))) - 1

        factor = 1 / double_sigmoid(1)

        @np.vectorize
        def double_sigmoid(x):
            return (factor / (1 + math.exp(-slope * (x - 0.666667)))) + (
                        factor / (1 + math.exp(-slope * (x - 0.33333)))) - factor

        return double_sigmoid

    @classmethod
    @np.vectorize
    def discrete(cls, x):
        if x > 0.5:
            return 1
        else:
            return -1

    @classmethod
    @np.vectorize
    def double_discrete(cls, x):
        if x > 0.666666666667:
            return 1
        elif x < 0.333333333333:
            return -1
        else:
            return 0

    @classmethod
    @np.vectorize
    def linear(cls, x):
        return 2*(x-0.5)


class NeuronActivationFunctions:
    """\
    NeuronActivationFunctions are applied element-wise to the output of SynapseConnect@NeuronActivations such that:

    `NextNeuronActivations - NeuronActivationFunctions(SynapseConnect @ PreviousNeuronActivations)`

    NeuronActivationFunctions have a domain and range of all real numbers. The mapping must monotonically increase.
    """
    @classmethod
    def make_linear(cls, domain_min, domain_max, range_min, range_max):
        assert domain_min < domain_max
        assert range_min < range_max

        domain_width = domain_max - domain_min
        range_height = range_max - range_min
        factor = range_height/domain_width
        diff = range_min - domain_min*factor

        @np.vectorize
        def linear(x):
            return x * factor + diff

        return linear

    @classmethod
    def make_discrete(cls, threshold, range_min=0, range_max=1):
        assert range_min < range_max

        @np.vectorize
        def discrete(x):
            if x < threshold:
                return range_min
            else:
                return range_max

        return discrete

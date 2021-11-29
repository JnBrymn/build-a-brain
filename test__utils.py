import numpy as np

import utils


@np.vectorize
def about_equal(left, right):
    return -0.001 < left-right < 0.001


def test_NeuronActivationFunctions_make_linear():
    func = utils.NeuronActivationFunctions.make_linear(2,4,3,8)

    assert all(func(np.array([2,3,4])) == np.array([3,5.5,8]))

def test_NeuronActivationFunctions_make_discrete():
    func = utils.NeuronActivationFunctions.make_discrete(0.5)

    assert all(func(np.array([-1,0.49,0.51,5])) == np.array([0,0,1,1]))

    func = utils.NeuronActivationFunctions.make_discrete(0.5,5,9)

    assert all(func(np.array([-1, 0.49, 0.51, 5])) == np.array([5, 5, 9, 9]))


def test_SynapseConnectionFunctions_make_scaled_sigmoid():
    func = utils.SynapseConnectionFunctions.make_scaled_sigmoid(10)

    assert all(
        about_equal(
            func(np.array([ 0, 0.5, 1])),
            np.array(     [-1,   0, 1]),
        )
    )

    # make sure it's really sigmoid and not linear
    assert func(0.1) < -0.999
    assert func(0.9) > 0.999

def test_SynapseConnectionFunctions_make_scaled_double_sigmoid():
    func = utils.SynapseConnectionFunctions.make_scaled_double_sigmoid(100)

    assert all(
        about_equal(
            func(np.array([ 0, 0.33333333, 0.5, 0.66666667, 1])),
            np.array(     [-1,       -0.5,   0,        0.5, 1]),
        )
    )

    # make sure it's really sigmoid and not linear
    assert func(0.45) > -0.00001
    assert func(0.55) < 0.00001
    assert func(0.95) > 0.9999
    assert func(0.05) < -0.9999

def test_SynapseConnectionFunctions_discrete():
    func = utils.SynapseConnectionFunctions.discrete

    assert all(
        about_equal(
            func(np.array([ 0, 0.1, 0.49, 0.51, 0.9, 1])),
            np.array(     [-1,  -1,   -1,    1,   1, 1]),
        )
    )

def test_SynapseConnectionFunctions_double_discrete():
    func = utils.SynapseConnectionFunctions.double_discrete

    assert all(
        about_equal(
            func(np.array([ 0, 0.1, 0.4, 0.5, 0.6, 0.9, 1])),
            np.array(     [-1,  -1,   0,   0,   0,   1, 1]),
        )
    )

def test_SynapseConnectionFunctions_double_linear():
    func = utils.SynapseConnectionFunctions.linear

    assert all(
        about_equal(
            func(np.array([ 0, 0.25, 0.5, 0.666666667, 1])),
            np.array(     [-1, -0.5,   0, 0.333333333, 1]),
        )
    )

from collections import namedtuple
from unittest.mock import patch

import numpy as np
import pytest
from scipy import sparse

import utils
from brain import Brain


def test_brain_creation():
    num_neurons = 4
    brn = Brain(
        num_neurons=num_neurons,
        synaptic_density=1,
    )
    assert all(brn.synapses_a.data == 1), \
        "with synaptic_density=1, all brain.synapses_a should be 1"
    assert all(brn.synapses_b.data == 1), \
        "with synaptic_density=1, all brain.synapses_b should be 1"

    with pytest.raises(AssertionError):
        brn = Brain(
            num_neurons=num_neurons,
            synaptic_density=1.1,
        )

    with pytest.raises(AssertionError):
        brn = Brain(
            num_neurons=num_neurons,
            synaptic_density=-0.1,
        )


def test_synaptic_activation():
    """tests that if the synapse has parameters a and b, that asymptotically, the number of connections will be a/(a+b)"""
    brain = Brain(
        num_neurons=4,
        synaptic_density=1,
    )
    brain.synapses_a[:] = 900
    brain.synapses_b[:] = 100

    history_length = 100
    history = brain.get_synaptic_connection()
    for i in range(history_length):
        history += brain.get_synaptic_connection()

    avg = history / history_length
    expected = brain.synapse_func(brain.synapses_a / (brain.synapses_a + brain.synapses_b))
    expected = expected[0, 0]
    assert all(expected - 0.03 < avg.data)
    assert all(avg.data < expected + 0.03)


def test_update_neuronal_states__basic():
    """/
    makes sure that given a known synaptic_activation matrix and initial neuronal state, that the update works as expected

    if we give it a linear neuron_fun and no random activation, then it should be the same as a dot product
    """
    brain = Brain(
        num_neurons=4,
        synaptic_density=1,
        neuron_func=utils.NeuronActivationFunctions.make_linear(),
        random_activation_scale=0
    )

    syn_con = np.array([[0,1,1,1], [0,0,-1,1], [0,1,1,-1], [0,0,0,1]])

    def mock_get_synaptic_connection():
        return sparse.csr_matrix(syn_con)

    brain.get_synaptic_connection = mock_get_synaptic_connection
    init_neuro_states = np.array([[1],[1],[1],[0]])
    brain.neuronal_states = init_neuro_states
    brain.update_neuronal_states()

    assert all(np.dot(syn_con, init_neuro_states) == brain.neuronal_states)

def test_update_neuronal_states__discrete_func():
    """/
    makes sure that given a known synaptic_activation matrix and initial neuronal state, that the update works as expected

    if we give it a linear neuron_fun and no random activation, then it should be the same as a dot product
    """
    neuron_func = utils.NeuronActivationFunctions.make_discrete()
    brain = Brain(
        num_neurons=4,
        synaptic_density=1,
        neuron_func=neuron_func,
        random_activation_scale=0,
    )

    syn_con = np.array([[0,1,1,1], [0,0,-1,1], [0,1,1,-1], [0,0,0,1]])

    def mock_get_synaptic_connection():
        return sparse.csr_matrix(syn_con)

    brain.get_synaptic_connection = mock_get_synaptic_connection
    init_neuro_states = np.array([[1],[1],[1],[0]])
    brain.neuronal_states = init_neuro_states
    brain.update_neuronal_states()

    assert all(neuron_func(np.dot(syn_con, init_neuro_states)) == brain.neuronal_states)


def test_update_neuronal_states__random_activation_scale():
    """/
    makes sure that given a known synaptic_activation matrix and initial neuronal state, that the update works as expected
    """
    num_neurons = 1000
    random_activation_scale = 10
    brain = Brain(
        num_neurons=num_neurons,
        synaptic_density=1,
        neuron_func=utils.NeuronActivationFunctions.make_linear(),
        random_activation_scale=random_activation_scale,
    )
    brain.neuronal_states = np.zeros([num_neurons,1])
    brain.update_neuronal_states()

    assert abs(sum(brain.neuronal_states)/num_neurons - random_activation_scale) < .5


def test_update_neuronal_states_including_hebbian_learning():
    """Test that the Hebbian learning works as expected.

    The pre neuronal state can be either 0 or 1
    The post neuronal state can be either 0 or 1
    The synaptic connection can be 0 (not connected), or 1 (connected)

    If the pre-neuronal state is 0 OR the synapse is not connected then the synapse shouldn't change.
    If the pre-neuronal state is 1 AND the synapse is connected, then neurons that fire together wire together:
      If the post neuronal state is 0, the synapse should be modified to be more inhibitory.
      If the post neuronal state is 1, the synapse should be modified to be more excitatory.

    If we look at this situation using matrix multiplication-like syntax
    post  ≈    connections   *   post
       0   ≈    A B               0
       1        C D               1

    Then if A,B,C, and D are disconnected, then their synapses_a and synapses_b should not be modified.

    But if A,B,C, and D are connected, then they should be modified as follows.

    A - nothing
    B - more negative (synapses_b++)
    C - nothing
    D - more positive (synapses_a++)
    """

    neuron_states = np.array([[0],[1]])
    def mock_update_neuronal_states(self):
        self.neuronal_states = neuron_states

    with patch.object(Brain, 'update_neuronal_states', mock_update_neuronal_states):
        brain = Brain(num_neurons=2)
        brain.neuronal_states = neuron_states
        brain.synapses_a = sparse.csr_matrix(np.ones([2, 2]))
        brain.synapses_b = sparse.csr_matrix(np.ones([2, 2]))

        brain.update_neuronal_states_including_hebbian_learning()

        assert np.all(brain.synapses_a.todense() == np.array([[1,1],[1,2]]))
        assert np.all(brain.synapses_b.todense() == np.array([[1,2],[1,1]]))

        brain.synapses_a = sparse.csr_matrix(np.zeros([2, 2]))
        brain.synapses_b = sparse.csr_matrix(np.zeros([2, 2]))

        brain.update_neuronal_states_including_hebbian_learning()
        assert np.all(brain.synapses_a.todense() == np.zeros([2,2]))
        assert np.all(brain.synapses_b.todense() == np.zeros([2, 2]))

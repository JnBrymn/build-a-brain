from collections import namedtuple
from unittest.mock import patch

import numpy as np
import pytest

from brain import Brain


def test_brain_creation():
    num_neurons = 4
    brain = Brain(
        num_neurons=num_neurons,
        excitatory_synaptic_density=1,
        inhibitory_synaptic_density=0,
        neuronal_max_threshold=4,
        initial_active_neuron_density=0.5,
    )

    assert np.all(brain.synapses == np.ones([num_neurons, num_neurons], int)), \
        "with excitatory_synaptic_density=1, all brain.synapses should be 1"

    brain = Brain(
        num_neurons=num_neurons,
        excitatory_synaptic_density=0,
        inhibitory_synaptic_density=1,
        neuronal_max_threshold=4,
        initial_active_neuron_density=0.5,
    )

    assert np.all(brain.synapses == -np.ones([num_neurons, num_neurons], int)), \
        "with inhibitory_synaptic_density=1, all brain.synapses should be -1"

    with pytest.raises(AssertionError):
        brain = Brain(
            num_neurons=num_neurons,
            excitatory_synaptic_density=0.7,
            inhibitory_synaptic_density=0.7,
            neuronal_max_threshold=4,
            initial_active_neuron_density=0.5,
        )

    with pytest.raises(AssertionError):
        brain = Brain(
            num_neurons=num_neurons,
            excitatory_synaptic_density=0,
            inhibitory_synaptic_density=0,
            neuronal_max_threshold=4,
            initial_active_neuron_density=0.5,
        )

def test_synaptic_activation():
    num_neurons = 4
    brain = Brain(
        num_neurons=num_neurons,
        excitatory_synaptic_density=1,
        inhibitory_synaptic_density=0,
        neuronal_max_threshold=4,
        initial_active_neuron_density=0.5,
    )
    total_activations = np.zeros([num_neurons, num_neurons], int)
    for i in range(32):
        total_activations += brain.get_synaptic_activation()

    assert np.all(total_activations > 0), \
        "with excitatory_synaptic_density=1, then eventually all synapses should be activated positively"

    brain = Brain(
        num_neurons=num_neurons,
        excitatory_synaptic_density=0,
        inhibitory_synaptic_density=1,
        neuronal_max_threshold=4,
        initial_active_neuron_density=0.5,
    )
    total_activations = np.zeros([num_neurons, num_neurons], int)
    for i in range(32):
        total_activations += brain.get_synaptic_activation()

    assert np.all(total_activations < 0), \
        "with inhibitory_synaptic_density=1, then eventually all synapses should be activated negatively"


def test_synaptic_activation_2():
    brain = Brain(
        num_neurons=1,
        excitatory_synaptic_density=1,
        inhibitory_synaptic_density=0,
        neuronal_max_threshold=4,
        initial_active_neuron_density=0.5,
    )
    brain.synapses_a[0,0] = 9
    brain.synapses_b[0,0] = 1

    history_length = 10000
    history = []
    for i in range(history_length):
        history.append(brain.get_synaptic_activation())

    avg = sum(history) / history_length
    expected = brain.synapses_a / (brain.synapses_a + brain.synapses_b)
    assert expected - 0.03 < avg < expected + 0.03


def test_update_neuronal_states():
    brain = Brain(
        num_neurons=4,
        excitatory_synaptic_density=1,
        inhibitory_synaptic_density=0,
        neuronal_max_threshold=2,
        initial_active_neuron_density=0.5,
    )

    def mock_get_synaptic_activation():
        return np.array([[0,1,1,1], [0,0,1,1], [0,1,1,-1], [0,0,0,1]])

    brain.get_synaptic_activation = mock_get_synaptic_activation
    brain.neuronal_states = np.array([[1],[1],[1],[1]])
    brain.update_neuronal_states()

    assert np.all(brain.neuronal_states == np.array([[1], [1], [0], [0]]))
    assert np.all(brain.neuronal_thresholds == np.array([[2], [2], [1], [1]]))


def test_update_neuronal_states_including_hebbian_learning():
    """Test that the Hebbian learning works as expected.

    The pre neuronal state can be either 0 or 1
    The post neuronal state can be either 0 or 1
    The synaptic activation can be -1 (inhibitory), 0 (not connected), or 1 (excitatory)

    We must test that in each case self.synapses_a and self.synapses_b are
    updated properly as seen in the table below.
    """
    Test = namedtuple('Test', ['pre_neuron', 'post_neuron', 'synaptic_activation', 'delta_a', 'delta_b'])
    tests = [
        # if pre_neuron is off, there should be no update no matter what
        Test(0,0,-1,0,0),
        Test(0,0,0,0,0),
        Test(0,0,1,0,0),
        Test(0,1,-1,0,0),
        Test(0,1,0,0,0),
        Test(0,1,1,0,0),

        # general idea below - if the synapse isn't connected, don't update anything
        # if the outcome is against the synapse, then increment b (always +1)
        # if the outcome is in agreement with the synapse, then increment a

        # if pre is on and post is off and the synapse is inhibitory then it worked - increment a
        Test(1,0,-1,1,0),
        # the synapse is not connected, do no updates
        Test(1,0,0,0,0),
        # if pre is on and post is off and the synapse is excitatory then it failed - increment b
        Test(1,0,1,0,1),
        # if pre is on and post is on and the synapse is inhibitory then it failed - increment b
        Test(1,1,-1,0,1),
        # the synapse is not connected, do no updates
        Test(1,1,0,0,0),
        # if pre is on and post is on and the synapse is excitatory then it worked - increment a
        Test(1,1,1,1,0),
    ]

    # we are going to run all tests using only the first neuron and the synapse that connects the first neuron to itself
    def get_mock_update_neuronal_states(neuron_activation):
        def func(self):
            self.neuronal_states = self.neuronal_states.copy()
            self.neuronal_states[0,0] = neuron_activation

        return func

    for t in tests:
        with patch.object(Brain, 'update_neuronal_states', get_mock_update_neuronal_states(t.post_neuron)):
            brain = Brain(
                num_neurons=4,
                excitatory_synaptic_density=1,
                inhibitory_synaptic_density=0,
                neuronal_max_threshold=2,
                initial_active_neuron_density=0.5,
            )
            brain.neuronal_states[0, 0] = t.pre_neuron
            brain.synapses[0, 0] = t.synaptic_activation
            brain.synapses_a[0, 0] = 0
            brain.synapses_b[0, 0] = 0

            brain.update_neuronal_states_including_hebbian_learning()

        assert brain.synapses_a[0, 0] == t.delta_a, f'Failure for {t} - delta_a was {brain.synapses_a[0, 0]}'
        assert brain.synapses_b[0, 0] == t.delta_b, f'Failure for {t} - delta_b was {brain.synapses_a[0, 0]}'

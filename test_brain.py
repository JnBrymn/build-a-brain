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
        total_activations += 1*brain.get_synaptic_activation()

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
        total_activations += 1*brain.get_synaptic_activation()

    assert np.all(total_activations < 0), \
        "with inhibitory_synaptic_density=1, then eventually all synapses should be activated negativel"

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

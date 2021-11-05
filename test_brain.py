import numpy as np

from brain import Brain


def test_brain_creation():
    num_neurons = 4
    brain = Brain(
        num_neurons=num_neurons,
        excitatory_synaptic_density=1,
        inhibitory_synaptic_density=0.1,
        neuronal_max_threshold=4,
        initial_active_neuron_density=0.5,
    )

    assert np.all(brain.synapses == np.ones([num_neurons, num_neurons], int)), \
        "with excitatory_synaptic_density=1, all brain.synapses should be 1"

    # TODO - test inhibitory_synaptic_density

def test_synaptic_activation():
    num_neurons = 4
    brain = Brain(
        num_neurons=num_neurons,
        excitatory_synaptic_density=1,
        inhibitory_synaptic_density=0.1,
        neuronal_max_threshold=4,
        initial_active_neuron_density=0.5,
    )
    total_activations = np.zeros([num_neurons, num_neurons], int)
    for i in range(32):
        total_activations += 1*brain.get_synaptic_activation()

    assert np.all(total_activations > 0), \
        "with excitatory_synaptic_density=1, then eventually all synapses should be activated"

    # TODO - test inhibitory_synaptic_density
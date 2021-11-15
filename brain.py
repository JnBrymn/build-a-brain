import numpy as np

import matplotlib.pyplot as plt
from scipy.sparse import (
    rand as sparse_rand,
    csr_matrix
)

import utils

class Brain:
    def __init__(
            self,
            num_neurons=5,
            synaptic_density=0.2,
            synapse_func=None,
            neuron_func=None,
            random_activation_scale=0.2,
    ):
        """

        :param num_neurons: default val is 5
        :param synaptic_density: the portion of synapses that act to excite on inhibit the downstream neuron
            default value is 0.2
        :param synapse_func: a function that is applied to the raw connection value (0..1) that results is drawn from the synapse's beta distribution
            default val is utils.SynapseConnectionFunctions.linear which scales the output linearly to -1..1
        :param neuron_func: a function that is applied to the raw neuronal activation ( SynapseConnection @ NeuronState )
            default val is utils.NeuronActivationFunctions.make_discrete(0.5) which maps values <=0.5 to 0 and >= 0.5 to 1
        :param random_activation_scale: the scale of np.random.exponential - random values are drawn from this distribution
            and added to the neuron stats prior to applying the neuron_func
            default val is 0.2
        """
        self.iteration = 0
        self.num_neurons = num_neurons
        if not synapse_func:
            synapse_func = utils.SynapseConnectionFunctions.linear
        self.synapse_func = synapse_func

        if not synapse_func:
            neuron_func = utils.NeuronActivationFunctions.make_discrete(0.5)
        self.neuron_func = neuron_func

        self.random_activation_scale = random_activation_scale

        # neuronal states
        self.neuronal_states = np.zeros([self.num_neurons, 1])

        # synaptic strengths
        assert 0 <= synaptic_density <= 1

        # self.synapses_{a,b} hold the beta distribution parameters for each synapse that control whether or not they
        # actually activate - the more b, the more inhibitory; the more a, the more excitatory
        self.synapses_a = sparse_rand(num_neurons, num_neurons, density=num_neurons, format='csr')
        self.synapses_a.data = 1
        self.synapses_b = self.synapses_a.copy()

    def get_synaptic_connection(self):
        """\
        Connection of the synapses is a number between -1 and 1 (-1 inhibitory and 1 excitatory)
        """

        # this is ridiculously inefficient
        synaptic_connections_data = np.random.beta(self.synapses_a.data,self.synapses_b.data)
        synaptic_connections_data = self.synapse_func(synaptic_connections_data)
        synaptic_connections = self.synapses_a.copy()
        synaptic_connections.data = synaptic_connections_data

        return synaptic_connections

    def update_neuronal_states(self):
        """the next neuronal state depends upon what neurons are active currently, what synapses are active, and
        whether the resulting activation is above the threshold for activation."""
        #TODO! step through and watch this work - make sure the vectorized functions work on sparce vectors

        synaptic_activations = self.get_synaptic_connection()
        next_neuronal_states = synaptic_activations.dot(self.neuronal_states) \
                               + np.random.exponential(self.random_activation_scale, [self.num_neurons,1])
        next_neuronal_states = self.neuron_func(next_neuronal_states)

        self.neuronal_states = 1*next_neuronal_states
        self.iteration += 1

    def update_neuronal_states_including_hebbian_learning(self):
        """runs update_neuronal_states but also updates self.synapses_{a,b} appropriately"""
        # in the future neuronal states will be continuous, I'm not sure this is a threshold that will always make sense
        # we should let the user specify this in the __init__ #TODO!
        activation_threshold = 0.5
        pre_neuronal_states = csr_matrix(1 * (self.neuronal_states > activation_threshold))
        self.update_neuronal_states()
        post_neuronal_states = 1 * (self.neuronal_states > activation_threshold)
        post_neuronal_states__on = csr_matrix(post_neuronal_states)
        post_neuronal_states__off = csr_matrix(1-post_neuronal_states)

        synapse_pre_1_post_0 = post_neuronal_states__off.dot(pre_neuronal_states.T)
        synapse_pre_1_post_1 = post_neuronal_states__on.dot(pre_neuronal_states.T)

        # synapses that are connected in any way, inhibitory, excitatory, strong, or weak
        # TODO for efficiency sake, if the connectivity never changed, then we could create the connected_synapses in
        #  the __init__.
        connected_synapses = self.synapses_a.copy()
        connected_synapses.data = 1

        # these synapses have upstream neurons that were active and were connected to the downstream neurons
        connected_synapse_pre_1_post_0 = synapse_pre_1_post_0.multiply(connected_synapses)
        connected_synapse_pre_1_post_1 = synapse_pre_1_post_1.multiply(connected_synapses)

        # if a connected synapse was excitatory, then we increment a so that in the future it will be more excitatory
        self.synapses_a += connected_synapse_pre_1_post_1
        # if a connected synapse was inhibitory, then we increment b so that in the future it will be more inhibitory
        self.synapses_b += connected_synapse_pre_1_post_0

    def simulate_brain(self, num_updates=150, with_hebbian_learning=True):
        neuron_history = self.neuronal_states

        for i in range(num_updates):
            if with_hebbian_learning:
                self.update_neuronal_states_including_hebbian_learning()
            else:
                self.update_neuronal_states()
            neuron_history = np.concatenate([neuron_history, self.neuronal_states], axis=1)

        plt.imshow(neuron_history)

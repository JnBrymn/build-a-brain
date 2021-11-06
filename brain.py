import numpy as np

import matplotlib.pyplot as plt


class Brain:
    def __init__(self, num_neurons, excitatory_synaptic_density, inhibitory_synaptic_density, neuronal_max_threshold, initial_active_neuron_density):
        """

        :param num_neurons:
        :param excitatory_synaptic_density: the portion of synapses that act to excite the downstream neuron
        :param inhibitory_synaptic_density: the portion of synapses that act to inhibit the downstream neuron
        :param neuronal_max_threshold: if a neuron fires, then immediately after it has this activation threshold, and then thereafter the threshold decreases by 1 each iteration
        :param initial_active_neuron_density: at t=0 what portion of neurons are activated?
        """
        self.num_neurons = num_neurons
        self.neuronal_max_threshold = neuronal_max_threshold

        # neuronal states
        self.neuronal_states = 1 * (np.random.rand(self.num_neurons, 1) < initial_active_neuron_density)
        self.neuronal_thresholds = np.ones([self.num_neurons, 1]) * self.neuronal_max_threshold

        # synaptic strengths
        assert 0 < excitatory_synaptic_density + inhibitory_synaptic_density <= 1, "excitatory_synaptic_density + inhibitory_synaptic_density must be between 0 and 1"

        # self.synapses serves as a pattern for how neurons are connected and whether the connection is excitatory or
        # inhibitory
        self.synapses = np.random.rand(self.num_neurons, self.num_neurons)
        self.synapses = 1*(self.synapses>=(1-excitatory_synaptic_density)) - 1*(self.synapses<inhibitory_synaptic_density)
        # self.synapses_{a,b} hold the beta distribution parameters for each synapse that control whether or not they
        # actually activate
        self.synapses_a = 0.1 * (self.synapses != 0)
        self.synapses_b = self.synapses_a.copy()

    def get_synaptic_activation(self):
        """activation of the synapses is True if the random number exceeds the synaptic_threshold"""

        # average of beta distributions determines the probability of synaptic activation
        synaptic_threshold = self.synapses_a / (self.synapses_a + self.synapses_b)
        synaptic_threshold *= self.synapses  #
        rands = np.random.rand(self.num_neurons, self.num_neurons)
        # if the synapses
        synaptic_activations = 1 * (rands > (1 - synaptic_threshold)) + -1 * (-rands < -(1 + synaptic_threshold))

        return synaptic_activations

    def update_neuronal_states(self):
        """the next neuronal state depends upon what neurons are active currently, what synapses are active, and
        whether the resulting activation is above the threshold for activation."""
        synaptic_activations = self.get_synaptic_activation()
        next_neuronal_states = synaptic_activations @ self.neuronal_states
        next_neuronal_states = 1 * (next_neuronal_states >= self.neuronal_thresholds)
        self.neuronal_states = next_neuronal_states

        # decrease the neuronal_threshold for all neurons but reset neuronal_thresholds for neurons that activated
        self.neuronal_thresholds = self.neuronal_thresholds - 1
        self.neuronal_thresholds[self.neuronal_states == 1] = self.neuronal_max_threshold

    def update_neuronal_states_including_hebbian_learning(self):
        """runs update_neuronal_states but also updates self.synapses_{a,b} appropriately"""
        pre_neuronal_states = self.neuronal_states.copy()
        self.update_neuronal_states()
        post_neuronal_states = self.neuronal_states

        synapse_pre_1_post_0 = np.transpose(1-post_neuronal_states)*pre_neuronal_states
        synapse_pre_1_post_1 = np.transpose(post_neuronal_states) * pre_neuronal_states

        excitatory_synapses = self.synapses > 0
        inhibitory_synapses = self.synapses < 0

        # succeeded, so we increment self.synapses_a "neurons that fire together wire together"
        self.synapses_a += synapse_pre_1_post_0 * inhibitory_synapses
        self.synapses_a += synapse_pre_1_post_1 * excitatory_synapses

        # failed, so we increment self.synapses_b "neurons that DON'T fire together wire apart"
        self.synapses_b += synapse_pre_1_post_0 * excitatory_synapses
        self.synapses_b += synapse_pre_1_post_1 * inhibitory_synapses

    def simulate_brain(self, num_updates=150, with_hebbian_learning=True):
        neuron_history = self.neuronal_states

        for i in range(num_updates):
            if with_hebbian_learning:
                self.update_neuronal_states_including_hebbian_learning()
            else:
                self.update_neuronal_states()
            neuron_history = np.concatenate([neuron_history, self.neuronal_states], axis=1)

        plt.imshow(neuron_history)
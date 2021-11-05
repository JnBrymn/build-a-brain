import numpy as np


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
        self.excitatory_synaptic_density = excitatory_synaptic_density
        self.neuronal_max_threshold = neuronal_max_threshold

        # neuronal states
        self.neuronal_states = 1 * (np.random.rand(self.num_neurons, 1) < initial_active_neuron_density)
        self.neuronal_thresholds = np.ones([self.num_neurons, 1]) * self.neuronal_max_threshold

        # synaptic strengths
        self.synapses = 1 * (np.random.rand(self.num_neurons, self.num_neurons) < self.excitatory_synaptic_density)
        self.synapses_a = 0.1 * self.synapses.copy()
        self.synapses_b = self.synapses_a.copy()

    def get_synaptic_activation(self):
        """activation of the synnapses is True if the random number exceeds the synaptic_threshold"""
        synaptic_threshold = self.synapses_a / (self.synapses_a + self.synapses_b)
        synaptic_activations = (self.synapses.copy() * np.random.rand(self.num_neurons, self.num_neurons)) > (
            synaptic_threshold)
        return synaptic_activations

    def update_neuronal_states(self):
        """the next neuronal state depends upon what neurons are active currently, what synapses are active, and
        whether the resulting activation is above the threshold for activation."""
        synaptic_activations = self.get_synaptic_activation()
        next_neuronal_states = synaptic_activations @ self.neuronal_states
        next_neuronal_states = 1 * (next_neuronal_states > self.neuronal_thresholds)
        self.neuronal_states = next_neuronal_states

        # decrease the neuronal_threshold for all neurons but reset neuronal_thresholds for neurons that activated
        self.neuronal_thresholds = self.neuronal_thresholds - 1
        self.neuronal_thresholds[self.neuronal_states == 1] = self.neuronal_max_threshold

    def simulate_brain(self, num_updates=150):
        neuron_history = self.neuronal_states

        for i in range(num_updates):
            self.update_neuronal_states()
            neuron_history = np.concatenate([neuron_history, self.neuronal_states], axis=1)

        plt.imshow(neuron_history)
# build-a-brain
John attempts to build a brain using numpy.


## Versions
* v0.1.0
    * excitatory and inhibitory neurons; linear activation threshold decay
    * the natural firing of neurons died out quickly and each neuron just started firing independently as their activation threshold decreased to 0
* v0.2.0
    * activation threshold a constant across all neurons rather than having it decrease after a neuron isn't being fire
    * added random excitation parameter so that neurons just sometimes fire on their own
    * made both of the above optionally into functions that take iteration number and return a value
  
## Dev Ideas
* Should we do anything for connected_synapse_pre_0_post_* synapses? Maybe we could scale down a and b so that the neuron starts relearning a better connection - but that might add too much randomness
* The synapse matrix should be sparse, or eventually maybe block dense
* `probability_of_random_excitation` should be replaced with just adding and arbitrary input INTO each neuron that follows an exponential distribution
* Can hebbian learning happen in parallel and in batch?
* There is a way to use a beta distribution for the synapse activation that is continuous and encompasses both inhibitory and excitatory activation
    * You can post filter it to make it exactly -1,0,1 or you can make it continuous
    * The actual neural activation can be a sigmoid'ed result of the activation and either be discrete, {0,1}, or continuous and capped {0..1}, or RELU, {0..inf}
* Build a NAND using 2 inputs, an always on neuron, an output and an internal neuron.
    * Construct it directly to prove it works.
    * Set up a synapses so that they will converge to it.
    * Set up arbitrary synapses that at least _contain_ connections that could converge to it and see if it works
* Built a markov model that is sent letters (one of 27 neuron inputs) and it predicts the next letter. See if it can make meaningful sentences.
    * Design: input neuron for every char; there is a 2nd layer that is a delayed version of the first layer; there is a layer that ANDs together the first 2 laters; the AND layer connects to an output laters (one output per character); the output layer indicates the probability of each char being the next layer;
    * Exercise by feeding the output layer back to the input layer
* Make a flip-flip; because that would demonstrate a simple ability to keep memory. 

## Neat ideas
* Since you can build NANDs into this model, then you can hand construct sub-components of the network to perform particular computations rather than having to learn those computations
* You can play with the certainty (variance) of synaptic activations by scaling the Beta params `a`, and `b`.
    * If you've decided that a neuron is _certain_ then you can just replace the Beta value with a/(a+b) and make computation faster.
    * If you want to start paring or adding neurons, then associated synapses can have `a` and `b` scaled down so that their value is effectively re-evalueated
    * I propose that there is some biological mechanism that enacts the above bullet. I bet it that if a neuron starts NOT exciting the downtream neuron as it expects to, then it's biological version of `a` and `b` gets scaled down and the connection is reevaluated.
        * Further wild guess: I bet that if a neuron start not being activated as much by upstream neurons, then it will get more sensitive to the upstream neurons and firing with even less upstream activation. This behavior would provide a mechanism for healing damage by reconnecting by less-dominant paths. Consider this scenario: neurons X and Y mutually excite downsteam neuron Z. If X dies, then Y might not be excitatory enough to activate Z. Y will have it's `a` and `b` parameters reduced to effectively start searching for it's new appropriate connectivity value. Because of the damage, Z starts firing less, and therefore becomes more sensitive to upstream excitation. This causes Y to actual _increase_ the synaptic connectivity to Z and repair the transmission to Z. This might have implications in treating brain damage. 
    
## Next version    
* change the design of synapses: instead of randomly choosing whether or not they are active `{-1, 0, 1}` choose _how_ active they are `{-1..1}`. Do this by backing with a single beta distribution and then mapping to -1,1. (You can actually make this mapping non-linear so that it again mimics `{-1, 0, 1}`; so this is a generalization. 
* The neurons would be a sum of the input Synapse*Neuron with a non-linear function applied to the output
    * This could be sigmoid, or RELU
    * Could also be a discrete function that mimicked the current 0,1 implementation. So this is also a generalization.
* There are two ways to determine the synapse activation:
    1. `nonlinearity(a/(a+b))` - this is always going to have the same output corresponding to the expected value of the Beta distribution. This might be good for running the model; but who knows, maybe, like the brain, randomness will be good for computation.
    2. `nonlinearity(b)` where `b` is drawn from `Beta(a,b)` - this will be different each time, but if the neuron is "certain of the distribution" (large a and b) then it will be almost the same everytime. This behavior might be good during training because the neurons can converge on the right solution.
    
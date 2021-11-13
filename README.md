# build-a-brain
John attempts to build a brain using numpy.


## Versions
* v0.1.0
  * excitatory and inhibitory neurons; linear activation threshold decay
  * the natural firing of neurons died out quickly and each neuron just started firing independently as their activation threshold decreased to 0
  
## Ideas
* Can hebbian learning happen in parallel and in batch?
* There is a way to use a beta distribution for the synapse activation that is continuous and encompasses both inhibitory and excitatory activation
  * You can post filter it to make it exactly -1,0,1 or you can make it continuous
  * The actual neural activation can be a sigmoid'ed result of the activation and either be discrete, {0,1}, or continuous and capped {0..1}, or RELU, {0..inf}
# Causal Perceptron

The goal of this project is to construct a perceptual algorithm at the single neuron level, upon which to build a neural network for an agent to learn and operate in the world.

# Main Idea
The challenge in designing a perception algorithm at the single neuron level lies in: how can neurons assembled together demonstrate collective capabilities? It seems that if the neurons are not under unified command, only chaos will result. 

In the original backpropagation algorithm, we set a global objective function and update the parameters by gradient descent across the entire network. 

However, this objective function is typically inseparable and the neural network can't be split into individual parts for separate training. Hence, we need to find new objectives and concepts that can be broken down to the neuron level, and yet hold certain significance on a macro level.

One potential idea is to allow neurons to discover and maintain an information loop between themselves and the environment.

For the internal flow of information, it is relatively easy to control, but how can we ensure the smoothness of the external information pathway? 

We just need to verify the causal relationship from output to input. The verification method is exactly Do (random behavior) + finding a mapping function from input to output. 

Interestingly, Do also brings the benefit of exploring the environment and learning new things. 

And to maintain the information loop, the learned behavior (mapping function) needs to be able to return the input to a state that has been seen before.

This idea seems to be scalable to macro level as well. We always do things that can have an impact on the environment and are observable, and we repeat these actions every day.

# Algorithm

For verifying the causality, we accomplish it through a combination of random behavior and function mapping. 

Each neuron has a certain probability P to produce a random output. Then, it learns the mapping function from the returned input to this random output. 

In this case, the function we chose is a parameterized Sigmoid function, hence, the range of random output is between (0,1).

For maintaining the cycle, we utilize cosine similarity to calculate the number of input state cycles, referred to as Loop Count. 

Additionally, we employ a variable named Trace to retain memory of the current learning content and decide whether to learn new content. 

The value of Trace will decrease continuously following the rule: Trace = max(Trace - 1, Loop Count).

# Experiments

## MNIST World
In this environment, we have 10 keys, numbered from 0 to 9. The agent's output represents the key it presses. For example, if the agent presses 3, an image with a label of 3 is randomly selected from the MNIST dataset and passed to the agent as its input.

Rather than being motivated by any particular rewards, the agent aims to establish a recurring cycle of actions and states. By pressing a key and receiving a corresponding image, the agent forms a causal loop, learning to map a specific output (key press) to an expected input (image). The goal here is to maintain a consistent and predictable pattern of behavior, striving to sustain the information loop within the MNIST world.

## Inverted Pendulum
In this environment, the challenge is to keep the pendulum standing upright. The agent's actions control the force applied to the base of the pendulum. Instead of using a traditional reward system (which would typically provide higher scores for keeping the pendulum upright), we use our novel perception algorithm.

The objective in this context is to maintain a cycle of states and actions, as opposed to merely maximizing a reward signal. Our agent strives to explore and maintain an understanding of its environment, and to keep itself in a familiar state by continuously updating its behavior. The pendulum is kept upright not because of a reward, but because this action maintains the agent's internal state and information loop.

# Conclusion
The perception algorithm at the neuron level we proposed offers a new perspective on training neural networks. Rather than being guided by a global objective function, each neuron explores and maintains its own understanding of the world, based on a local information loop. It will be interesting to see how this approach scales to more complex environments and tasks.

Please note that this project is in the early stages of development, and there is much more to discover and improve. Contributions and suggestions are welcomed!
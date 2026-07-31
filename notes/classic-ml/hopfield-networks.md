
## Point-attractors (Hopfield networks)
The most basic attractor network, converges to a single state.

Hopfield networks work by having a single layer of neurons all connected to each other (neurons cannot be connected to themselves). Neurons consist of binary states, -1 or 1, and weights (-1, 1). The connection between two neurons is set such that either neuron can translate to the connecting neuron via the weight. Below are the possible configurations:

```
-1  <-->   1  (neurons are different)
-1  <-->  -1  (neurons are the same, both negative)
1   <-->   1  (neurons are the same, both positive)
```
If the neurons are the same, then a weight of 1 is set. The output of a neuron `n`, connected with neuron `c` with weight `w` is determined by the following `n = c * w`. Having `w = 1` just sets `n` to be whatever `c` is. If the neurons are opposite, then a weight of -1 will set neuron `n` to be the opposite of neuron `c`.

Now this doesn't mean anything with only two neurons, since if one is corrupted, there is no way to tell which one is correct. But with one corrupted neuron and one learnt pattern, having a network of at least 4 neurons gives us learnt error correction. Given the following network of 4 neurons:

```
C --- O
| \  /|
|  X  |
| /  \|
O --- O
```

If one is corrupted (C), then its three neighbours will know from the connected weights that the value of the corrupted neuron should be the opposite of what it currently is, here the majority overrules the single corrupt neuron. Similarly, when the corrupt neuron is multiplied with each corresponding weight, the neuron will claim that it's neighbour is the opposite value of what it should be. However, there will be two other correct neurons that will know the true value and will thus outvote the corrupt neuron.
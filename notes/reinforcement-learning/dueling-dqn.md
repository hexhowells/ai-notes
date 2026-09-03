# Dueling DQNs

The standard DQN architecture estimates the action-value function directly $Q(s, a)$. However, for certain states, actions are not too important as the state's overall value is good regardless of the action taken. Thus it is beneficial to estimate the value of both the actions and state.

This is done by separating out the hidden layers into two streams:
- The value stream $V(s)$, representing how good the state $s$ is.
- The advantage stream $A(s, a)$ of size $\vert{}\mathcal{A}\vert{}$, representing the value of each action $a$ given state $s$

These two streams leverage backpropagation to allow the model to output the same Q-values and be trained using the training paradigm without any modifications. Whilst also training these two separate value estimators.

```math
Q(s, a) = V(s) + \left( A(s, a) - \frac{1}{\vert{}\mathcal{A}\vert{}} \sum_{a'} A(s, a') \right)
```

Here, when backpropagating the errors for certain action values, the error is directly backpropagated through the value stream:

```math
\frac{\partial L}{\partial V(s)} = -\delta
```

Whereas in the advantage stream, the same error is backpropagated through the chosen action, while all other actions' advantages are uniformly pushed in the opposite direction:
```math
\frac{\partial L}{\partial A(s, a_j)} = -\delta \left( \delta_{ij} - \frac{1}{\vert{}\mathcal{A}\vert{}} \right)
```

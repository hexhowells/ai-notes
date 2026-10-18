# Advantage Actor-Critic (A2C)

> Original paper: [Asynchronous Methods for Deep Reinforcement Learning](https://arxiv.org/abs/1602.01783)

Advantage Actor-Critic (A2C) is a hybrid reinforcement learning algorithm that combines policy-based methods (which directly optimise the policy distribution) and value-based methods (which estimate expected returns, like [DQN](dqn.md)). 

In pure policy gradient methods (like REINFORCE), the policy is updated using the empirical return $G_t$ accumulated over an entire episode. However, because rewards and actions across a full episode are stochastic, $G_t$ has massive variance, which leads to noisy gradients and unstable training.

Actor-Critic methods resolve this by splitting the model into two heads (typically sharing a common feature extraction backbone):
- **The Actor** $\pi_\theta(a \mid s)$, which outputs a probability distribution over actions given state $s$ and updates the policy in the direction suggested by the critic.
- **The Critic** $V_\phi(s)$, which estimates the expected return of state $s$ and acts as a learned baseline to evaluate the actor's chosen actions.

## Advantage Estimation

To update the actor, we want to know how much better a specific action $a_t$ was compared to the average action the policy would take in state $s_t$. This is defined by two value functions:
- **State-Value** $V(s_t)$: The expected return of being in state $s_t$ on average across the policy's action distribution (how good is it to be in state $s_t$ on average across all possible actions the policy might take?)
- **Action-Value** $Q(s_t, a_t)$: The expected return of taking a specific action $a_t$ in state $s_t$ and following the policy after (how good it is to take the specific action $a_t$ while in state $s_t$?).

The difference between the two is the **Advantage** $A(s_t, a_t)$:

```math
\large A(s_t, a_t) = Q(s_t, a_t) - V_\phi(s_t)
```

Subtracting the state-value $V_\phi(s_t)$ centers the return signal around zero:
- If $A(s_t, a_t) > 0$, the action yielded a better return than expected, so the actor increases the probability of selecting $a_t$.
- If $A(s_t, a_t) < 0$, the action performed worse than average, so the actor pushes the probability of $a_t$ down.

Rather than training two separate networks to estimate both $Q(s_t, a_t)$ and $V_\phi(s_t)$, we can approximate $Q(s_t, a_t)$ using the actual rewards collected over $k$ rollout steps plus the critic's bootstrapped estimate of the final state $s_{t+k}$:

```math
\large R_t = \sum_{i=0}^{k-1} \gamma^i r_{t+i} + \gamma^k V_\phi(s_{t+k})
```

```math
\large A(s_t, a_t) = R_t - V_\phi(s_t)
```

Where:

`r` = reward

`γ` = discount factor

`R_t` = $k$-step bootstrapped return (target value)

`V_ϕ(s_t)` = predicted value of the current state from the critic head

Assume an episode rollout of 3 steps ($k=3$, $\gamma = 0.9$) where the agent receives rewards `[0, 0, 1]` and the critic predicts the value of the next state $V(s_4) = 0.5$ (and $0$ if $s_4$ is terminal). Working backwards from $t=3$ to $t=1$, the target returns $R_t$ are computed as:

```text
t=3  R3 = (r3=1) + 0.9 * V(s4)=0.5   = (1 + 0.9 * 0.5)   = 1.45
t=2  R2 = (r2=0) + 0.9 * R3=1.45     = (0 + 0.9 * 1.45)  = 1.305
t=1  R1 = (r1=0) + 0.9 * R2=1.305    = (0 + 0.9 * 1.305) = 1.1745
```

If the critic's baseline predictions for those states were $V(s_1) = 1.0$, $V(s_2) = 1.5$, and $V(s_3) = 1.1$, the resulting advantages become:

```text
t=3  A3 = R3(1.45)   - V(s3)(1.1) = +0.35   (better than expected -> reinforce a3)
t=2  A2 = R2(1.305)  - V(s2)(1.5) = -0.195  (worse than expected  -> penalise a2)
t=1  A1 = R1(1.1745) - V(s1)(1.0) = +0.1745 (better than expected -> reinforce a1)
```

## Loss Function

Because the actor and critic often share a backbone (or are optimised in a single backward pass), the total loss is the sum of three components: the policy (actor) loss, the value (critic) loss, and an entropy bonus:

```math
\large \mathcal{L} = \mathcal{L}_{\text{policy}} + c_1 \mathcal{L}_{\text{value}} - c_2 \mathcal{L}_{\text{entropy}}
```

where

```math
\begin{gather}
\mathcal{L}_{\text{policy}} = -\log \pi_\theta(a_t \mid s_t) \cdot \text{sg}(A_t) \\
\mathcal{L}_{\text{value}} = \frac{1}{2} \left( R_t - V_\phi(s_t) \right)^2 \\
\mathcal{L}_{\text{entropy}} = -\sum_{a} \pi_\theta(a \mid s_t) \log \pi_\theta(a \mid s_t)
\end{gather}
```

where

```math
\begin{gather}
\text{sg}(A_t) = \text{stop-gradient (detach) applied to the advantage so policy gradients do not flow into the critic} \\
c_1 = \text{value loss scaling coefficient (typically } 0.5\text{)} \\
c_2 = \text{entropy bonus coefficient (typically } 0.01\text{)}
\end{gather}
```

The entropy term measures the randomness of the action distribution $\pi_\theta(\cdot \mid s_t)$. Subtracting entropy from the loss (maximising entropy) penalises the actor when its distribution collapses to a single deterministic action too early, forcing the agent to keep exploring.

---

## Vectorised Environments (A3C vs A2C)

Because A2C is an **on-policy** algorithm, the policy distribution $\pi_\theta(a \mid s)$ used to sample actions must match the policy being updated. This means we cannot use an experience replay buffer like in DQN, as past transitions in a buffer were generated by older weights. However, training on sequential transitions from a single environment causes the training batch to be heavily correlated, leading to unstable updates.

To decorrelate the training data without a replay buffer, we run $N$ independent instances of the environment in parallel. Since each worker is in a different state of the environment at any given timestep, a batch collected across all $N$ workers simultaneously contains diverse, uncorrelated transitions.

### A3C (Asynchronous Advantage Actor-Critic)
The original paper introduced A3C, which spawns $N$ independent CPU worker threads. Each worker holds a local copy of the network weights, steps through its own environment for $k$ steps, computes gradients locally, and asynchronously pushes those gradients to a global master network without locking.

The issue with A3C is **stale gradients**. By the time Worker 1 finishes its rollout and pushes its gradients to the global network, Workers 2 and 3 have already updated the global weights. Worker 1's gradients are therefore computed with respect to outdated parameters, adding noise to the optimisation and under-utilising GPU batching.

### A2C (Synchronous Advantage Actor-Critic)
A2C drops the asynchronous worker threads in favour of a single model on the GPU paired with synchronous, vectorised environments:

```text
Env 1:  s1 ──► [               ] ──► a1 ──► step(a1) ──► (s1', r1)
Env 2:  s2 ──► [ Policy / Val  ] ──► a2 ──► step(a2) ──► (s2', r2)
  ⋮            [ Network (GPU) ]       ⋮          ⋮            ⋮
Env N:  sN ──► [               ] ──► aN ──► step(aN) ──► (sN', rN)
```

Instead of maintaining $N$ separate network copies, A2C stacks the observations from all $N$ environments into a single batched tensor of shape $(N, d_{\text{obs}})$:
- The network performs a single batched forward pass on the GPU to sample $N$ actions.
- All $N$ environments step forward in lockstep for $k$ timesteps, producing a rollout buffer of shape $(k, N)$.
- A synchronisation barrier waits for all workers to finish the $k$-step rollout.
- Returns $R_t$ and advantages $A_t$ are computed across the $(k \times N)$ batch, and a single averaged backward pass updates the network weights.
- The rollout buffer is discarded, and the next $k$-step rollout begins with the updated policy.

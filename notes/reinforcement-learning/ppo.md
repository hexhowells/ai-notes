# Proximal Policy Optimisation (PPO)

> Original paper: [Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347)

## TRPO
TRPO solves a problem in RL of how to make the largest improvement step without breaking the policy. In supervised training of neural networks, one bad weight update can introduce error into the network, but this is quickly corrected in the next forward/backwards passes. However with RL since the model weights affects the data collected, a bad weight update can cause the model to get stuck in a bad environment state and collect no more useful data to recover. TRPO attempts to solve this by limiting how much the policy can change with each weight update. This could be done by limiting the actual weight deltas, but this would require choosing some fixed value and we would potentially be hindering the training performance, instead TRPO does this in behaviour-space, measuring the behavioural change of the model.

Suppose our current policy has weights $\theta_{\text{old}}$, and we want to find new weights $\theta$. We first need to score how much better or worse $\theta$ is compared to $\theta_{\text{old}}$. This can be done with the following

$$L(\theta) = \hat{\mathbb{E}}_t \left[ \frac{\pi_\theta(a_t \mid s_t)}{\pi_{\theta_{\text{old}}}(a_t \mid s_t)} \hat{A}_t \right]$$

where

If $\hat{A}_t > 0$ (good action)
- we want to make $\pi_\theta(a_t \mid s_t)$ bigger than $\pi_{\theta_{\text{old}}}(a_t \mid s_t)$

If $\hat{A}_t < 0$ (bad action)
- we want to make $\pi_\theta(a_t \mid s_t)$ smaller

<br>
To ensure the update is safe, we measure how the action probability shifts using KL divergence. This KL divergence is forced to stay under $\delta$ (usually 0.01).

$$\underset{\theta}{\text{maximize}} \; L(\theta) \quad \text{subject to} \quad \bar{D}_{\text{KL}}(\theta_{\text{old}} \parallel \theta) \le \delta$$

Whilst TRPO works effectively, it requires large computation to perform (computing second derivates, running conjugate gradients, etc).

## PPO
Proximal Policy Optimisation (PPO) aims to fix the computational issues of TRPO by performing the same checks, but with only first order operations, greatly reducing the computation required.

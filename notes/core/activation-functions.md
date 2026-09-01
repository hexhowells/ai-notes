# Activation Functions

## Sigmoid
Classic activation function, not used much in modern architectures. 

Range: (0, 1)

```math
\sigma(x) = \frac{1}{1 + e^{-x}}
```

## Tanh
Similar to Sigmoid, but increases the range to negative values.

Range: (-1, 1)

```math
\tanh(x) = \frac{e^x - e^{-x}}{e^x + e^{-x}} = 2\sigma(2x) - 1
```

## ReLU (Rectified Linear Unit)

More computationally efficient, and avoids vanishing gradients.

Range: $[0, \infty)$
```math
\text{ReLU}(x) = \max(0, x)
```

## LeakyReLU
Similar to ReLU but allows small negative values to mitigate dying neurons due to zero gradients.

Range: $(-\infty, \infty)$

```math
\text{LeakyReLU}(x) = \max(\alpha x, x) = \begin{cases} x & \text{if } x \ge 0 \\ \alpha x & \text{if } x < 0 \end{cases} \quad (\text{typically } \alpha \approx 0.01)
```

## ReLU² (Squared ReLU)

Range: $[0, \infty)$
```math
\text{ReLU}^2(x) = (\max(0, x))^2
```

## PReLU (Parametric ReLU)
Includes learnable parameters for the negative slope. Allows single layers neurons to solve the XOR problem.

Range: $(-\infty, \infty)$

```math
\text{PReLU}(x) = \max(a x, x) = \begin{cases} x & \text{if } x \ge 0 \\ a x & \text{if } x < 0 \end{cases}
```

## GeLU (Gaussian Error Linear Unit)
GeLU merges the properties of ReLUs, dropout, and zoneout into a single activation function. With GeLU the input $x$ is multiplied by $m \sim \text{Bernoulli}(\Phi(x))$. As $x$ becomes more positive, it becomes more likely to be retained.

Instead of computing $m \sim \text{Bernoulli}(\Phi(x))$ which is more computational and introduces stochasticity. We can represent it using the Gaussian Cumulative Distribution Function $\Phi(x)$

```math
\text{GELU}(x) = x \cdot \Phi(x) = x \cdot P(X \le x), \quad X \sim \mathcal{N}(0, 1)
```

## SELU (Scaled Exponential Linear Unit)
SELU was introduced as a self-normalising activation function. SeLU is designed such that if you pass a distribution of activations with a mean of 0 and variance of 1 through it, the output will naturally preserve a mean of 0 and variance of 1. This effectively eliminates the requirement of batch normalisation.

```math
\text{SELU}(x) = \lambda \begin{cases} x & \text{if } x > 0 \\ \alpha (e^x - 1) & \text{if } x \le 0 \end{cases} \\
```
where
```math
\begin{gather}
\lambda \approx 1.0507 \\
\alpha \approx 1.6732
\end{gather}
```

## Swish (SiLU)
```math
\text{Swish}(x) = x \cdot \sigma(\beta x)
```

## SwiGLU (Swish Gated Linear Unit)
```math
\text{SwiGLU}(x, W, V, b, c) = \text{Swish}(xW + b) \otimes (xV + c)
```

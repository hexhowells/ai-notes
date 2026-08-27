# Statistics

## Bernoulli distribution
The Bernoulli distribution is a discrete probability distribution of a random variable $X \in \{0, 1\}$ of an individual success probability $p \in [0, 1]$ and failure probability $q = 1 - p$

## Binomial Distribution
The Binomial Distribution models the number of successes $k$ across $n$ independent Bernoulli trials.

If $X_1, X_2, \dots, X_n \stackrel{\text{i.i.d.}}{\sim} \text{Bernoulli}(p)$, then:

```math
K = \sum_{i=1}^n X_i \sim \text{Binomial}(n, p)
```

The probability mass function gives the probability of observing exactly $k$ successes in $n$ trials:

```math
P(K = k) = \binom{n}{k} p^k (1 - p)^{n - k} \quad \text{for } k \in \{0, 1, \dots, n\}
```

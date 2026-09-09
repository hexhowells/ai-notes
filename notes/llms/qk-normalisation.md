# QK Normalisation

> Original paper: [Query-Key Normalization for Transformers](https://arxiv.org/abs/2010.04245)

QK normalisation is a modification to the standard attention mechanism designed to solve instabilities in training (usually models with large parameter counts)

In the standard attention mechanism, pre-softmax logits are computed as

```math
A = \frac{QK^T}{\sqrt{d_k}}
```

The scaling factor $\sqrt{d_k}$ is designed to keep the variance of the logits at ~1. However, during training the learned weights can cause the magnitudes of $Q$ and $K$ to grow disproportionally. This causes the dot product $QK^T$ to produce extreme values, which pushes the softmax function to saturated regions, this in turn causes issues with the gradients and can result in loss spikes.

QK normalisation solves this by bounding the magnitudes of $Q$ and $K$ after the linear projects but before the dot product via a normalisation layer (LayerNorm or RMSNorm).

Given the standard projections $Q = XW_q$ and $K = XW_k$

```math
\begin{gather}
\bar{Q} = \text{Norm}(Q) \\
\bar{K} = \text{Norm}(K) \\
\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{\bar{Q}\bar{K}^T}{\tau}\right)V
\end{gather}
```

Here the dot product $\bar{Q}\bar{K}^T$ is mathematically equivalent to the cosine similarity scaled by $d_k$.

Using QK normalisation, the denominator $\sqrt{d_k}$ is usually replaced by a learnable scalar value $\tau$. Since the cosine similarity is bounded between $[-1, 1]$, feeding this into softmax results in a high entropy uniform distribution. This distribution needs to be sharpened when the model is more confident. Thus the learnable parameter allows the optimiser to control the softmax entropy dynamically during training.

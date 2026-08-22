# Transformers

The core objective of an autoregressive language model is the joint probability of a full sequence x (comprised of n tokens) as the product of the conditional probabilities of each token s_i, given all preceding tokens.

$$
p(x) = \prod_{i=1}^{n} p(s_i \mid s_1, \dots, s+{i-1})
$$


![transformer architecture](https://github.com/hexhowells/ai-notes/blob/main/images/llms/transformer.png)


#### Token embedding
Converts the one-hot encoded token vectors into learned embeddings.

$$
X_{embed} = E \cdot T
$$

Where,

```math
\begin{gather}
T \in \mathbb{R}^{d_{batch} \times d_{seq}} \\
E \in \mathbb{R}^{d_{\text{vocab}} \times d_{\text{embed}}} \\
X_{embed} \in \mathbb{R}^{d_{batch} \times d_{seq} \times d_{embed}}
\end{gather}
```
<br>
Add in position encoding so the LLM knows the tokens position in the text

```math
X_{embed} = X_{embed} + X_{pos}
```

#### Attention mechanism

```math
\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}} + M\right)V
```

```math
\begin{gather}
\sqrt{d_{k}} = \text{scaling factor to prevent vanishing softmax gradients} \\
M = \text{masking matrix}
\end{gather}
```


#### Multi-headed attention
```math
\begin{gather}
\text{MultiHead}(Q, K, V) = \text{Concat}(\text{head}_1, \dots, \text{head}_h)W^O \\
\text{where } \text{head}_i = \text{Attention}(QW_i^Q, KW_i^K, VW_i^V)
\end{gather}
```

Feature embedding is split into three projection matrices (Q, K, V)
```math
\begin{gather}
W_{i}^{Q} \in \mathbb{R}^{d_{model} \times d_{K}} \\
W_{i}^{K} \in \mathbb{R}^{d_{model} \times d_{K}} \\
W_{i}^{V} \in \mathbb{R}^{d_{model} \times d_{v}} \\
and \\
W_{i}^{O} \in \mathbb{R}^{hd_{d} \times d_{model}}
\end{gather}
```

#### Residuals and Normalisation
The original transformer architecture used post-later normalisation, all modern implementations perform normalisation before the MHA & linear layers.

Inputs to layers are also added to layer outputs via residual connections. The residual can be thought of as a single trunk that connects the input embedding through all the blocks into the output embedding. The MHA and feed-forward layers read and write to the trunk, modifying it. The primary purpose of the MHA layer is to manipulate the attention between tokens, and the feed-forward layers perform the "computation", deriving certain logic from the selected tokens / embeddings.




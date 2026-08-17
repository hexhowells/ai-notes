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
T \in \mathbb{R}^{d_{\text{vocab}} } \\
E \in \mathbb{R}^{d_{\text{vocab}} \times d_{\text{embed}}}
\end{gather}
```

#### Attention mechanism

```math
\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}} + M\right)V
```


#### Multi-headed attention
```math
\begin{gather}
\text{MultiHead}(Q, K, V) = \text{Concat}(\text{head}_1, \dots, \text{head}_h)W^O \\
\text{where } \text{head}_i = \text{Attention}(QW_i^Q, KW_i^K, VW_i^V)
\end{gather}
```



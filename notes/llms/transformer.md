# Transformers

The core objective of an autoregressive language model is the joint probability of a full sequence x (comprised of n tokens) as the product of the conditional probabilities of each token s_i, given all preceding tokens.

$$
p(x) = \prod_{i=1}^{n} p(s_i \mid s_1, \dots, s+{i-1})
$$

![transformer architecture](https://github.com/hexhowells/ai-notes/blob/main/images/llms/transformer.png)

# Positional Encodings

Positional encodings are added to token embeddings to tell the LLM the position of each token given a token sequence.

## Learned Position Embedding
Uses a learned weight matrix (of shape $len_{seq} \times d_{model}$) that maps each absolute position index to a dense vector. These position encoding is then added to the token embedding $x_i = e_i + p_i$.

Since the embeddings are trained on a fixed context size, it does not allow models to process token sequences larger than the pre-trained size.


## Absolute Sinusoidal Positional Encodings
A basic way to try and encode position to each token is to assign a value to each position (1, 2, 3, ...) - this fails though as a token in position 1000 has a position value of 1000, which is much larger than the token embedding values. We could instead normalise values to be between [0, 1] - but the scale of the positional encoding values will change depending on how many tokens there are (more tokens = smaller step sizes), this makes it impossible for a model to learn without a consistent position/distance metric.

In order to keep the positional values between a certain range (-1 and 1) and keep the step size between values the same, we can use a sine function to encode the positional value. Since a single sine function will wrap around on itself too quickly, causing repeating values, we stack N sine functions together with different wavelengths. High frequency wavelengths change values quickly over positions, whilst lower frequency wavelengths change slowly. These position encodings are also coupled with cosine functions, since the sine value of 0.5 doesn't give the model any information on what direction the value is going, the addition of the cosine value gives this information as they are out of phase by $\frac{\pi }{2}$.

For a given position in the sequence, we generate a positional vector $PE$ of dimension $D_{model}$. The $PE$ vector is constructed by combining sine and cosine functions. given a dimension index $i$ where $i \in [0, \dots, d_{\text{model}}/2)$

```math
PE_{(pos, 2i)} = \sin\left(\frac{pos}{10000^{2i/d_{\text{model}}}}\right)
```

```math
PE_{(pos, 2i+1)} = \cos\left(\frac{pos}{10000^{2i/d_{\text{model}}}}\right)
```

where

```math
\begin{gather}
pos = \text{absolute index of the token in the sequence (0, 1, 2, ...)} \\
i = \text{index of the dimension in embedding space} \\
10000^{2i/d_{\text{model}}} = \text{the wavelength factor, as i increases the wavelength gets exponentially larger and the frequency drops}
\end{gather}
```

## Rotary Positional Embeddings (RoPE)
The primary issue with sinusoidal position encoding is that the position was added to the token embedding $embedding(x) + position(m)$. The token embedding stores semantic meaning and by adding position information that semantic meaning is diluted and needs to be pulled apart by downstream transformer layers. The position vectors also add noise to the attention computation as you get the dot product of the meaning of one token with the position of another. Take a query $q$ at position $m$ and key $k$ at position $n$:
```math
(q + p_m) \cdot (k + p_n) = (q \cdot k) + (q \cdot p_n) + (p_m \cdot k) + (p_m \cdot p_n)
```

$q \cdot k$ represents semantic similarity, $p_{m} \cdot $p_{n}$ represents positional similarity. However, $q \cdot p_n$ and $p_m \cdot k$ is just noise that the model has to learn to ignore.

---

To fix this issue, RoPE applies the same rotation operation, but instead of applying it to additional positional encodings, the rotations are applied to the query and key vectors directly. Given $d_{model} = 1024$ - this vector represents the coordinates of a 1024 vector space, we can simply apply the rotational operation to this coordinate. Since during the attention computation we compute $Q \cdot K$, we must rotate both vectors. By rotating each tokens embedding by its absolute position, the dot product of $Q$ and $K$ subtracts the two absolute rotations and just leaves the relative distances between each vector.

Given two tokens `[cat]` and `[sat]`. Assume the Query value of `[cat]` points to 10 degrees and Key value of `[sat]` points to 20 degrees. The angle between them is 10 degrees, representing their semantic similarity.

Now applying a single plane of the RoPE embedding, assume the frequency $\theta$ is 15 degrees per step:

- If $Q$ is at position $m = 3$, it gets rotated by $3 \times 15 = \text{45 degrees}$. Thus its new angle is $10 + 45 = \text{55 degrees}$
- If $K$ is at position $n = 5$, it gets rotated by $5 \times 15 = \text{75 degrees}$. Thus its new angle is $20 + 75 = \text{95 degrees}$

During the attention mechanism, we take the dot product of $Q$ and $K$, the dot product will only care about the new angle between them:
```math
95 - 55 = \text{40 degrees}
```

Thus the absolute positions are removed during the dot product, and the only positional information that remains is the distance between each token, which is the information we care about.

#### Proof
Since rotation matrices are orthogonal ($R^T R = I$), the transpose of a rotation matrix is equivalent to rotating in the opposite direction $(R_{\Theta, m}^d)^T = R_{\Theta, -m}^d$

Thus computing the attention score between $Q$ and $K$, the dot product of the rotated vectors reduces to the relative distance:

```math
(R_{\Theta m}^d q)^T (R_{\Theta n}^d k) = q^T (R_{\Theta m}^d)^T R_{\Theta n}^d k = q^T R_{\Theta n-m}^d k
```


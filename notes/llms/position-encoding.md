# Positional Encodings

Positional encodings are added to token embeddings to tell the LLM the position of each token given a token sequence.

A basic way to try and encode position to each token is to assign a value to each position (1, 2, 3, ...) - this fails though as a token in position 1000 has a position value of 1000, which is much larger than the token embedding values. We could instead normalise values to be between [0, 1] - but the scale of the positional encoding values will change depending on how many tokens there are (more tokens = smaller step sizes), this makes it impossible for a model to learn without a consistent position/distance metric.

## Absolute Sinusoidal Positional Encodings

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

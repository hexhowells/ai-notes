# Vision Transformers

> Original paper: [An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale](https://arxiv.org/abs/2010.11929)

Vision transformers (ViT) aim to apply the transformer architecture to image data instead of textual data. It uses largely the same architecture as a language transformer, but requires modifications of the input to handle images.

Given an image

```math
x \in \mathbb{R}^{H \times W \times C}
```

Reshape it into a sequence of flattened 2D patches

```math
x_{p} \in \mathbb{R}^{N \times (P^2 \cdot C)}
```

Where $(H, W)$ is the resolution of the original image, $C$ is the number of channels, $(P,P)$ is the patch resolution, and $N = HW / P^2$ is the number of patches (and sequence length of the transformer).

Patches are mapped to the latent vector size $D$ of the transformer using a linear projection layer.

## Position Embedding
A learnable position embedding is added to each patch embedding to retain positional information. The ViT paper uses a standard learnable 1D positional embedding.

```math
\begin{gather}
z_{0} = [x_{class}, x_{patch[1]}E, ..., x_{patch[N]}E] + E_{pos} \\
E_{pos} \in \mathbb{R}^{(N+1) \times D}
\end{gather}
```

## Output Embeddings & Classifications
Given the input sequence of shape $(B, N, D)$ is retained throughout the model into the output sequence. There are $N$ token distributions in the output. However, classification requires a single 1D vector of size $D$, to resolve this a special `[class]` token is prepended to the input sequence. In the final layer $L$, the 0-th token is isolated $(z_{L}^{0})$ and passed to the classification head, the output spatial patch tokens $(z_{L}^{1}...z_{L}^{N})$ are ignored. The class token is used to incrementally aggregate global information into the single output token via self-attention.

#### Example
```
Input image
# # # @
@ @ @ #
# @ # #
@ # # @

Patches
# # | # @
@ @ | @ #
----+----
# @ | # #
@ # | # @

# Flattened patch tokens
[# # @ @], [# @ @ #], [# @ @ #], [# # # @]
```

## Resolution scaling
Scaling up the input image after pre-training is simple in terms of adding more patches (assuming the same patch size), resulting in the sequence length increasing, which is easily handled by a transformer. However, adding more patch tokens corrupts the information in the position embeddings (since for example, the position of the bottom right token is no longer representing the bottom right patch if more patches have been added).

To resolve this the position embedding is spatially scaled with bicubic interpolation. Given the position embedding $E_{pos} \in \mathbb{R}^{(N+1) \times D}$, isolate the spatial tokens (remove the class token) $E_{pos} \in \mathbb{R}^{(N) \times D}$.

You can then reshape the spatial embeddings into a grid

```math
E_{grid} = Reshape(E_{spatial}) \in \mathbb{R}^{D \times \sqrt{N} \times \sqrt{N}}
```

Given the 2D grid of spatial tokens, apply a 2D bicubic interpolation function to increase the resolution to the new number of patches $M$.

```math
E_{scaled} = Interpolate_{bicubic}(E_{grid}) \in \mathbb{R}^{D \times \sqrt{M} \times \sqrt{M}}
```

The scaled grid can then be flattened and the class token prepended.

```math
\begin{gather}
E_{pos[scaled]} \in \mathbb{R}^{(M+1) \times D} \\
where, \quad M = HW / p^2
\end{gather}
```

Since bicubic interpolation acts as a low-pass filter, upscaling the grid averages out the sharp, high-frequency boundaries between patches. The higher you try to scale the positional embeddings, the wider this averaging becomes, and since the pre-trained attention heads are optimised for high frequency values, this smoothing causes local spatial boundaries to "smear". As such, scaling the resolution this way only works so far, scaling the image too high causes the delta to become too large, drastically effecting performance.


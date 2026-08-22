# Vision Transformers

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

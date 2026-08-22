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

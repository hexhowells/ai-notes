# Data Augmentation

## Mixup
Mixup takes two random images from a batch, blends their pixel values together and blends their labels proportionality.

Draw two random samples from a batch $(x_i, y_i)$ and $(x_j, y_j)$ and create a virtual training sample $(\tilde{x}, \tilde{y})$ using convex combinations

```math
\begin{gather}
\tilde{x} = \lambda x_i + (1 - \lambda) x_j \\
\tilde{y} = \lambda y_i + (1 - \lambda) y_j
\end{gather}
```

the mixing coefficient $\lambda \in [0, 1]$ determines the blending ratio between the two images. 

If $\lambda = 0.7$, then $\tilde{x}$ is 70% of image $i$ and 30% of image $j$. The target label $\tilde{y}$ then becomes a probability distribution: (e.g `[0.7 for Dog, 0.3 for Cat]`).

## CutMix
CutMix also creates virtual training samples $(\tilde{x}, \tilde{y})$, but instead of blending two images together, a bounding box is cut out of one image and pasted onto the other image.

To cut out a section of one image, a bounding box $B = (r_x, r_y, r_w, r_h)$ is created. Coordinates for the box are uniformly sampled, the size of the box is constrained so that the area of the box relative to the image size $(W, H)$ is equal to $1 - \lambda$.

```math
\frac{r_w r_h}{W H} = 1 - \lambda
```

To combine the images, a binary mask $M \in \{0, 1\}^{W \times H}$ is applied. Where pixels inside the boxing box are 0 and pixels outside the box are 1. This mask is applied to the image receiving the bounding box $x_i$, the inverse mask is applied to the image where the bounding box is taken from $x_j$. After masking the image pixels can be combined from both images to make the new virtual image.

```math
\tilde{x} = M \odot x_i + (1 - M) \odot x_j
```

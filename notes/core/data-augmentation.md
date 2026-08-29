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

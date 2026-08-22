# KV Caching

LLM inference natively requires $O(T^2)$ compute as to predict $y_{t}$ we need to re-evaluate all past tokens. KV caching reduces this to $O(T)$ by caching K and V for each layer, as under causal masking these states remain static.

Given the equation for causal masked attention

```math
\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}} + M\right)V
```

We can see the relationship between Q and K during their matrix multiply
```text

Q11 Q12 Q13     K11 K21 K31     A11 A12 A13        A11  .   .
Q21 Q22 Q23  x  K12 K22 K32  =  A21 A22 A23  Mask  A21 A22  .
Q31 Q32 Q33     K13 K23 K33     A31 A32 A33        A31 A32 A33

Calculation for the 3rd row:
A_31 = (Q_31 * K_11) + (Q_32 * K_12)
A_32 = (Q_31 * K_21) + (Q_32 * K_22)
A_33 = (Q_31 * K_31) + (Q_32 * K_32)
```

From the above, we can see that all rows of K are used to compute the last row of attention scores (for the next predicted token), however only the last row of Q (the most recent token) is used. V carries the same dependencies as K.

Following this, the other rows of the attention matrix (and the output of the MHA layer) do not change as due to the causal mask, they are not affected by new tokens. So when these rows are propagated into the next block, they are used to compute that block's K and V values... which have not changed (and are already cached). Hence we do not need to compute the other rows of the attention block (and thus only need the last row / token from Q).

As such, the matrix multiply between Q and K for the attention layer become

```text

 .   .   .      K11 K21 K31      .   .   .
 .   .   .   x  K12 K22 K32  =   .   .   .   Mask
Q31 Q32 Q33     K13 K23 K33     A31 A32 A33        A31 A32 A33
```

So we only compute the attention scores for the most recent token (which require the last row of Q, but all rows for K and V).

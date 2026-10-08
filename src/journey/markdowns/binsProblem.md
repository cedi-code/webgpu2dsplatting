## Formalizing Bins Problem

We have $M$ splats and $N^2$ boxes. Our boxes are arranged in a $N \times N$ grid. let $(j, k) \in \{ 1, \dots, N \} \times \{ 1, \dots, N \}$ be the $j,k$-th bin and $i \in \{ 1, \dots, M \}$ be the index of $i$-splat.

### Given

We are given coordinate ranges for each splat $i$

$$
splat[i] = ( l_i, r_i, t_i, b_i)
$$
such that:
$$
  i \in \text{box}[j,k] \iff l_i \leq j \leq r_i \quad \textrm{and} \quad  b_i \leq k \leq t_i 
$$
 which describes if splat $i$ is in box $j,k$:

### Goal

we want to accumulate all the splats $i$ that belong to box $j,k$:
$$
box[j,k] = \{ i \in \{1,\dots, M\} \quad | \quad l_i \leq j \leq r_i \land   b_i \leq k \leq t_i  \}
$$

note that $M >> N$ and that the resulting size of the boxes is quite sparse $|box[j,k]| << M$ 


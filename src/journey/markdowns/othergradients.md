## Other Gradients
Well now we can fit our 2d gaussians position and scale on a image.
For more complex images we will also need to optimize over the other gaussian splat properties:
- rotation (scalar)
- color (3d vector)
- alpha (scalar)

### Rotation
Like in the previous chapter [Transforming Gaussians](), we can define our variance to have following shape:
$$
\Sigma = RSS^TR^T \quad \quad \Sigma^{-1} = RS^{-2}R^T
$$
which changes our parameterization to
$$
R(r) = \begin{bmatrix}
cos(r) & -sin(r) \\
sin(r)  & cos(r)
\end{bmatrix} 

\quad \quad \Sigma(s,r) = R(r) \begin{bmatrix}
\sigma(s_x)^2 & 0 \\
0  & \sigma(s_y)^2 
\end{bmatrix} R(t)^T
$$
where $R(r)$ constructs a rotation matrix, that rotates a 2d vector by $r$-degrees counterclock wise.

Great we would just need to take the gradient of $r$ then and also adapt the gradients of $\mu$ and $s$.

$$
\frac{\partial}{\partial r}g(x) = ? 
$$

before takling the full derrivation I would give this "trick" as a hint:

$$
\begin{align*}
\frac{\partial}{\partial r}R(r) &= \frac{\partial}{\partial r}\begin{bmatrix} 
cos(r) & -sin(r) \\
sin(r)  & cos(r)
\end{bmatrix} \\
&= \begin{bmatrix} 
-sin(r) & -cos(r) \\
cos(r)  & -sin(r)
\end{bmatrix} \\

&= \underbrace{\begin{bmatrix} 
0 & -1 \\
1 & 0
\end{bmatrix}}_L

\underbrace{
\begin{bmatrix} 
cos(r) & -sin(r) \\
sin(r)  & cos(r)
\end{bmatrix}}_R \\ \\
&= LR(r) \\
\end{align*} 
$$

<details>
  <summary>Show derrivation</summary>
For ease of notation i will use:
- $R(r) = R$
-  $S^{-2}(r) = S$
-  $(x-\mu) = d$.
- $\Sigma^{-1}(s,r) = \Sigma$ (yes inverse hidden)


$$
\begin{align*}

\frac{\partial}{\partial r}g(x) &= \frac{\partial}{\partial r} e^{-\frac{1}{2}d^T\Sigma^{-1}(s,r)d} & \\

 &= \frac{1}{2}g(x) \frac{\partial}{\partial r} d^T(RSR^T)d & \\
 &= c * d^T( \frac{\partial}{\partial r} RSR^T)d & \\
 &= c * d^T(LRSR^T + RSR^T L^T)d & \\
 &= c * d^T(LRSR^T + RS(-R^T L)d & \\
 &= c * d^T(L\Sigma - \Sigma L)d & \\
 &= c * (d^T L\Sigma d - d^T \Sigma L d) \\
 R^Td = v & \\
 &= c * (d^T LR S v - v^T S R^TL d) \\
 &= c * (d^T (RL) S v - v^T S (LR^T) d) \\
 &= c * (v^T LS v - v^T SL v^T) \\
 &= c * v^T (LS - SL) v \\
 &= c * v^T ( \begin{bmatrix} 
0 & -s^2_y \\
s²_x  & 0
\end{bmatrix} - \begin{bmatrix} 
0 & -s^2_x \\
s²_y & 0
\end{bmatrix}) v \\
 &= c * v^T \begin{bmatrix} 
0 & s²_x-s^2_y \\
s²_x - s^2_y & 0
\end{bmatrix} v \\

&= c * (
(s²_x - s^2_y) v_1 v_2 + (s²_x - s^2_y) v_2 v_1  ) \\
&= 2c (s²_x - s^2_y) v_1 v_2

\end{align*}
$$

plugging back the notations we get:

$$
\frac{\partial}{\partial r}g(x) = g(x) (s²_x - s^2_y) v_1 v_2 \quad \quad \begin{bmatrix} 
v_1 \\
v_2
\end{bmatrix} = R^T(x-\mu)
$$

</details>
## Testing

### Gradients Checking

To make sure my gradients are computed correctly, I wanted to compare them to the numerical approximation. Sampling a bunch of random  2d points $x_i$ i check the following:

$$
\nabla f^{?}(x_i) \approx \nabla \^{f}(x_i)
$$

where $\nabla f^{?}(x_i)$ is my derrivation and $\nabla \^{f}(x_i)$ is the numerical approximation. For here i will reduce it to one dimension but the approximation has to hold for each of the $k$-dimensions $\frac{\partial}{\partial x_k}f^{?}(x_i)$

$$
f'^{?}(x_i) \approx \^{f'}(x_i)
$$

The numerical approximation is just the definition of the differentation but with a finite step $h$:

$$
f'(x_i) = \frac{f(x_i + h) - f(x_i)}{h} - O(h)
$$

Which we then can use for testing if our derrivation matches the approximation.checking if each partial matches with its approx

$$
f'^{?} \text{ is correct} \implies |f'^{?}(x_i) - \frac{f(x_i + h) - f(x_i)}{h} | \leq O(h) 
$$


To make the approximation more accurate, we can just make our $h$ smaller? sadly this wont work since floating points can not store arbitrary precicion...
- error term $\downarrow h$ $ \implies \downarrow$ formula error + $\uparrow $ floating point error 
- error term $\uparrow h$ $\implies$ $\downarrow $ floating point error + $\uparrow$ formula error 
<p><a href="https://commons.wikimedia.org/wiki/File:AbsoluteErrorNumericalDifferentiationExample.png#/media/File:AbsoluteErrorNumericalDifferentiationExample.png"><img src="https://thumb.wikimedia.org/wikipedia/commons/thumb/4/41/AbsoluteErrorNumericalDifferentiationExample.png/500px-AbsoluteErrorNumericalDifferentiationExample.png?utm_source=en.wikipedia.org&amp;utm_campaign=imageinfo&amp;utm_content=thumbnail" alt="AbsoluteErrorNumericalDifferentiationExample.png" height="295" width="500"></a></p>
so we have to find a good error term O(h) that balances these two.

and luckly, wikipedia article about [Numerical differentiation](https://en.wikipedia.org/wiki/Numerical_differentiation) has a formula for that:

$$
|f'(x_0) - \frac{f(x_0 + h) - f(x_0)}{h} | \leq \frac{h}{2}M_1 + \frac{2\epsilon}{h}M_2
$$

where variables mean the follwoing:

* $h \quad$ small step size which we will use $10^{-4}$ 
* $\epsilon \quad$ machine epsilon, for which i will use $2^{-23}$ [(binary32 Interval machine eps)](https://en.wikipedia.org/wiki/Machine_epsilon)
* $M_1 \quad$ upper bound of $f''(x)$ in the range of $x \in [x_0, x_0+h]$

* $M_2 \quad$ upper bound of $f(x)$ in the range of $x \in [x_0, x_0+h]$

Oh wait, to get $O(h)$ we need $f''(x_i)$ which is a bit ironic since we are doing this to test if our derrivatoin $f'^{?}(x_i)$ is correct.
The reason i still go with this method is that its asking for the upper bound, so the derrivation of $f''$ just has to be assymptotically correct!

A example $M_1$ for the function $ \frac{\partial }{\partial \mu_k}g(x)$ where $g(x;\mu) = e^{-\frac{1}{2}(x-\mu)^T\Sigma^{-1}(x-\mu)}$ I approximate the following:

$$
\begin{align*}

\frac{\partial }{\partial \mu_k^2} g(x_i ; \mu) & \leq c *\frac{\partial }{\partial \mu_k} g(x_i ; \mu)s_x^{2}(x_{ix}-\mu_k) \\

&\leq c * g(x_i ; \mu)s_x^{4}(x_{ix}-\mu_k)^2 \\
& \leq M_1
\end{align*} 
$$

For some large enough $c$ i can then use this for my upper bound $M_1$.

For $M_2$ its easier since $g(x)$ is bounded by $1$ but to be a bit more accurate i take the maximum of the edge cases:

$$
M_2 = c * max(g(x_i;\mu_0), g(x_i;\mu_0+he_k))
$$


Now for bellow one can execute the tests for $N$ samples $x_i$ and random parameters for a 2d gaussian $g(x)$:

#### Test $\mu$ 
- Numerical estimate for dimension $k$:
    
$$
\Delta_k = \frac{g({x}_i; {\mu}_0 + h{e}_k) - g({x}_i; {\mu}_0)}{h} 
$$

- my gradient for dimension $k$:

$$
g_k'^{?} = \frac{\partial}{\partial \mu_k}g({x}_i; {\mu}_0)
$$

- my truncation error:

$$
T_k = \frac{h}{2}M_{1} + \frac{2\epsilon}{h}M_2
$$

- my Test for $\nabla_{\mu}g^{?}(x;\mu_0)$:

$$ |g_1'^{?}  - \Delta_1|  \leq T_1   \land  |g_2'^{?}  - \Delta_2| \leq T_2 \implies \nabla_{\mu} g^?({x}_i; \mu_0) \text{ is valid}$$

Test $s$ and Test $r$ are the same but with $s_0$ (scaling term) and $r_0$ (rotation term).
$\Delta_k, g_k'^{?}, T_k$ are computed in wgsl, and stored in a buffer which we read back and evaluate:



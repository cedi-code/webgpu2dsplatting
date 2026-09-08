## Optimization Step
We can now draw splats, now lets try to make them fit to our images

This again I will break down in substeps, to slowley arrive at our goal.

### Optimization Step
* [ ] gradient descent + comute shaders
* [ ] Gradient over a Image? (+ Texture loading)
* [ ] SGD with multiple gaussians
* [ ] Backpropagation
* [ ] Combining all the losses

Lets start with a simple example

###### *Loss graph visualization is powerd by [tweakpane](https://tweakpane.github.io/docs/)*

## gradient descent (and compute shaders)

First, what do we want to optimize? When talking about optimization we mean optimization of a function $f(x)$ for which we have some data: inputs $x$ and wanted outputs $y^*$. We assume our function to have a certain "shape" here already but we dont know the exact values of that shape, these "values" are our parameters we want to optimize for $\theta$.

find the optimal $\theta$ such that $f(x ; \theta) = y \approx y^*$

### Example
Lets start with a simple example, the function we are trying to optimize has the following shape:

$$
f(x) = e^{-(x-\mu)^2}
$$

the goal output we want our function $f$ to have is:

$$
y^*(x) = e^{-(x-2)^2} \implies \mu^* = 2
$$

$N$-inputs $x_i$ and outputs $y_i$ are evenly spaced points form the interval [-5, 5]

$$
x_i = \frac{10}{N}i - 5 \quad y_i = y^{*}(x_i), \quad \text{for } i \in \{0, 1, ..., N\}
$$


We can measures how similar our function $f$ to $y^*$ is by defining a loss function (L2-norm):

$$
L(\mu) = \frac{1}{N+1}\sum_{i=0}^{N}(f(x_i; \mu) - y_i)^2
$$

We start with a random guess for our parameter $\mu_0 \in [-10, 10]$. Our goal now is to find the optimal $\mu^{*}$ that approximates our $y_i$ the best, this can be done by minimizing our loss function $L$, so our new goal is:

$$
\operatorname*{arg\,min}_{\mu \in \mathbb{R}} L(\mu)
$$

We now do itterative optimization using gradient descent, meaning updating our inital guess $\mu$ in the direction the the gradient $\nabla L$ with a certain stepsize $\eta$. 

$$
    \mu_{t+1} = \mu_t - \eta * \nabla L(\mu_{t})
$$
we keep updating until the gradient $\nabla L(\mu_{t}) \approx 0$ reaching a local minima.

in wgsl gradient descent could look like this:
```wgsl
...
@compute @workgroup_size(1) fn gradientDescent() {
    let q = rand() * 8.0 - 4.0; // inital guess
    loop {
        let step = gradL(q);
        q -= n * step;
        break if (step >= 0.01);
    }

    workResult[id] = q;
}
```

For setting up compute shaders one can follow the wonderful tutorial webgpufundamentals: [Run computations on the GPU](https://webgpufundamentals.org/webgpu/lessons/webgpu-fundamentals.html#a-run-computations-on-the-gpu).


Now what about the line `gradL(q)`? we still have to derrive this term. 

<details>
  <summary>Show derrivation</summary>

Since $L(\mu)$ is made out of a composition of functions, we have to apply the chain rule:

$$
\begin{align*}
\frac{d}{d\mu}L(\mu) & = \frac{d}{d\mu} \sum_{i=0}^{N}(f(x_i; \mu) - y_i)^2 &   \\
&= \sum_{i=0}^{N}\frac{d}{d\mu} (f(x_i; \mu) - y_i)^2        &   \\
&= \sum_{i=0}^{N}2(f(x_i; \mu) - y_i)f'(x_i;\mu)             &   \\
& \\

\frac{d}{d\mu}f(x;\mu) &=  \frac{d}{d\mu}  e^{-(x-\mu)^2} & \\
&= e^{-(x-\mu)^2}*-\frac{d}{d\mu}(x-\mu)^2 & \\
&= 2e^{-(x-\mu)^2}(x-\mu) & \\
\end{align*}
$$

putting it togheter we get:

$$
\nabla L(\mu) =  \frac{4}{N+1} \sum_{i=0}^{N}[(f(x_i; \mu) - y_i)f(x_i; \mu) (x_i-\mu)]
$$
</details>


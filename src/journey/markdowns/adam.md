## Adam
Having multiple parameters means we have a high dimensional solution space we are trying to optimize for, and trying to find the right stepsize $\eta$ will get increasingly more difficult. 

The 3dgs paper uses [adam (a method for stochastic optimization)](https://arxiv.org/pdf/1412.6980) which adaptivley finds a good stepsize using gradients of the function (gd with extra steps). 

I will try to explain it shortly in this article but I can highly recommend the [video](https://youtu.be/MD2fYip6QsQ?si=g-HwLIIYCsug1U39) by sourish kundu on the topic

### Basic GD setup

- $t$ time step, itteration
- $\alpha$ stepsize, just a constant we can modify
- $L$ our objective function we would like to minimize
- $\theta$ parameter ( which affects $L$ behaviour) we would like to find the optimal value for such that it minimizes $L$
- $\nabla L$ gradient of our objective function with respect to $\theta$

Wich gives us basic gradient descent update rule:
$$
    \theta_{t+1} = \theta_t - \alpha  \nabla L(\theta_t)
$$

### GD with Momentum
Instead of changing our step sized only based on the current evaluation of $\nabla L(\theta_t)$ we could take the average of the last couple updates, giving it the update rule acceleration term or *momentum* based on previous steps sizes.

For that we need to store the current *momentum* in a variable $M_t$
which contains averages of previous steps weighted by $\beta$ (parameter we can modify)

$$
M_{t+1} = \beta M_t + (1-\beta)\nabla L(\theta_t)
$$

$$
    \theta_{t+1} = \theta_t - \alpha M_{t+1}

$$

### GD with RMSProp

When optimizing over multiple dimensions in our parameter space, some gradients might be much bigger than others, taking much bigger steps, which can lead our current estimae of $\theta$ in a position where its almost flat for the other dimensions which did not have such big stepsizes in the beginning.

It would be nice to be able to damp the larger gradients such that all dimensions are updated somewhat "equally". For that we can use Root Mean Squared Propagation which kind of "normalizes" the graidents. This normalization term can be called "second moment estimate" and we call it $V$ here with hyperparameter $\beta$:

$$
V_{t+1} = \beta V_t + (1-\beta)\nabla L(\theta_t)^2
$$

update $\theta$ looks now as follows ($\epsilon$ is there so we dont divide by zero if gradient hits a minima or sattle point):

$$
    \theta_{t+1} = \theta_t - \alpha \frac{ \nabla L(\theta_t)}{\sqrt{V_{t+1}+\epsilon}}
$$

to illustrate the "normalization" a bit better one can plug in $\beta = 0$ and the update rule would change to:

$$
\begin{align*}
    \beta = 0 \implies \theta_{t+1} &= \theta_t - \alpha \frac{ \nabla L(\theta_t)}{\sqrt{\nabla L(\theta_t)²+\epsilon}} \\ 
    &\approx \theta_t - \alpha * sign(\nabla L(\theta_t))
\end{align*}

$$

### combine to get adam

Now lets use both these methods and combine them.
we have now the following:

hyperparameters:
- $\alpha = 0.01$ stepsize
- $\beta_1 = 0.9$ exponential decay for first moment
- $\beta_2 = 0.99$ exponential decay for second moment

intermediate variables:
- $M_0 = 0$ first moment vector
- $V_0 = 0$ second moment vector
- $t = 0$ current step of gd

now first we update our momentums:

$$
M_{t+1} = \beta_1 M_t + (1-\beta_1)\nabla L(\theta_t)
$$

$$
V_{t+1} = \beta_2 V_t + (1-\beta_2)\nabla L(\theta_t)^2
$$

Then in this step is unique, we now correct for bias:

$$
\^{M}_{t+1} = \frac{M_{t+1}}{1-\beta_1^t}
$$

$$
\^{V}_{t+1} = \frac{V_{t+1}}{1-\beta_2^t}
$$

lets see what this does to $\^{M}$ and at the extrema:

$$
\lim_{t \to \infty} \quad \^{M}_{t+1} = {M_{t+1}} 
$$

 
$$

\lim_{t \to 1} \quad \^{M}_{1} = \frac{{\beta_1 M_0 + (1-\beta_1)\nabla L}}{1-\beta_1} = \frac{\beta_1 M_0}{1-\beta_1} + \nabla L = \nabla L
$$

so what this does is when initalizing $\beta_1 = 0,9$ or $0.99$ the momentum terms would supress the current gradient term and there would be a bias towards the first initalization. which it corrects for, but when the step size increases the momentum terms take over again.

lastly we update our parameters:

$$
    \theta_{t+1} = \theta_t - \alpha \frac{\^{M}_{t+1}}{\sqrt{\^{V}_{t+1} + \epsilon}}

$$

all the steps combined + small optimization, our shader code could look as follows:

```wgsl 
t = t + 1
let alpha_t = a * sqrt((1.0 - pow(b2, t))) / (1.0-pow(b1, t));

let g = gradLoss;
m = b1 * m + (1.0 - b1) * g;     // raw first momentum estimate
v = b2 * v + (1.0 - b2) * g * g; // raw second momentum estimate
param = param - alpha_t * m / (sqrt(v) + eps); // step
```

where we combine the $\^{M}$ and $\^{V}$ computation directly by creating $\alpha_t$:

$$
\alpha_t = \alpha \frac{\sqrt{1-\beta_2^t}}{1-\beta_1^t}
$$

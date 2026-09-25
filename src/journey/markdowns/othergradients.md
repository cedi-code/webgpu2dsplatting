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

For ease of notation I will use:
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

We also need to adapt the other gradients $\frac{\partial}{\partial \mu}g$ and $\frac{\partial}{\partial s}g$. Using intermediate variable $v = R^T (x-\mu)$ is quite straight forward:


<details>
  <summary>Show updated derrivations</summary>

$$
\frac{\partial}{\partial \mu}g(x) = g(x) R\begin{bmatrix} 
\sigma(s_x)^2 & 0 \\
0 & \sigma(s_y)^2
\end{bmatrix} v
$$

and

$$
\frac{\partial}{\partial s}g(x) = -g(x) v \circ v \circ  \begin{bmatrix}
\sigma(s_x)^2  \\
\sigma(s_y)^2 
\end{bmatrix} 
$$

</details>


<details>
  <summary>Show updated shader</summary>

```wgsl

struct GaussParams {
    pos : vec2f,
    scale : vec2f,
    rot : f32, // [!code ++]
};

fn gGrad(x :vec2f, p : Params) -> GaussParams {

    let gauss = g(x,p);

    let d = (x - p.pos); // [!code ++]
    let R = rotMat(p.rot); // [!code ++]
    let v = transpose(R)*dist; // [!code ++]
    let sInv = mat2x2f(exp(2.0 * p.scale).x, 0.0, 0.0, exp(2.0 * p.scale).y);  // [!code ++]

    let gradQ = gauss * p.scale * p.scale * (x - p.pos); // [!code --]
    let gradQ = gauss * R * sInv * v; // [!code ++]

    let gradS = -1.0 * gauss * (x - p.pos) * (x - p.pos) * exp(2.0*p.scale) // [!code --]
    let gradS = -1.0 * gauss * v * v * exp(2.0*p.scale) // [!code ++]

    let gradR = -1.0 * gauss * v.x * v.y * (exp(2.0 * p.scale.x) - exp(2.0 * p.scale.y)); // [!code ++]


    return GaussParams(gradQ, gradS, gradR);   
}
```

</details>



To make sure we made no mistakes during our derrivations and implementations of the gradients in our shader, we could compare our analyitcal solution we found with a numerical approximation. for that we can use the definition of a definition of the differentation and take a small enough $h$ for step:

$$
\frac{\partial}{\partial s}g(x;r) \approx \frac{g(x; r + h) - g(x;r)}{h}
$$

we can calulate this value for some input $x_i \in \R^2$ and compare it with our implementation. Well its actually a bit more difficult than that, so if you are intressted in testing our gradients, one can read the article: [Gradient Checking](https://cedi-code.github.io/webgpu2dsplatting/test.html) ! where I check the gradients we just implemented and compare them to the numerical approximation.

### Color and Alpha Gradients

Previously our color used to just be the luminance (scalar value), which in our case is just $g(x)$. but for color this is a 3d vector. A Red, green and blue channel. Each gaussian has itself assigned a unique color and a alpha value since we are not doing spherical harmonics this makes it quite ez. I will define our output image as:

$$
\^I(x) = \alpha S(x) = \alpha C g(x) 
$$

$$
C = \begin{bmatrix}
c_r \\
c_g \\
c_b \\
\end{bmatrix} \quad \quad S(x) = Cg(x)
$$

$g(x)$ just says how intensive that color is. Looking at our loss function and assuming just a single gaussian we have something like this:

$$
L(\mu, s, r, c) = \frac{1}{(N+1)^2}\sum_{i=0}^{N}\sum_{j=0}^{N}(\^{I}(x)- I^*)^2
$$

so the derrivative $\nabla_C L$ is quite trivial since we have a constant:

$$
\nabla_c L= \frac{1}{(N+1)^2}\sum_{i=0}^{N}\sum_{j=0}^{N}(\^{I}(x)- I^*) \circ \begin{bmatrix}
1 \\
1 \\
1 \\
\end{bmatrix}\alpha g(x_{ij})
$$

For alpha it would be similarly easy since the alpha is just a scalar value. 

$$
\nabla_{\alpha} L= \frac{1}{(N+1)^2}\sum_{i=0}^{N}\sum_{j=0}^{N}(\^{I}(x)- I^*) ^TS(x_{ij})
$$

it gets a little more compliated when we introduce multiple gaussian splats to the mix, since then we have to start *alpha blending* where order matters!

Our output image is defined as follows with *alpha blending*:

$$

\^I_G(x) = \alpha_k S_G(x) + (1-\alpha_G)\^I_{G-1} \\

$$

where $G$ = #splats and $S_k$ is the k-th splat where k=G is most frontal splat and k=1 is the splat all the way in the back.
expanding the term one can maybe catch a pattern...

$$
\begin{align*}
\^I_G(x) &= \alpha_G S_G(x) + (1-\alpha_G)[\alpha_{G-1} S_{G-1}(x) + (1-\alpha_{G-1})\^I_{G-2} ]\\
 &= \alpha_k S_G(x) + \alpha_{G-1} S_{G-1}(x) (1-\alpha_{G}) + \^I_{G-2} (1-\alpha_{G})(1-\alpha_{G-1}) \\

\end{align*}
$$

generalizes to:

$$
\^I_G(x) = \sum^{G}_{k=1} \alpha_kS_k(x) \prod^{G}_{l=k+1}(1-\alpha_l)
$$

cool, for color derrivative not a lot changes:



<details>
  <summary>Show derriatives</summary>

Lets say for a specific splat $k$ that has color $C_k$ the gradient is:

$$
\nabla_{C_k} \^I(x) = \alpha_k g_k(x)  \prod^{G}_{l=k+1}(1-\alpha_l)  \begin{bmatrix}
1 \\
1 \\
1 \\
\end{bmatrix}
$$

and for splat $k$ , the alpha $\alpha_k$ we get:

$$
\begin{align*}
\nabla_{\alpha_k} \^I(x) &=  S_k(x) [\prod^{G}_{l=k+1}(1-\alpha_l)] - \sum^{k-1}_{j=1}\alpha_j S_j(x) \frac{1}{1-\alpha_k}[\prod^{G}_{l=j+1}(1-\alpha_l)]  \\
&= S_k(x) [\prod^{G}_{l=k+1}(1-\alpha_l)] - \^I_{k-1}(x)[\prod^{G}_{l=k+1}(1-\alpha_l)] \\
&= (S_k(x) - \^I_{k-1})\prod^{G}_{l=k+1}(1-\alpha_l)

\end{align*}
$$

</details>

For the implementation we need to calculate two variables before we can evaluate the gradients for each gaussian $\alpha_k S_k$.

- accumulated $(1-\alpha)$ values

$$
(1-a)_{k} = \prod^{G}_{l=k+1}(1-\alpha_l)
$$

- the intermediate images (result after renderin up to k-gaussians in order!)

$$
\^I_{k} = \sum^{k}_{j=1} \alpha_j S_j(x) (1-a)_k
$$



Naive way to implement this is to calculate $\prod^{G}_{l=k+1}(1-\alpha_l)$ for each $\^I_k$  like this (pseudo code):

```wgsl
// == looping throu image coords ===
for (var i = 0; i < width; i++) {
  for (var j = 0; j < height; j++) {
    ...
    var I_k = vec3f(0.0); // black background
    for(var k = 0; k < nGauss; k++) {
      // === setup ===
      let gauss : f32   = g(x,p[k]);
      let color : vec3f = p[k].color;

      let S_k : vec3f   = color * gauss;
      let alpha_k : f32 = p[k].alpha;

      // === prod (1-a) ===
      var alphaMinus1 = 1.0;
      for(var l = k+1; l < nGauss; l++) {
        alphaMinus1 *= (1.0 - p[l].alpha);
      }

      // === img result at gaussian k ===
      I_k += alpha_k * S_k * alphaMinus1;

      // calculate gradients
      GradGauss(...);
    }

// looping again throu all gaussians and multiplying gradients by (I* - I_k), L2 loss term.
...
  }
}
```

this runtime is $O(n^2)$ where $n$ = #gaussians, ofc we can do better by first looping throu all  $(1-\alpha_k)$ and store the intermediate results in a array only needing to loop throu twice all the gaussians. (but making the storage goes from $O(1)$ to $O(n)$) which is a trade offer we are taking for now :

```wgsl
...
      // === prod (1-a) === // [!code ++]
    var alphaMinus1 = array<f32, nGauss>(); // [!code ++]
    alphaMinus1[nGauss-1] = 1.0; // [!code ++]
    for(var l = nGauss-1; l >= 1; l--) { // [!code ++]
      alphaMinus1[l-1] = (1.0 - p[l].alpha) * alphaMinus1[l]; // [!code ++]
    } // [!code ++]

    var I_k = vec3f(0.0); // black background
    for(var k = 0; k < nGauss; k++) {
      // === setup ===
      let gauss : f32   = g(x,p[k]);
      let color : vec3f = p[k].color;

      let S_k : vec3f   = color * gauss;
      let alpha_k : f32 = p[k].alpha;

      // === prod (1-a) === // [!code --]
      var alphaMinus1 = 1.0; // [!code --]
      for(var l = k+1; l < nGauss; l++) { // [!code --]
        alphaMinus1 *= (1.0 - p[l].alpha); // [!code --]
      } // [!code --]

      // === img result at gaussian k ===
      I_k += alpha_k * S_k * alphaMinus1; // [!code --]
      I_k += alpha_k * S_k * alphaMinus1[k]; // [!code ++]

      // calculate gradients
      GradGauss(...);
    }
    ...
``` 

##### *next chapter there will be a even cleaner solution! runtime $O(n)$ and storage $O(1)$*

// todo explain activation function for color!

// todo show gradient in code for color and alpha
<details>
  <summary>Show shader code gradient calculation</summary>

    ...
    let gradColor += alpha * gauss * alphaMinus1_k * vec3f(1.0) * dSigmoidColor;

    // alpha gradient  
    let gradAlpha =  (gauss - I_k) * alphaMinus1_k * dSigmoidAlpha;

</details>

// todo have code playground be displayed after this (such that one can initalize color and alpha and pos and rot and scale? (thats a lot...))

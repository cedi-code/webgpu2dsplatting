## Transforming Gaussians
idea:

$$\mathcal{N}(\mu,\Sigma) \implies \mathcal{N}(0, I)$$

so make sure the vertex shader passes / transforms it such that only fragment shader simplifies:

$$
G'(x) = e^{-\frac{1}{2}(x-\mu)^T\Sigma^{-1}(x-\mu)} \implies G'(x) = e^{-\frac{1}{2}x^Tx}
$$

this has the huge benefit of the fragment shader not needing to load any splat information! which enables instancing for us!


example of + - code
```wgsl
-   let d = p.xy - mean; // [!code --]   
+   let d = (p.xy / canvas.xy) - mean; // [!code ++]  
```
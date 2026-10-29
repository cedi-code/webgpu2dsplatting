

The paper proposes a solution for this called "Densification". It looks as follows:

measure a splats $i$ positinal gradient $\nabla_{\mu}g_i$.
* if $\nabla_{\mu}g_i$ is large after steps densify the splat!
    * if $\nabla_{\mu}g_i$ and $\nabla_{s}g_i$ (scale) are high

        $\implies$ split splat-$i$ (Over-reconstruction)
    * if $\nabla_{\mu}g_i$ large only 
    
         $\implies$ duplicate splat-$i$ (Under-reconstruction)

![densification figure](https://onchaings.xyz/assets/gaussian-densification-scheme-U_PUj6Nx.png)
##### figure from paper

to implement this we need to first detect when a splat needs refining (`splatRefine.wgsl`):

```wgsl
@group(0) @binding(5) var<storage, read_write> gradients: array<Grad, MAX_SLOTS>; 
...

@compute @workgroup_size(GROUPSIZE, 1, 1)
fn cs(...) {

    ...
    let posGradFlag = 
    any(abs(gradients[i].pos) >= vec2f(uniforms.posGradThresh));

    let scaleGradFlag = 
    any(abs(gradients[i].scale) >= vec2f(uniforms.scaleGradThresh));

    // refine and create a new splat!

    }
```

since now instead of removing we will add a new splat we just write this in our offset calculation:

```wgsl
    let secondSplat = select(0u, 1u, posGradFlag);  // [!code ++]
    sums[i] = select(1u, 0u, offScreen); // [!code --]
    sums[i] = select(1u + secondSplat, 0u, offScreen); // [!code ++]
```

now to create the new splat for Under-reconstruction we do nothing special, we just create the exact same splat at the same location.

```wgsl
    ...
    let splatCopy = splats[i];
    ...
    workgroupBarrier(); 
    ...
    let newI = sums[i] - 1u;

    if(!offScreen) {
        // refine
        if(posGradFlag) {

            // copy the current splat
            let splatClone = Params(splatCopy);

            if(scaleGradFlag) { // over-reconstruction
                ...
            }
            else { // under-reconstruction
                // do nothing :)
            }

            // store cloned splat
            splats[newI + 1u] = splatClone;
        }
        
        // store old splat at new position
        splats[newI] = splatCopy;

    }
``` 

doing nothing works, but having two splats with the same alpha overlapp does the following, assume that we duplicate the front most splat $g_l$ which is a bit transparent, meaning $\alpha_l < 1$, we get:

$$
\begin{align*}
\^I &=  ... + a_l C_l \\
 & \implies \\
 \^I' &= ... + a_l C_l (1-a_l) + a_l C_l \\ 
  &=  ... + a_l C_l(2-a_l) \\
 
\end{align*}
$$

the $\^I'$ will appear brighter than $\^I$ since:
$$
\begin{equation}
a_l C_l(2-a_l) > a_l C_l
\end{equation}
$$

so instead we can correct a bit for that sudden brighness spike by:

$$
a_l'  = 1 - \sqrt{1 - a_l}
$$

pluggin this corrected alpha term on the LHS of eq. (1) gives us:

$$
\begin{align*}
a_l' C_l(2-a_l') &= (1 - \sqrt{1 - a_l}) C_l(2-(1 - \sqrt{1 - a_l})) \\
&=  C_l (1 - \sqrt{1 - a_l})(1 + \sqrt{1 - a_l})) \\
&= C_l (1^2 - (\sqrt{1 - a_l})^2) \\
&= a_lC_l
\end{align*}
$$

which keeps the brightness intact! in code we could write it as such:

```wgsl
    else { // under-reconstruction
        let alphaCorr = 1.0 - sqrt(1.0 - sigmoid(splatCopy.alpha));
        splatClone.alpha = alphaCorr;
        splatCopy.alpha = alphaCorr;
    }
```

now for over-reconstruction the paper suggest to divide the size by a constant $1.6$ and important sample the new positions based on the position gradient. They dont go into much detail, so in my method I instead split i the splat half in the axis of the minor principal component.

so let $\lambda_1 \geq \lambda_2$ / $v_1, v_2$ be the eigen values/ eigen vectors of $g_l$ that we are about to split.

I use the following heuristic then to determine the new position of both $g_l'$ and $g_l''$.

$$
\mu' = \mu \pm \sqrt{\lambda_1}v_1
$$

and for scale like in the paper (reminder that scale has activation function $e^{-s}$):

$$
e^{-s} / 1.6 \implies s' = s + log(1.6)
$$


```wgsl
    if(scaleGradFlag) { // over-reconstruction
        let eigenVec : vec4f = getEigenVecCorr(splatCopy); // (v1, v2)
        let eigenVal : vec2f = getEigenVal(splatCopy); // (lambda1, lambda2)

        // new position
        let oldPos = splatCopy.pos;
        splatClone.pos = oldPos - sqrt(eigenVal.x) * eigenVec.xy;
        splatCopy.pos = oldPos - sqrt(eigenVal.x) * eigenVec.xy;

        // new scale
        let oldScale = splatCopy.scale;
        splatClone.scale = oldScale + log(1.6);
        splatCopy.scale = oldScale + log(1.6);

    }
```


### Alpha degradation (floaters)

One funny thing that i noticed was that very small splats that where created from the Densification where so small that they bearly contributed to the loss, so they would just stay.

![aaa](assets/documentationScreenshots/floaters.png)

The paper actually discusses this issue and calls them floaters, which they remove after 3000-steps, setting the alpha value very low and hoping for GD to increase the alpha for relevant features again.

While my solution is to just every 30-50 steps have a pass that  decreasing the alpha value, so floaters slowley fade away in the background. 

```wgsl
...
fn cs() {
    let splat_id = global_invocation_id.x;
    splats[splat_id].alpha *= uniforms.opacityReset;
}
``` 

ok now trying again with refine steps. we can approximate the image!
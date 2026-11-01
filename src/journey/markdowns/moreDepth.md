
## Big Splats problem
This is a more fundamental problem that originates since the start of the project.

While testing my splater method for multiple images i kept noticing that very large splats would try to cover very large areas...

##### *jeff.jpg after 100 iteration, initalized with small blobs*
<img height="300" src="../../../assets/documentationScreenshots/jeff100IterationBeforeBlurry.png" />
<img height="300" src="../../../assets/documentationScreenshots/jeff100IterationBeforeBlobs.png" />


##### *jeff.jpg after 1k iteration, mostly big blobs making the image*
<img height="300" src="../../../assets/documentationScreenshots/jeff1kIterationBeforeBlurry.png" />
<img height="300" src="../../../assets/documentationScreenshots/jeff1kBeforeBlobs.png" />

### Remove them?
my first intuition was
<div align="center" style="font-size: 2rem; font-family: 'Brush Script MT', 'Segoe Script', cursive">
  "Big blobs are bad! Let me try to remove them!"
</div>
<div align="right" style="font-size: rem; font-family: 'Brush Script MT', 'Segoe Script', cursive">
  ~ my dumb ass
</div>

Since more smaller splat could add more detail and therfore make the approximation of the image better! 

So i tried multiple methods trying to get rid of big splats and have more smaller splats that could capture more detail.

I tried:
- more refinment steps
    - => didnt work, splats cannot converge, unstable!
- densify or remove splats with large eigen values
    - => did not work, other splats just start replacing the missing large splats

But then this last approach worked
- add additional loss term on scale $(s_x, s_y)$ that increases the gradient for larger splats!

which kept the splats at a certain size. But instead of seeing a lot of small splats trying to cover the whole image, i just had a worse approximation!

After all these approaches and finally beeing able to limit the splats size, i realized that the big splats where actually not a problem! If one thinks of a image of a collection of frequencieis, the big splats are just approximating the low frequencies (overall image) and the smaller splats try to approximate the higher frequencies! 

So removing the low frequencies (big splats) is just not going to work to have a better approximation. But at the same time a lot of big splats are covering smaller splats, and due to alpha blending, they loose on relevance. 

So what i actually need is:


<div align="center" style="font-size: 2rem; font-family: 'Brush Script MT', 'Segoe Script', cursive">
  "Big blobs are good! Let me try put them in the back!"
</div>
<div align="right" style="font-size: rem; font-family: 'Brush Script MT', 'Segoe Script', cursive">
  ~ my less dumb ass
</div>

this makes intuativly also sense to have big large details in the back, and high frequency things at the front. But all of this means we somehow need to learn the "depth" of the splats, which for now was just a fixed order!

### Depth Gradient

Now how do we tell our optimizer to somehow put larger splats in the back and smaller splats to the front? We need our splats to optimize also over the depth $\mu_z$

$$
\mu_{old} = \begin{bmatrix}
\mu_x \\
\mu_y
\end{bmatrix} \implies \mu_{new} = \begin{bmatrix}
\mu_x \\
\mu_y \\
\mu_z 
\end{bmatrix}
$$ 

I first tought of doing a perspective transform and have the splats always facing the camera. This would work if i also assign each splat a size in the z-dimension (depth). But actually we can just assume our splats are flat disks and still have them move in depth by changing our current alpha blending!

Our current alpha blending looks as follows:

$$
\^I_G = \sum^{G}_{k=1} a_kC_k \prod^{G}_{l=k+1}(1-a_l)
$$
with $a_k = \alpha_k g_k(x)$ and $C_k$ beeing the color value.

to make the next point a bit more clear, lets make it much simpler and assume $G=2$, we get:

$$
\^I_G = a_1C_1 (1-a_2) + a_2C_2
$$

important here is to say hat splat 1 lies behind splat 2: $z_1 >  z_2$. One can also express this relation using there depth values, which could look as follows:

$$
\^I_G = a_1C_1 (1- \mathbb{I}_{z_1 > z_2} a_2) + a_2C_2 (1- \mathbb{I}_{z_2 > z_1}a_1)
$$

where:
$$\mathbb{I}_{x > y} = \begin{cases} 1 & \text{if } x > y \\ 0 & \text{otherwise} \end{cases}$$

<p><img width="300" height="250" style="background-color: white" src="https://thumb.wikimedia.org/wikipedia/commons/thumb/d/d9/Dirac_distribution_CDF.svg/500px-Dirac_distribution_CDF.svg.png?utm_source=en.wikipedia.org&amp;utm_campaign=imageinfo&amp;utm_content=thumbnail" alt="Dirac distribution CDF.svg" height="375" width="500"></p>

this alphablending is not continious in terms of $z$ which determinds the order of the splats. So its going to be hard to tell our optimizer where to move the splat.

Instead we could describe our alpha blending as follows:

$$
\^I_G = a_1C_1 (1- \sigma(z_1 - z_2) a_2) + a_2C_2 (1- \sigma(z_2 - z_1) a_1)
$$

where $\sigma$ is the sigmoid activation function

$$
\sigma(\alpha) = \frac{1}{1+e^{-x}}
$$

<img width="300" height="250" src="https://raw.githubusercontent.com/Codecademy/docs/main/media/sigmoid-function.png">

to have a bit more controll over when the treshhold is reached we can add a normalization term $\tau$ that represents how "thicc"/"thin" our splat is

$$
\^I_G = a_1C_1 (1- \sigma(\frac{z_1 - z_2}{\tau}) a_2) + a_2C_2 (1- \sigma(\frac{z_2 - z_1}{\tau}) a_1)
$$

- large $\tau \implies$ "thicc", splats touch each other from far away (z-axis)


- small $\tau \implies$ "thin", splats need to be very close to change order!

generalizing this to multiple splats we get the following:

$$
\^I_G = \sum^{G}_{k=1} a_kC_k \prod^{G}_{l=k+1}(1- \sigma(\frac{z_k - z_l}{\tau})a_l)
$$

we might want to also make $\tau$ e property per splat to simulate as if they where 3d blobs, we can give smaller splats smaller $\tau$ and larger splat larger $\tau$.

$$
\tau = \beta s_xs_y 
$$

where $s$ is the scale of the splat and $\beta$ a hyperparameter we can controll.

so in the end we get:

$$
\^I_G = \sum^{G}_{k=1} a_kC_k \prod^{G}_{l=k+1}(1- \sigma(\frac{z_k - z_l}{\sqrt{\tau_k + \tau_l}})a_l)
$$

To now update our depth $z$ we can now use the gradient of $\nabla_{z} I_G$ of the alpha blending method we just described.

Note that this inherantly does not push back larger splats and it also does not respect the forward alpha blending anymore, but it gives us a depth value to optimize over!

to push back the bigger splats we could define a additional loss term which punishes larger splats that are infront of smaller ones!

$$
L = L_{\text{image}} + \lambda L_{\text{depth}}
$$

$$
L_{\text{depth}} = \begin{cases} z_j - z_i & \text{if } s_i > s_j \land z_j - z_i > 0 \land \mu_i \approx \mu_j \\ 0 & \text{otherwise} \end{cases}
$$



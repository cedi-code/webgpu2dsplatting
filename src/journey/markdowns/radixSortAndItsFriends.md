## Problem
current shader passes:

``` 
forwardPass()
backwardsPass()
reducePass()
adamPass()
```

in our backwards pass we calculate the gradient for a 16x16 patch of the image. 

```
@workgroup(16,16)
gradient : atomic<gradient>

foreach splat
    gradient += grad(splat);
```

which will get expensive if we have more splats!
Splats will start approximating pixels, not contributing a lot to other pixels on the other side of the image.

So one idea is to group all the splats that are relevant for that 16x16 patch which i will call Box.

We have $M$ splats and $N^2$ boxes. Our boxes are arranged in a $N \times N$ grid. let $(j, k) \in \{ 1, \dots, N \} \times \{ 1, \dots, N \}$ be the $j,k$-th bin and $i \in \{ 1, \dots, M \}$ be the index of $i$-splat.

so we want to get every splat that is relevant for a box.

$$
box[j,k] = \{ i \in \{1,\dots, M\} \quad | g_i(p_b) > \epsilon\}
$$
where $p_b$ is the position at the box (j,k) and $\epsilon$ is a really small number.

but getting that array $box[j,k] $ of splat indicies is actually quite tricky!

## Solution

i frist tried to make a range approach [(see problem description)](binsProblem.md) but after arguing with a LLM chatbot for a while they coninced me to stick with the approach of the paper and doing a radix sort over all the boxes.

this adds a couple of new render passes that need to be implemented:

```
forwardPass()
setupBoxArray() // [!code ++]
radixSort() //  // [!code ++]
backwardsPass()
reducePass()
adamPass()
```

the splats are then stored into a array which is sorted by boxes, so the boxes in the backwardsPass only need to read from there start index and itterate over all the relevant splats (yes splats are dublicated in that array).

### SetupBox array

We basically want get this:
$$
\{ (i, (k,j)) \quad | g_i(p_{k,j}) > \epsilon  \}
$$

and we want it to be sorted by $(k,j)$
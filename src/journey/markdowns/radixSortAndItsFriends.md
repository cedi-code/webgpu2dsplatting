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
radixSort()  // [!code ++]
backwardsPass()
reducePass()
adamPass()
```

the splats are then stored into a array which is sorted by boxes, so the boxes in the backwardsPass only need to read from there start index and itterate over all the relevant splats (yes splats are dublicated in that array).

### SetupBox array

The goal of this part is to have data arranged as follows:
$$
\{ ((k,j), i) \quad | \quad g_i(p_{k,j}) > \epsilon \quad  \}
$$

for example:
```text
[(chunk2, splat2), (chunk3, splat2), (chunk1, splat1), (chunk2, splat1)]
```
the data comes in pairs of splats $i$ and for simplifaction we just have a single index for chunk $j$

in the example, `splat1` covers 2 chunks `[chunk1, chunk2]` and `splat2` covers also 2 chunks `[chunk2, chunk3]` 


we run a workgroup for each splat
and have the following struct:

```wgsl
struct radixMemory {
    data : array<u32, MAX_SLOTS>,
    currI : atomic<u32>,
};

@group(0) @binding(0) var<storage, read_write> splats: array<Params>;

@workgroup(NUM_SPLATS,1,1)

fn registerSplats(
        @builtin(global_invocation_id) global_invocation_id : vec3u,
) {
    let i = global_invocation_id.x;

    let splat = splats[i];

    // figure out how many chunks the splat covers

    // grab the current index and increase it by # chunks

    // write down the chunk-j,splat-i pairs.

}
```

### SetupBox better

we can do better...
our splats are already orderd by depth, so leaving it to a atomic currI we have to resort for splats $i$!

instead we can write them in order from the beginning

```text
[(?, splat1), (?, splat1), (?, splat2), (?, splat2), ...]
```

To achive this, we have need to know how many chunks a splat even covers beforehand, so we do these 3 steps:

1. Count: Calculate the number of boxes each splat covers

2. Scan: Run a prefix sum over those counts. This gives each splat the exact  starting offset for the radix array of key-value pairs

3. Write: write down the key value pairs starting from the prefix sum offset (j,i).

written in code:

```wgsl
struct radixMemory {
    data : array<u32, MAX_SLOTS>,
};
@group(0) @binding(0) var<storage, read_write> splats: array<Params>;

var<workgroup> offsets: array<u32>; // [!code++]

@workgroup(256,1,1)
fn registerSplats(
        @builtin(global_invocation_id) global_invocation_id : vec3u,
) {
    let i = global_invocation_id.x;

    // 1. get the left, right, top, bottom most points of gaussian and calculate the number of chunks it would cover.
    offsets[i] = allocationSize(splats[i]);

    workgroupBarrier();

    // 2. calculate where in 'splats' the current splat would start writing its pairs
    parallelPrefixSum();

    workgroupBarrier();

    var currOff = offsets[i]; // corrected now

    // 3. write down the key value pairs
    for(var j = minChunk; j < maxChunk; j++) {

        // box id
        radixMem.data[currOff] = j;

        // splat id
        radixMem.data[currOff + 1u] = i;

        currOff += 2;
    }

}
```

with this, we already have it sorted by splat id $i$!

Now for the next step, since we do backprop per chunk and not per splat, we need to sort it by chunk id $j$.

#### Parallel Prefix sum

wip: i need to explain parallel prefix sum

### Radix Sort

wip: i need to explain parallel block prefix sum

### Backwards pass update

Since backwards pass is groupped into chunks, the chunks finally dont need to loop over every splat but only over the splats that overlapp them,
which we can easly read now given our sorted array that is orderd by chunks!

```
data = [(chunk1, splat1), (chunk1, splat2), ..., (chunk2, splat1), ..., (chunkN, splat1), ..]

offsets = { chunk i starts at index j }
```
we then change our old code:

```wgsl
@group(0) @binding(5) var<storage, read_write> gradientChunks : array<array<Grad, NUM_SPLATS>>; // [!code--]
@group(0) @binding(5) var<storage, read_write> gradientChunks : array<array<Grad, MAX_SPLATS_CHUNK>>; // [!code++]

...
let offset = radixMem.offsets[chunk_id]; // [!code++]
let numSplats = offset - radixMem.offsets[chunk_id - 1] // [!code++] 
for(var i = numSplats; i > 0u; i--) {
    let currOff = (offset + i - 1u);  // [!code++]
    let splatId = radixMem.data2[2u * off + 1u];  // [!code++]
    GradLoss(splats[splatId], i-1u, uv, colorGrad, &background, &oneMinusAlpha);  
}
``` 

but one issue now is how do we sum back our gradient chunks?

for example lets assume we have grad chunk A and grad chunk B,

grad chunk A may have the following:

```
(gradient_splat1, gradient_splat3)
```

and grad chunk B:

```
(gradient_splat2, gradient_splat3)
```

one approach could be using a global gradient memory buffer where each thread adds the gradient into, this would need to be atomic and when having a lot of tiles and one big splat, would lead to a big overhead.

another idea is to perform another radix sort on top of the gradients, this time sorting by splats, and then summing up in parallel all the chunks

well actually we already have this order from the Radix setup! there the order is already in (splat, chunk), so we just need to store all the gradients in that order and then they can easly be summed up in parallel! no need to resort them or do atomic operations! 

```text
[(chunk1, splat1, transpose1), (chunk1, splat2, transpose2), (chunk2, splat2, transpose3), (chunk2, splat3, transpose4), ...]
```

the transpose stores the index of the (chunk, splat) pair when it used to be sorted by splat id and not chunk id. 
One can think of the sorting from splats, to sorting to chunks as a transposing of the adjacency matrix! this matrix is very sparse so thats why we store the old index seperatly.






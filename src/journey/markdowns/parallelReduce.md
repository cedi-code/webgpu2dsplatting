## Parallelize Backwardspass
<div align="center" style="font-size: 2rem; font-family: 'Brush Script MT', 'Segoe Script', cursive">
  We need to parallelize!
</div>
<div align="center">
  <img src="assets/shadowClones.gif" width="300">
</div>

current shader passes do not utilize one of the most usefull features of GPU programming...
``` 
forwardPass() // per instance
backwardsAndAdamPass() // one dispatch!
```

wip

now we have 
``` 
forwardPass() // per instance
backwardsPass() // per tile
reducePass() // per splat
adamPass() // per splat
```

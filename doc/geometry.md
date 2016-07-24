# geometry

## neon winograd geometries

```
calcU:
   in:                 [ c][kh][kw][ k]
  out:             [gk][ c][xi][nu][k%]
calcV:
   in:                 [ c][ h][ w][ n]
  out:     [th][tw][gn][ c][xi][nu][n%]
```

## neonCl winograd geometries, current

```
calcU
   in:                 [ c][kh][kw][ k]
 loop:                         [xi][nu]
 grid:                         [gk][ c]
block:                             [k%]
  out:             [xi][nu][gk][ c][k%]
----------------------------------------
calcV
   in:                 [ c][ h][ w][ n]
 loop:                         [xi][nu]
 grid:                 [gn][th |tw][ c]
block:                             [n%]
  out:     [xi][nu][gn][th][tw][ c][n%]
----------------------------------------
calcM
  grid:                         [th |tw]

 Uloop:             [gn][gk][xi][nu][gc]
Ublock:                         [c%][k%]
----------------------------------------
 Vloop:             [gn][gk][xi][nu][gc]
Vblock:                         [c%][n%]

  loop:             [gn][gk][xi][nu][ c]
 block:                         [ c][n%]

   out: [gn][n%][gk][k%][th][tw][xi][nu]
----------------------------------------
calcO:
```

## neonCl winograd properties, new2


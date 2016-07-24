# geometry

## neon winograd geometries

```
calcU:
   in:                  [ c][kh][kw][ k]
  out:              [gk][ c][xi][nu][k%]
calcV:
   in:                  [ c][ h][ w][ n]
  out:      [th][tw][gn][ c][xi][nu][n%]
```

## neonCl winograd geometries, current

```
calcU
   in:                  [ c][kh][kw][ k]
 loop:                          [xi][nu]
 grid:                          [gk][ c]
block:                              [k%]
 gmem:              [xi][nu][gk][ c][k%]
========================================
calcV
   in:                  [ c][ h][ w][ n]
 loop:                          [xi][nu]
 grid:                  [gn][th |tw][ c]
block:                              [n%]
 gmem:      [xi][nu][gn][th][tw][ c][n%]
========================================
calcM
  grid:                         [th |tw]
----------------------------------------
  in U:
  loop:             [gn][gk][xi][nu][gc]
 block:                         [c%][k%]
  gmem:             [xi][nu][gk][ c][k%]
----------------------------------------
  in V
  loop:             [gn][gk][xi][nu][gc]
 block:                         [c%][n%]
  gmem:     [xi][nu][gn][th][tw][ c][n%]
----------------------------------------
   out
  loop:             [gn][gk][xi][nu][ c]
 block:                         [ c][n%]
  gmem: [gn][n%][gk][k%][th][tw][xi][nu]
========================================
calcO:
```

## neonCl winograd properties, new2

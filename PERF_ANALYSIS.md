# Why the performance numbers are what they are

DE1-SoC, 16×16 int8 weight-stationary systolic array at 90 MHz, driven from the
HPS over the h2f bridge with two DMA channels.

---

## The one conclusion

**Performance is a bound by software tiling scheme.**

It's observed that computing a 256×256 matmul with 128×128 output tiles gives the same throughput
as computing a 128×128 matmul. But if changed to a separate tiling scheme where W is tiled by 256x64 strips and X is tiled by 64x256 strips, the performance can be doubled. The key observation is that the second tiling scheme halved the number of matmuls and also achieved about twice the throughput as shown in the measured performance. 



---

## Notation

Every matmul the hardware runs is described by three numbers:

```
tk   output rows this matmul produces      (WEIGHT_ROWS)
tn   contraction length this matmul covers (WEIGHT_COLS)
tm   output columns this matmul produces   (INPUT_COLS)

weight buffer <- a tk x tn slab of W
input buffer  <- a tn x tm slab of X
output canvas <- a tk x tm patch of the result
```

A full matmul is covered by repeating this over output rows, output columns, and
contraction chunks. Below, "a tk=64, tn=64, tm=256 scheme" means every matmul
produces a 64-row by 256-column patch of output using a 64x64 block times with 64x256 block.

---

## Three tiling schemes for a 256×256 @ 256×256 matmul

All three do the same 16.78 MMAC. They differ only in how that work is cut up.

**Scheme A — 64×256 output strips.** `tk=64, tn=64, tm=256`.
Four row bands, one column tile (full width), four contraction chunks =
**16 matmuls**. 

**Scheme B — 128×128 output cubes.** `tk=128, tn=128, tm=128`.
A 2×2 grid of output tiles, two contraction chunks = **8 matmuls**.

**Scheme C — 256×256 output square.** `tk=256, tn=64, tm=256`.
One output tile covering the whole result, four contraction chunks =
**4 matmuls**. This is the optimum.

| | A: 64×256 strips | B: 128×128 cubes | C: 256×256 square |
|---|---|---|---|
| tk, tn, tm | 64, 64, 256 | 128, 128, 128 | 256, 64, 256 |
| matmuls | 16 | 8 | 4 |
| weight buffer used | 4096 B (25%) | 16384 B (100%) | 16384 B (100%) |
| input buffer used | 16384 B (100%) | 16384 B (100%) | 16384 B (100%) |
| canvas used | 16384 B (25%) | 16384 B (25%) | 65536 B (100%) |
| density | 51.2 MAC/B | 64.0 MAC/B | **128.0 MAC/B** |
| W sent / X sent | 1× / 4× | 2× / 2× | 1× / 1× |
| operand bytes | 320 KB | 256 KB | **128 KB** |
| compute | 819 µs | 910 µs | 819 µs |
| dma_0 in | 2231 µs | 1785 µs | 892 µs |
| total | 3166 µs | 2811 µs | **1828 µs** |
| throughput | 10.60 GOP/s | 11.94 GOP/s | **18.36 GOP/s** |

Scheme A measured **10.47 GOP/s** on hardware. B and C are projected from the
same model.

### Why scheme C is twice scheme B

C's output tile is 256×256, four times the area of B's 128×128, for the same
32 KB of operands loaded per matmul. Both schemes fill both operand buffers
completely; C simply extracts four times the MACs from each load, which is why
it needs only 4 matmuls instead of 8 and moves 128 KB instead of 256 KB.





---

## The separate 128×128 @ 128×128 matmul

A single matmul at `tk=128, tn=128, tm=128`, because the whole problem fits one
tile — there is nothing to tile.

```
density 64.0 MAC/B, utilization 80.0%
measured: compute 120.10 us, dma_0 in 218.62 us, dma_1 out 29.10 us
          total 367.82 us -> 11.403 GOP/s
```

That tile is the same shape as one of scheme B's eight matmuls, so it has
identical density and identical utilization — and therefore nearly identical
throughput, 11.40 measured here against 11.94 projected for scheme B. 

### Why a 128×128 matmul cannot reach 128 MAC/B

A matmul's own arithmetic intensity is `N/2` MAC/byte — it has N³ of work
against 2N² of operands — and a tile cannot exceed the matrix:

```
matmul size   best tk = tm   density     limited by
128 x 128     128             64.0 MAC/B  the matrix itself
256 x 256     256            128.0 MAC/B  the matrix, and exactly the canvas cap
512 x 512     256            128.0 MAC/B  the 64 KB canvas
```

A 128×128 matmul leaves three quarters of the canvas idle, not because of a
tiling mistake but because **the problem is smaller than the largest tile the
hardware can run**. A 256×256 matmul is the crossover point where the limit
stops being the problem size and becomes the canvas capacity, which is why it is
the sweet spot and why everything above it plateaus at 128 MAC/B.

---

## Peak and achieved throughput

```
peak = 2 ops x 256 PE x 90 MHz = 46.1 GOP/s
```

Best measured compute-only across the sweep was **40.15 GOP/s**, about **87% of
peak**. The shortfall is the head and tail cycles below, plus a small fixed cost
per matmul launch (register writes and the `STATUS` poll, which is an uncached
Device access over the h2f bridge).

---

## The head and tail

The array is 16 PEs deep, and inputs enter **skewed**: element `X[k][t]` has to
arrive at `PE(k,j)` exactly when the partial sum for output column `t` reaches
that PE from `PE(k-1,j)` above. Partial sums travel down one row per cycle, so
input row `k` is delayed by `k` cycles relative to row 0.

That skew has two consequences at the edges of every weight block:

**Head.** At the first cycle only PE row 0 can do useful work, since `X[0]` has
just arrived and `X[1]` is still a cycle away. Row 1 joins on the next cycle,
row 2 after that, and so on. It takes **16 cycles** before all 16 rows are
contributing.

**Tail.** The mirror image. Once the last input element has entered, no new work
starts, but the partial sums still moving down the columns need **16 cycles** to
reach the bottom and be written out. The number of active rows falls away 16,
15, 14 … 1.

So every weight block costs **32 cycles on top of its `tm` cycles of useful
streaming**, which is where

```
utilization = tm / (tm + 32)
```

comes from. A longer `tm` amortizes those 32 cycles over more useful work —
80.0% at tm=128, 88.9% at tm=256 — which is a second reason to prefer wide
output tiles, independent of the bridge.

**This is an area for improvement.** The 32 cycles are only unavoidable because
the control logic waits for the previous block to drain completely before
feeding the next. If the next block's head were allowed to overlap the previous
block's tail, utilization would approach 100% regardless of `tm`. That needs the
output path to hold two blocks in flight rather than draining between them.



---

## Summary

| question | answer |
|---|---|
| Why is a 256×256 matmul faster than a 128×128 one? | It can use a 256×256 output tile, 128 MAC/B against 64. A tile cannot exceed the matrix, so 128×128 is capped at 64. |
| Would a 256×256 matmul tiled with 128×128 output tiles match a 128×128 matmul? | Yes — same tile shape, same density, same utilization, same throughput. |
| Why doesn't a 512×512 matmul beat a 256×256 one? | Both use the same 256×256 output tile. Density plateaus once the canvas, not the matrix, is the limit. |
| Why is the array only ~87% utilized? | 16 cycles of head and 16 of tail per weight block, serialized between blocks. |


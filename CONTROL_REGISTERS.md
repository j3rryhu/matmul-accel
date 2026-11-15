# Control registers

What each control register means, when the hardware reads it, and how the
registers combine to place and format a result. The registers are defined
in [`rdl/ctrl_reg.rdl`](rdl/ctrl_reg.rdl); after changing that file,
regenerate the register file with [`rdl/python_gen.py`](rdl/python_gen.py)
(it patches the generated code - don't run PeakRDL by hand).

Notation used below. One matmul computes

```
Y (K x M) = W (K x N) @ X (N x M)
K = WEIGHT_ROWS   output rows
N = WEIGHT_COLS   contraction length
M = INPUT_COLS    output columns
```

`WEIGHT_ROWS` is the **output-row** count and `WEIGHT_COLS` the
**contraction** length, whatever the names suggest (see the README's
"Data flow and axis convention").

## Access rules

- The registers live at `CTRL_BASE = 0x0000`. Each one occupies its own
  16-byte bus word, with the value in the low bytes, so a write needs
  `avs_byteenable` bits 0..3 set.
- Every register resets to 0.
- Only `STATUS` can be read; the others are write-only.

When the hardware looks at each register matters:

| Read by hardware | Registers | Consequence |
|---|---|---|
| Once, at `CONTROL.matmul_start` | `WEIGHT_ROWS`, `WEIGHT_COLS`, `TILE_CFG`, `TILE_ROWS`, `TILE_COLS`, `ACT_CFG`, `CANVAS_WIDTH` (and `INPUT_COLS` where it stands in for a 0 `CANVAS_WIDTH`/`TILE_COLS`) | Can be reprogrammed for the next matmul while this one runs. The output read layout is latched here too, so a result reads back with the layout it was written with. |
| Continuously during a matmul | `INPUT_COLS`, `INPUT_MAX_ADDR`, `OUTPUT_SCALE` | Must not change until `STATUS.done`. |
| On every host buffer write | `WEIGHT_COLS` (weight writes), `INPUT_MAX_ADDR` (input writes) | Program them before writing the buffers. |

## Summary

| Offset | Register | Field | Bits | Access | Meaning |
|---|---|---|---|---|---|
| `0x00` | `CONTROL` | `matmul_start` | `[0]` | W, pulse | Start a matmul. |
| `0x10` | `WEIGHT_ROWS` | `value` | `[15:0]` | W | K - output rows. |
| `0x20` | `WEIGHT_COLS` | `value` | `[15:0]` | W | N - contraction length; multiple of 16. |
| `0x30` | `INPUT_MAX_ADDR` | `value` | `[9:0]` | W | Reads per input band; set to M. |
| `0x40` | `INPUT_COLS` | `value` | `[15:0]` | W | M - output columns. |
| `0x50` | `STATUS` | `busy` | `[0]` | R | Matmul in progress (live). |
| | | `done` | `[1]` | R, read-to-clear | Matmul finished (sticky). |
| `0x60` | `OUTPUT_SCALE` | `value` | `[15:0]` | W | Q0.16 rescale factor to int8. |
| `0x70` | `TILE_CFG` | `tile_r` | `[14:0]` | W | Tile row index on the tile grid. |
| | | `tile_c` | `[29:15]` | W | Tile column index on the tile grid. |
| | | `use_prev_out` | `[30]` | W | Accumulate onto what the canvas holds. |
| `0x80` | `CANVAS_WIDTH` | `value` | `[15:0]` | W | Whole result's column count; 0 = M. |
| `0x90` | `ACT_CFG` | `bias_en` | `[0]` | W | Add the per-row int32 bias. |
| | | `gelu_en` | `[1]` | W | Apply GELU to the int8 result. |
| | | `prev_int8` | `[2]` | W | `use_prev_out` adds onto the previous int8 result. |
| | | `out_raw` | `[3]` | W | Keep raw 32-bit sums; no int8 result. |
| `0xA0` | `TILE_ROWS` | `value` | `[15:0]` | W | Tile grid height; 0 = K. |
| `0xB0` | `TILE_COLS` | `value` | `[15:0]` | W | Tile grid width; 0 = M. |

## Registers

### `CONTROL` (0x00)

`matmul_start` - write 1 to start a matmul. It is a one-cycle pulse that
clears itself. It snapshots the registers listed above as "read once" and
starts the hardware sequencer; nothing else needs to be written until the
matmul finishes. The weight and input buffers must already hold this
matmul's data.

### `WEIGHT_ROWS` (0x10)

K, the number of output rows this matmul produces (the rows of `W` and of
`Y`). Any value from 1 up; a K that is not a multiple of 16 is handled by
masking the last 16-row block. Also the default tile height when
`TILE_ROWS` is 0.

### `WEIGHT_COLS` (0x20)

N, the contraction length (the columns of `W`, rows of `X`). **Must be a
multiple of 16** - the hardware cannot mask a partial contraction block, so
zero-pad `W` and `X` up to one. Also sets the row pitch of the weight
buffer image, so program it before writing the weights. Longer contractions
than one matmul can hold are split with `TILE_CFG.use_prev_out` (see
below).

### `INPUT_MAX_ADDR` (0x30)

How many elements are read from each input row band; set it equal to M.
10 bits, so M <= 1023. Also sets the row pitch of the input buffer image,
so program it before writing the inputs.

### `INPUT_COLS` (0x40)

M, the number of output columns this matmul produces (the columns of `X`
and of `Y`). Also the default canvas width when `CANVAS_WIDTH` is 0, and
the default tile width when `TILE_COLS` is 0.

### `STATUS` (0x50, read-only)

- `busy` - a live level: 1 while a matmul is being sequenced or its last
  results are still being written. Reading it has no side effect.
- `done` - a sticky event: set once when a matmul has completely finished
  (every block computed and the last result written), cleared by the read
  that returns it.

**Any read of `STATUS` clears `done`**, since both bits share one address.
Read it once per poll and test both bits from that one value. A completion
that lands in the same cycle as a read is not lost - it stays set for the
next read.

### `OUTPUT_SCALE` (0x60)

The factor that turns the 32-bit accumulator into int8:
`y = clamp((sum * OUTPUT_SCALE) >> 16, -128, 127)`, arithmetic shift
(rounds toward minus infinity). Q0.16: the real factor is
`OUTPUT_SCALE / 65536`, range `[0, 1)`.

Under per-tensor symmetric quantization with scales `w_scale` (weights),
`x_scale` (inputs) and `y_scale` (output):

```
OUTPUT_SCALE = round(w_scale * x_scale / y_scale * 65536)     must be < 65536
```

It is applied **once**, after the last contraction block, to the full
32-bit sum (plus bias) - partial sums between blocks are kept at full
precision. With `ACT_CFG.gelu_en`, `y_scale` must be the GELU table's input
scale (1/32 for the example table in `rtl/activation/example/`).

### `TILE_CFG` (0x70), `TILE_ROWS` (0xA0), `TILE_COLS` (0xB0)

Where this matmul's K x M result goes in the output canvas. The canvas is
divided into a grid of `TILE_ROWS x TILE_COLS` tiles, and `TILE_CFG`
picks the grid cell:

```
first canvas row    = tile_r * TILE_ROWS     (TILE_ROWS = 0 means K)
first canvas column = tile_c * TILE_COLS     (TILE_COLS = 0 means M)
```

The result fills K rows and M columns from that corner. K and M may be
smaller than the grid cell - that is how edge tiles of a result that
doesn't divide evenly are done - but not larger, or the result runs into
the next cell. Each int8 result is written as one byte at its exact canvas
position, so the whole result - every tile - reads back as one dense int8
image.

Rules (not checked by hardware):

- `tile_r * TILE_ROWS` must be a multiple of 16. The output buffers store
  canvas row R in bank `R mod 16`, and the array's row r always writes
  bank r, so a tile can only start on a 16-row boundary. A `TILE_ROWS`
  that is a multiple of 16 satisfies this for every `tile_r`. The first
  column has no such restriction.
- The result must fit inside the canvas: `first column + M <= CANVAS_WIDTH`.
- The canvas must fit inside the output buffers (see
  [Output canvas](#output-canvas)).

`use_prev_out` - 1 = this matmul continues what is already at its
position instead of starting fresh. Used to split a contraction too long
for one matmul: run the first part with `ACT_CFG.out_raw = 1`, then the
later parts with `use_prev_out = 1`, the last one with `out_raw = 0`. The
raw sums are held for one tile only, so run all parts of a tile back to
back (same tile, same `WEIGHT_ROWS`/`INPUT_COLS`) before the next tile.
With `ACT_CFG.prev_int8` it instead adds onto a previous int8 result,
which stays in the output canvas, so other tiles can run in between.

### `CANVAS_WIDTH` (0x80)

The width W of the output canvas in elements - the whole result's column
count when it is built from several tiles. 0 = M, for a single untiled
matmul. It sets the canvas row pitch, used both when results are written
and when the host reads them back; it is latched at `matmul_start`, so keep
it the same for every tile of one result.

### `ACT_CFG` (0x90)

What happens to each output element after its last contraction block.

- `bias_en` - add the row's int32 bias, from the bias buffer, to the
  32-bit sum before the rescale. Canvas row R uses bias entry `R mod 256`.
  The bias is in accumulator units: `bias_q = round(b_real / (w_scale *
  x_scale))` - the int32 bias a standard int8 model export provides.
- `gelu_en` - map the final int8 through the GELU lookup table. The table
  is a 256-entry ROM loaded at build time (`accel_top`'s `GELU_LUT_FILE`
  parameter); its input and output scales are fixed by how it was
  generated (1/32 and 1/32 for the example table).
- `prev_int8` - with `TILE_CFG.use_prev_out`, add onto the int8 result
  already in the output canvas at this position (not raw sums). This
  matmul's raw sums start fresh, and on its last contraction block the
  previous int8 is added to its rescaled value, before saturation and
  GELU, so it must be in the same scale as `y`. Any number of contraction
  blocks.
- `out_raw` - after the last contraction block, keep the raw 32-bit sums
  (in the internal psum buffer): no bias, rescale, GELU, and no int8
  result. For a slice of a split contraction that a later `use_prev_out`
  matmul on the same tile continues, with no other matmul that keeps raw
  sums (more than one contraction block, or `out_raw`) in between. Raw
  sums are not readable by the host - only int8 results are.

## What happens to each output element

For element (k, m) of a matmul, after its last contraction block:

```
s = sum over all contraction blocks of W[k][n] * X[n][m]      32-bit
if use_prev_out and not prev_int8:   s += raw sum kept by the previous matmul
if out_raw:                          keep s (psum buffer), done

if bias_en:   s += bias[canvas row]
y = (s * OUTPUT_SCALE) >> 16
if use_prev_out and prev_int8:       y += int8 already in the output canvas
y = clamp(y, -128, 127)
if gelu_en:   y = GELU_table[y]
write y as one byte at its output canvas position
```

## Output canvas

The output buffer at `OBUF_BASE = 0x9000` (64 KB) holds the finished int8
results as one canvas, `W = CANVAS_WIDTH` (or M) bytes per row, each row
padded to a whole 16-byte word, and reads back as one dense row-major
image:

```
pitch = roundup(W, 16)
int8 (R, C) is at  OBUF_BASE + R * pitch + C
```

Every tile's columns sit side by side in that image, so a whole canvas row
is one contiguous burst. The pitch is the one latched at the last
`matmul_start`.

The raw 32-bit partial sums live in a separate, internal **psum buffer**
(64 KB) that holds only the tile being computed - every tile reuses the
same slots. The host cannot read it.

Capacity: canvas row R is stored in bank `R mod 16` of the output buffer,
so the canvas needs `ceil(total rows / 16) * roundup(W, 16) <= 4096` bytes
(for example 256 x 256), and each tile `ceil(WEIGHT_ROWS / 16) *
INPUT_COLS <= 1024` psum slots (for example 128 x 128). See
[`OUTPUT_BUFFER_LAYOUT.md`](OUTPUT_BUFFER_LAYOUT.md) for how this maps
onto the banks.

## Bias buffer

At `BBUF_BASE = 0x0800`, write-only, 256 entries of little-endian int32:
the bias for canvas row R is at `0x0800 + 4 * R`. Write whole entries (all
four byte enables of an entry; a partly enabled entry is ignored). Write it
before starting a matmul that uses `ACT_CFG.bias_en`, and not while one
runs.

## Examples

**One matmul, int8 result with bias and GELU**

```
WEIGHT_ROWS = K;  WEIGHT_COLS = N;  INPUT_MAX_ADDR = M;  INPUT_COLS = M
OUTPUT_SCALE = round(w_scale * x_scale / gelu_in_scale * 65536)
TILE_CFG = 0;  CANVAS_WIDTH = 0;  TILE_ROWS = 0;  TILE_COLS = 0
ACT_CFG = bias_en | gelu_en
write bias buffer, weights, inputs
CONTROL.matmul_start = 1;  poll STATUS until done
row k's M int8 results: OBUF_BASE + k * roundup(M, 16), bytes 0..M-1
```

**A 40 x 37 result from tiles on a 32 x 22 grid**

```
CANVAS_WIDTH = 37;  TILE_ROWS = 32;  TILE_COLS = 22
tile (0,0): WEIGHT_ROWS = 32, INPUT_COLS = 22, TILE_CFG.tile_r = 0, tile_c = 0
tile (0,1): WEIGHT_ROWS = 32, INPUT_COLS = 15, tile_r = 0, tile_c = 1
tile (1,0): WEIGHT_ROWS =  8, INPUT_COLS = 22, tile_r = 1, tile_c = 0
tile (1,1): WEIGHT_ROWS =  8, INPUT_COLS = 15, tile_r = 1, tile_c = 1
```

Each tile is a full matmul with its own weights, inputs and
`INPUT_MAX_ADDR = INPUT_COLS`. Tile (1,*) starts at row 1 x 32 = 32, a
multiple of 16. Reading back, the result is one dense 40 x 37 int8 image
with rows padded to 48 bytes: row R at `OBUF_BASE + 48 * R`, tile column
1's int8s right after tile column 0's, at bytes 22..36.

**A contraction of 64 split into two matmuls of 32**

```
matmul 1: WEIGHT_COLS = 32, ACT_CFG.out_raw = 1, TILE_CFG.use_prev_out = 0
matmul 2: WEIGHT_COLS = 32, ACT_CFG.out_raw = 0, TILE_CFG.use_prev_out = 1
          (plus bias_en / gelu_en if wanted)
```

Matmul 1 leaves raw 32-bit sums; matmul 2 adds onto them and produces the
int8 result, exactly as one 64-long contraction would.

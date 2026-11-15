# matmul-accel

An int8 weight-stationary systolic matrix-multiply accelerator with an
Avalon-MM slave interface, targeting a DE1-SoC (Cyclone V). The array is
`ARRAY_ROWS x ARRAY_COLS` (currently **16x16**, set in
[`rtl/accel_top.sv`](rtl/accel_top.sv)); larger matrices are tiled into
blocks and sequenced automatically by hardware. Partial sums are kept at
full 32-bit precision across contraction blocks; the finished result is
rescaled to int8 once, with an optional per-row bias and GELU.

Companion docs:

- [`CONTROL_REGISTERS.md`](CONTROL_REGISTERS.md) - what every control
  register means, the order of operations on each output element, and
  programming examples
- [`OUTPUT_BUFFER_LAYOUT.md`](OUTPUT_BUFFER_LAYOUT.md) - how canvas rows,
  columns and tiles map onto the 16 output banks

Contents:

- [Data flow and axis convention](#data-flow-and-axis-convention) - **read this first**,
  the row/column naming is not what you would guess
- [Memory map](#memory-map)
- [Register map](#register-map)
- [Programming the accelerator](#programming-the-accelerator)
- [Buffer layouts](#buffer-layouts)
- [Capacity limits](#capacity-limits)
- [Block indexing](#block-indexing)
- [Buffer RAM IPs](#buffer-ram-ips)
- [Simulation](#simulation)
- [Known issues](#known-issues)

---

## Data flow and axis convention

> **This is the single most confusing thing about this design, and several
> in-tree comments get it wrong.** The headers of
> [`rtl/weight_ctrl.v`](rtl/weight_ctrl.v) and
> [`rdl/ctrl_reg.rdl`](rdl/ctrl_reg.rdl) both claim `WEIGHT_ROWS` is the
> contraction dimension. **It is not.** The description below is derived
> from [`rtl/pe.v`](rtl/pe.v) / [`rtl/pe_array.v`](rtl/pe_array.v)
> directly, and is confirmed by the passing testbench.

The accelerator computes a plain, untransposed

```
Y = W @ X          W is (K x N),  X is (N x M),  Y is (K x M)
```

where the contraction runs over the shared `N`. Mapping onto the array:

| Matrix axis | Register | PE array axis | Role |
|---|---|---|---|
| `W` rows, `Y` rows (`K`) | `WEIGHT_ROWS` | `ARRAY_ROWS` (`r`) | **output** rows |
| `W` cols, `X` rows (`N`) | `WEIGHT_COLS` | `ARRAY_COLS` (`c`) | **contraction** |
| `X` cols, `Y` cols (`M`) | `INPUT_COLS` | streamed over time | output columns |

Why, from the RTL:

- Partial sums cascade **across columns within a row**
  (`pe_array.v`: `pe_b_in = ROW[r].COL[c-1].pe_p_out`), and each row's final
  sum is tapped at its rightmost column
  (`if (c == ARRAY_COLS-1) assign p_out[r] = pe_p_out`). So the summation -
  the contraction - happens along `c` / `ARRAY_COLS`, and **each PE row
  independently produces one output row**.
- Activations flow **down** each column from the top (`pe_a_in` for `r==0`
  comes from `a_in[c]`, for `r>0` from the PE above), so `input_buffer` bank
  `c` feeds PE column `c` - the contraction axis.
- Weights load straight into `PE(row,col)` with no transpose
  (`w_addr = row*ARRAY_COLS + col`, and `weight_loader` walks
  `weight_buffer` row-major into that same layout), so `PE(r,c)` holds
  `W[r][c]`.

Putting those together: output lane `r` = `sum over c of W[r][c] * X[c][m]`.

**Practical consequence:** if your weight matrix is stored
`(out_features x in_features)` - the usual PyTorch `nn.Linear` layout - then
program it directly, no transpose:
`WEIGHT_ROWS = out_features`, `WEIGHT_COLS = in_features`.

---

## Memory map

One flat Avalon-MM slave. Each region is a simple range check in
[`rtl/accel_top.sv`](rtl/accel_top.sv); the region-local (offset-subtracted)
address drives that region's RAM or register file directly.

| Region | Base | Size | Access | Contents |
|---|---|---|---|---|
| `CTRL_BASE` | `0x0000_0000` | `0x100` | R/W, 32-bit words | Control/status register file (`ctrl_rf`) |
| `BBUF_BASE` | `0x0000_0800` | `0x400` (1 KB) | **write-only**, int32 | `bias_buffer` (banked): one int32 bias per canvas row |
| `WBUF_BASE` | `0x0000_1000` | `0x4000` (16 KB) | **write-only**, byte | `weight_buffer` (banked) |
| `IBUF_BASE` | `0x0000_5000` | `0x4000` (16 KB) | **write-only**, byte | `input_buffer` (banked) |
| `OBUF_BASE` | `0x0000_9000` | `0x10000` (64 KB) | **read-only**, byte | `output_buffer`: the int8 output canvas |

The whole map spans `0x0000_0000`-`0x0001_8FFF`, so the slave needs at
least a 17-bit byte address (a 128 KB span). `OBUF_BASE` moved from
`0x7000` to `0x9000` when `IBUF` grew to 16 KB, and the `OBUF` window grew
from 16 KB to 64 KB for the output canvas, e.g. 256x256. `BBUF` sits in
the gap below `WBUF`. The 32-bit partial sums live in a separate 256 KB
`psum_buffer` that is internal to the accelerator and has no window.

Access conventions:

- **All buffer accesses are full 128-bit words.** The host-facing port of
  every buffer is `AVS_DATA_WIDTH` (128) wide; the core-facing port is
  narrower (a byte for weights, inputs and int8 results, 32 bits for the
  bias). So one access moves 16 bytes and the byte address
  advances by **16**, not 1. `avs_byteenable` is **not** honoured for
  weight and input writes - there is no byte-granular write. The bias
  buffer honours it per 4-byte entry: an entry is written only when all
  four of its byte enables are set.
- **Byte order is little-endian within a word**: byte address `16n+j` sits
  at `avs_writedata[8*j +: 8]`, matching Avalon byte lanes.
- **Write whole contiguous images.** Write `W` and `X` each as one
  row-major image with every row zero-padded to a multiple of 16 bytes (see
  [Buffer layouts](#buffer-layouts)); hardware scatters the rows to banks.
- **Results read back as one contiguous int8 canvas.** It is `W` bytes
  wide, `W = CANVAS_WIDTH` (or `INPUT_COLS` when that is 0), and canvas row
  `R` starts at `OBUF_BASE + R*pitch(W)`. A matmul writes only its own tile
  of the canvas; every other byte keeps what it held. Raw 32-bit partial
  sums are not readable. See
  [Output canvas and tiling](#output-canvas-and-tiling).
- **Each register is its own 128-bit bus word** at the 16-byte-aligned
  offsets below; its value sits in the low bytes (lanes 0..3), so a write
  needs `avs_byteenable` bits 0..3 set.
- **Reads are pipelined with variable latency.** Every beat is marked by
  `avs_readdatavalid`, never in the cycle the read is accepted: `CTRL`
  beats arrive 1 cycle after acceptance, `OBUF` beats 3 cycles after,
  because the read address goes through the same remap as the writes. The
  slave holds one read in flight at a time, using `avs_waitrequest`
  to stall the next command until the previous read has returned its last
  beat. Declare `maximumPendingReadTransactions = 1`.
- **Burst reads are supported** for `CTRL` and `OBUF`: drive
  `avs_burstcount`, hold `avs_read` until `avs_waitrequest` drops, then
  take one beat per `avs_readdatavalid` pulse, address auto-incrementing.
- At 16x16 the weight and input buffers are 16 banks x 1 KB, the output
  buffer 16 banks x 4 KB and the bias buffer 16 banks x 16 int32 entries,
  each filling its window exactly. Growing a bank depth or the array means growing the matching
  window in `rtl/accel_top.sv`; elaboration fails if a window is too small
  to reach every bank.

---

## Register map

Defined in [`rdl/ctrl_reg.rdl`](rdl/ctrl_reg.rdl), compiled to
`ctrl_rf_rf.sv` by `rdl/python_gen.py` (PeakRDL). Offsets are relative to
`CTRL_BASE` (`0x0000_0000`). Full descriptions, the order of operations on
each output element and programming examples are in
[`CONTROL_REGISTERS.md`](CONTROL_REGISTERS.md).

| Offset | Register | Field | Bits | Access | Description |
|---|---|---|---|---|---|
| `0x00` | `CONTROL` | `matmul_start` | `[0]` | W, **self-clearing pulse** | Starts a matmul: latches `WEIGHT_ROWS`/`WEIGHT_COLS`, `TILE_CFG`, `TILE_ROWS`/`TILE_COLS`, `ACT_CFG` and the canvas pitch, and resets the block position to (0,0). |
| `0x10` | `WEIGHT_ROWS` | `value` | `[15:0]` | W | `K` - **output rows**. May be any value; a non-multiple of `ARRAY_ROWS` is handled by masking. |
| `0x20` | `WEIGHT_COLS` | `value` | `[15:0]` | W | `N` - **contraction size**. **Must be a multiple of `ARRAY_COLS`** (see [Known issues](#known-issues)). |
| `0x30` | `INPUT_MAX_ADDR` | `value` | `[9:0]` | W | `M` - number of reads issued per input band. Set equal to `INPUT_COLS`. |
| `0x40` | `INPUT_COLS` | `value` | `[15:0]` | W | `M` - output/activation column count of this matmul; also the canvas width when `CANVAS_WIDTH` is 0, and the tile grid width when `TILE_COLS` is 0. |
| `0x50` | `STATUS` | `busy` | `[0]` | R | Live level. Compute in progress: `weight_ctrl` sequencing **or** `output_loader` still draining. |
| `0x50` | `STATUS` | `done` | `[1]` | R, **read-to-clear** | Sticky. Set once per matmul, when every block is committed **and** the last one has drained. Cleared by the read that observes it. |
| `0x60` | `OUTPUT_SCALE` | `value` | `[15:0]` | W | Q0.16 unsigned rescale factor; actual scale = `value/65536`, range `[0,1)`. Applied once, to the full 32-bit sum - see below. |
| `0x70` | `TILE_CFG` | `tile_r` | `[14:0]` | W | Tile row index on the tile grid: this matmul's result starts at canvas row `tile_r * TILE_ROWS`, which must be a multiple of 16. |
| `0x70` | `TILE_CFG` | `tile_c` | `[29:15]` | W | Tile column index: the result starts at canvas column `tile_c * TILE_COLS`. |
| `0x70` | `TILE_CFG` | `use_prev_out` | `[30]` | W | 1 = continue what is already at this position (the previous matmul's raw sums, or with `ACT_CFG.prev_int8` its int8 result) instead of starting fresh. See [Output canvas and tiling](#output-canvas-and-tiling). |
| `0x80` | `CANVAS_WIDTH` | `value` | `[15:0]` | W | Canvas width `W`: the whole result's column count. 0 = use `INPUT_COLS` (one untiled matmul). Rows are padded to `pitch(W)`. Latched at `matmul_start` for both the writes and the host reads; keep it the same for every tile of one result. |
| `0x90` | `ACT_CFG` | `bias_en` | `[0]` | W | Add the canvas row's int32 bias (bias buffer) to the sum before the rescale. |
| `0x90` | `ACT_CFG` | `gelu_en` | `[1]` | W | Map the final int8 through the GELU table. |
| `0x90` | `ACT_CFG` | `prev_int8` | `[2]` | W | With `use_prev_out`: add onto the int8 result already in the output canvas, after the rescale. The raw sums start fresh; any number of contraction blocks. |
| `0x90` | `ACT_CFG` | `out_raw` | `[3]` | W | Keep the raw 32-bit sums (internal psum buffer): no bias, rescale, GELU or int8 result. For a later `use_prev_out` matmul - raw sums are not host-readable. |
| `0xA0` | `TILE_ROWS` | `value` | `[15:0]` | W | Tile grid height (the largest tile's row count). 0 = `WEIGHT_ROWS`. |
| `0xB0` | `TILE_COLS` | `value` | `[15:0]` | W | Tile grid width (the largest tile's column count). 0 = `INPUT_COLS`. |

Every register resets to 0, and only `STATUS` can be read back. Except
`STATUS`, the registers latched at `matmul_start` (above) can be
reprogrammed while a matmul runs; `INPUT_COLS`, `INPUT_MAX_ADDR` and
`OUTPUT_SCALE` are used throughout the matmul and must not change until it
is done.

The two bits behave differently, and the difference matters:

- **`busy` is a live level.** Reading it is free of side effects; it reads 0
  once the accelerator is idle.
- **`done` is a sticky, read-to-clear event.** Hardware sets it with a
  one-cycle pulse when the matmul actually completes, and the software read
  that returns a 1 also clears it. This is what makes back-to-back matmuls
  safe: a level saying "some matmul finished" cannot be told apart from the
  previous run's leftover, whereas a captured event is consumed exactly
  once.

> **Any read of `STATUS` clears `done`** - the whole register is one
> address, so a read intended only to check `busy` will silently consume a
> pending completion. Read `STATUS` **once** per poll iteration and test
> both bits from that one value; never read it twice expecting the same
> answer.

If a completion pulse arrives on the same cycle as a software read, the
hardware write wins and the bit stays set for the next read, so an event
cannot be lost to that collision. PeakRDL does not generate that priority:
[`rdl/python_gen.py`](rdl/python_gen.py) patches it into the generated
register file after export, and `test_done_set_wins_over_status_read`
checks it.

### `OUTPUT_SCALE` and the quantization scheme

The array accumulates at `ACC_WIDTH` (32-bit) so a full contraction of
int8 x int8 products cannot overflow. The output canvas keeps that 32-bit
sum between contraction blocks, and only after the last block does
`output_loader` produce the int8 result:

```
s      = full 32-bit sum            (+ bias[canvas row]  if ACT_CFG.bias_en)
scaled = saturate_int8( (s * OUTPUT_SCALE) >>> 16 )
y      = GELU_table[scaled]         if ACT_CFG.gelu_en, else scaled
```

Under per-tensor symmetric int8 quantization with scales `w_scale`,
`x_scale` and `y_scale`, the value to program is the single combined factor

```
OUTPUT_SCALE = round( w_scale * x_scale / y_scale * 65536 )
```

which must land in `[0, 65536)` - i.e. the combined scale must be `< 1.0`.
The bias is int32 in accumulator units, as a standard int8 model export
provides it: `bias_q = round(b_real / (w_scale * x_scale))`. With GELU,
`y_scale` must be the GELU table's input scale (1/32 for the example table
in `rtl/activation/example/`), and the result comes out in its output
scale (also 1/32).

> Earlier versions rescaled and saturated each contraction block's psum to
> int8 before accumulating it, so requantization error compounded across
> blocks. With the 32-bit canvas the result is exactly one requantization
> of the full-precision sum, whatever the number of contraction blocks
> (`golden_blocked` in the testbench).

---

## Programming the accelerator

```
 1. (optional) poll STATUS.busy == 0
 2. write WEIGHT_ROWS    = K
    write WEIGHT_COLS    = N         # must be a multiple of ARRAY_COLS
    write INPUT_MAX_ADDR = M
    write INPUT_COLS     = M
    write OUTPUT_SCALE   = round(w_scale * x_scale / y_scale * 65536)
    write TILE_CFG       = tile_r | tile_c << 15 | use_prev_out << 30
    write TILE_ROWS      = tile grid height  # 0 for a single matmul or equal tiles
    write TILE_COLS      = tile grid width   # 0 for a single matmul or equal tiles
    write CANVAS_WIDTH   = full result width # 0 for a single, untiled matmul
    write ACT_CFG        = bias_en | gelu_en << 1 | prev_int8 << 2 | out_raw << 3
 3. write weight bytes into WBUF  (layout below)
    write input  bytes into IBUF  (layout below)
    write int32 biases into BBUF  (only with bias_en; layout below)
 4. write CONTROL.matmul_start = 1  # single pulse, self-clearing
 5. poll STATUS until bit[1] (done) reads 1
    # NB: that read consumes done - keep the value, don't re-read
 6. read the result from OBUF     (layout below)
    # canvas row R at OBUF_BASE + R*pitch(W); with tiling, after the last tile
```

Step 4 is the **only** software trigger. From there `weight_ctrl` sequences
the entire matmul itself - per block it prefetches weights, preloads the
input band, commits the weights into the array, streams, and advances to the
next block - with nothing left for software to poke mid-matmul.

Step 2 must come before step 3: the weight and input writes are placed
using the live `WEIGHT_COLS` and `INPUT_MAX_ADDR`. Both must complete
before step 4, because `matmul_start` latches the dimension and tile
registers.

> **Do not skip step 5's poll before reading.** `output_loader` keeps
> writing `output_buffer` for some cycles after the last block is committed;
> `STATUS.done` accounts for that drain, and reading early returns
> partly-stale data.

---

## Buffer layouts

`ARRAY_ROWS`/`ARRAY_COLS` below are the array dimensions (16). The weight
and input buffers have 1 KB banks, the output buffer 4 KB banks.
`pitch(c) = roundup(c, 16)` is a row padded to whole 128-bit words.

All three are contiguous row-major images with padded rows: `W` and `X`
written, the output canvas read. `W` is stored exactly as written on paper,
with no transpose:

```
addr(k, n) = WBUF_BASE + k*pitch(N) + n     W: k in [0,K), n in [0,N)    write
addr(p, m) = IBUF_BASE + p*pitch(M) + m     X: p in [0,N), m in [0,M)    write
addr(R, C) = OBUF_BASE + R*pitch(W) + C     canvas: C in [0,W)           read
```

`W` is the canvas width: `CANVAS_WIDTH`, or `INPUT_COLS` when that is 0.
Inside each buffer, [`rtl/buf_addr_remap.v`](rtl/buf_addr_remap.v) maps row
`r` to bank `r % 16`, at offset `(r / 16) * pitch` within the bank. It uses
the live `WEIGHT_COLS` and `INPUT_MAX_ADDR`, so program them **before**
writing the inputs; for output reads it uses the canvas pitch latched at
`matmul_start`, so a result reads back with the layout it was written with.
`output_loader` writes each canvas row at that same padded pitch.

Results are **int8**; sign-extend the byte you read back. The raw 32-bit
partial sums live in the internal `psum_buffer` - the same canvas, `W`
slots per row - and are not readable.
[`OUTPUT_BUFFER_LAYOUT.md`](OUTPUT_BUFFER_LAYOUT.md) walks through both
buffers, tiles included, with a worked example.

### Output canvas and tiling

The output buffer holds one int8 canvas, `W` bytes wide, where
`W = CANVAS_WIDTH` is the whole result's column count. Each matmul writes
its `K x M` result as one tile of it. The canvas is divided into a grid of
`TILE_ROWS x TILE_COLS` cells, and `TILE_CFG` picks the cell:

```
first row     R0 = tile_r * TILE_ROWS     (TILE_ROWS = 0 means K)
first column  C0 = tile_c * TILE_COLS     (TILE_COLS = 0 means M)
result[k][m]  ->  canvas[R0 + k][C0 + m]
```

Size the grid to the largest tile. `K` and `M` may be smaller than the
grid cell, so the edge tiles of a result that doesn't divide evenly
compute only what they need and still land in place.

With `TILE_CFG = 0` and `CANVAS_WIDTH = 0` that is `canvas[k][m]` with
`W = M`: a single untiled matmul. To build a larger result, set
`CANVAS_WIDTH` to its full width and `TILE_ROWS`/`TILE_COLS` to the tile
size, run one matmul per tile, and read the canvas once at the end - it
comes back as one dense row-major matrix. With a
128-deep contraction, the weight and input buffers cap one matmul's result
at 128x128, so the 64 KB buffer holds 2x2 such tiles: `Y` (256x256) is
`CANVAS_WIDTH = 256` and four matmuls with `(tile_r, tile_c)` = (0,0),
(0,1), (1,0), (1,1). A 40x37 result on a 32x22 grid is `TILE_ROWS = 32`,
`TILE_COLS = 22` and tiles of 32x22, 32x15, 8x22 and 8x15
(`test_tiled_edge_tiles`).

`use_prev_out` splits a contraction too long for one matmul. With it set,
the matmul's first contraction block continues the raw 32-bit sums the
previous matmul left in `psum_buffer` instead of starting fresh. Run every
part but the last with `ACT_CFG.out_raw = 1`, so it keeps its raw sums;
the last part (`out_raw = 0`) adds onto them and produces the int8 result,
identical to one long matmul. Alternatively, `ACT_CFG.prev_int8` adds this
matmul's result onto the previous **int8** result in the output canvas
(after the rescale, before GELU), with any number of contraction blocks.

Rules - nothing in hardware checks them:

- `tile_r * TILE_ROWS` must be a multiple of 16, so the tile's rows start
  on a bank boundary; a `TILE_ROWS` that is a multiple of 16 always works.
  Column offsets and widths can be anything.
- A tile must not be larger than its grid cell, or it runs into the next.
- The tile must lie inside the canvas width, `C0 + M <= W`, and the canvas
  inside both buffers: `ceil(total rows / 16) * pitch(W) <= 4096` bytes
  and `ceil(total rows / 16) * W <= 4096` psum slots.
- `CANVAS_WIDTH` stays the same for every tile of one result.
- `TILE_CFG`, `TILE_ROWS`, `TILE_COLS`, `CANVAS_WIDTH` and `ACT_CFG` are
  latched at `matmul_start`, like the dimension registers; host reads use
  the canvas pitch latched by the last `matmul_start`.

### Bias buffer

256 little-endian int32 entries at `BBUF_BASE`, write-only: the bias for
canvas row `R` is at `BBUF_BASE + 4*R` (rows past 255 wrap, `R mod 256`).
It is only read with `ACT_CFG.bias_en`. One 128-bit write carries four
consecutive rows, and an entry is written only when all four of its byte
enables are set. Write it before `matmul_start`, not while a matmul that
uses it runs.

Growing the buffer itself takes `OBUF_BANK_ADDR_WIDTH` and the `OBUF`
window in [`rtl/accel_top.sv`](rtl/accel_top.sv), a matching
`output_buffer` IP, and `OBUF_BANK_ADDR_WIDTH` at the top of
`tb/test_accel_top.py`. The canvas shape within it is set at run time.

---

## Capacity limits

All three buffers are fixed-size, so software must check these before
programming a matmul:

| Constraint | Bound |
|---|---|
| `ceil(K / ARRAY_ROWS) * pitch(N) <= 1024` | weight_buffer per-bank depth |
| `ceil(N / ARRAY_COLS) * pitch(M) <= 1024` | input_buffer per-bank depth |
| `C0 + M <= W` and `ceil(total rows / 16) * pitch(W) <= 4096` | output canvas (4096 bytes per bank), `W` = canvas width |
| `ceil(total rows / 16) * W <= 4096` | psum buffer (4096 32-bit slots per bank) |
| `(tile_r * TILE_ROWS) % 16 == 0` | tile rows start on a bank boundary |
| `M <= 1023` | `INPUT_MAX_ADDR` is a 10-bit field |
| canvas rows using a bias `< 256` | bias buffer depth (rows wrap mod 256) |

`K = N = M = 128` fills the weight and input buffers exactly
(`test_128x128x128_matmul`); its result fills a quarter of the output buffer. Beyond that the input bounds trade off against
each other: larger `K` or `N` needs a smaller `M`, and vice versa. Splitting
the contraction across matmuls with `use_prev_out` (every part but the last
with `ACT_CFG.out_raw`) removes the limit on `N`.

---

## Block indexing

Tracked in [`rtl/weight_ctrl.v`](rtl/weight_ctrl.v). **Beware the names** -
they follow the register names, which per the
[axis convention](#data-flow-and-axis-convention) are the opposite of what
they suggest:

| Index | Steps over | Array axis | Meaning |
|---|---|---|---|
| `k_blk_idx` | `WEIGHT_ROWS` (`K`) | `ARRAY_ROWS` | **output**-row block |
| `n_blk_idx` | `WEIGHT_COLS` (`N`) | `ARRAY_COLS` | **contraction** block |

Traversal is `k_blk_idx` fast/inner, `n_blk_idx` slow/outer:

```
if not last block:
    if is_last_k_blk:  k_blk_idx = 0;  n_blk_idx += 1
    else:              k_blk_idx += 1
```

At each **commit** (`commit_pulse`/`w_load`, once the block's weights are
prefetched and its input band primed) `weight_ctrl` snapshots
`committed_k_blk_idx` / `committed_n_blk_idx` for `output_loader`. These are
snapshots rather than the live counters because `weight_ctrl` has usually
already advanced to prefetching the next block by the time `output_loader`
finishes draining the current one. `output_loader` samples them once at its
own start edge and holds them for that block's whole drain.

They then select:

- **`held_k_blk_idx`** - which canvas row block the pass writes. For a tile
  starting at row `R0 = tile_r * TILE_ROWS` and column `C0 = tile_c *
  TILE_COLS`, bank `r` gets canvas row `R0 + 16*k_blk + r`, at
  `(R0/16 + k_blk) * pitch + C0 + m` in each buffer (pitch in bytes for
  `output_buffer`, slots for `psum_buffer`): different output-row blocks
  write **disjoint** segments (concatenation, not accumulation). See
  [`OUTPUT_BUFFER_LAYOUT.md`](OUTPUT_BUFFER_LAYOUT.md).
- **`held_n_blk_idx`** - what the pass does:
  - the first contraction block (`n_blk == 0`, unless `use_prev_out`
    continues raw sums) writes its raw psum to `psum_buffer`;
  - every later block read-modify-writes the raw 32-bit running sum there;
  - the last block (`n_blk == WEIGHT_COLS/16 - 1`, unless `ACT_CFG.out_raw`)
    instead adds the bias, rescales to int8, applies GELU, and writes the
    int8 to `output_buffer`.

> `committed_first_k_blk` / `held_first_k_blk` are wired through and latched
> but **never read** - the accumulate decision uses `held_n_blk_idx == 0`
> instead. Dead signal.

---

## Buffer RAM IPs

`weight_buffer`, `input_buffer`, `output_buffer`, `psum_buffer` and
`bias_buffer` are placeholder leaf RAMs (no `rtl/*.v` source checked in -
only instantiated) standing in for Quartus-generated On-Chip Memory IPs.
[`tb/models/buffer_ram_models.v`](tb/models/buffer_ram_models.v) provides
simulation-only behavioral stand-ins with matching names and ports.

All five are **simple dual-port** RAMs - independent
`rdaddress`/`wraddress`, each with its own `rden`/`wren`, one cycle of
registered read latency. The first three are **mixed-width**, with the
host-facing port 128 bits wide and the core-facing port byte wide.
`psum_buffer` and `bias_buffer` are same-width 32-bit RAMs: `psum_buffer`
is only ever touched by `output_loader`, and `bias_buffer`'s wrapper
([`rtl/bias_buffer_32_bank.v`](rtl/bias_buffer_32_bank.v)) splits each
128-bit host word into four 32-bit entries for four banks. Port shapes
below are per bank:

| Buffer | Instances | Host port (Avalon) | Core port | Total | AVS region |
|---|---|---|---|---|---|
| `weight_buffer` | `ARRAY_ROWS` (16) | **write** 64 x 128, 6-bit | read 1024 x 8, 10-bit (`weight_loader`) | 16 KB | `WBUF_SIZE = 0x4000` |
| `input_buffer` | `ARRAY_ROWS` (16) | **write** 64 x 128, 6-bit | read 1024 x 8, 10-bit (`input_dispatch`) | 16 KB | `IBUF_SIZE = 0x4000` |
| `output_buffer` | `ARRAY_COLS` (16) | **read** 256 x 128, 8-bit | write 4096 x 8, 12-bit (`output_loader`) | 64 KB | `OBUF_SIZE = 0x10000` |
| `psum_buffer` | `ARRAY_COLS` (16) | none | read and write 4096 x 32, 12-bit (`output_loader`) | 256 KB | none |
| `bias_buffer` | `ARRAY_ROWS` (16) | **write** 16 x 32, 4-bit (via the wrapper) | read 16 x 32, 4-bit (`output_loader`) | 1 KB | `BBUF_SIZE = 0x400` |

> `output_buffer` is back to its original int8 shape (4096 x 8 write,
> 256 x 128 read), and `psum_buffer` and `bias_buffer` are new. Generate
> all three On-Chip Memory IPs at the shapes above before building in
> Quartus. A `bias_buffer` bank is only 512 bits - set its memory block
> type to MLAB, or each of the 16 takes a whole M10K. In total the
> buffers are about 350 KB of block RAM.

The core side is narrow because `weight_loader` and `input_dispatch`
consume one byte per cycle and `output_loader` writes one int8 result per
cycle. Configure each generated On-Chip Memory with these two port widths;
byte ordering must be little-endian (narrow address `16n+j` = wide bit
`8*j`), which is altsyncram's default.

> `output_buffer`'s single read port is shared - `output_loader` (reading
> a previous int8 result for `ACT_CFG.prev_int8`) and the Avalon host both
> use it, muxed by `busy`. Since that port is 128 bits, `output_loader`
> reads the *word* containing its byte and selects the lane with the
> delayed address; its write goes through the byte-wide write port.

Notes:

- The banked wrappers' flat addresses are derived, not hardcoded:
  `*_ADDR_WIDTH = $clog2(16) + 10` (14 bits at 16 banks), split as
  bank-select in the upper bits and per-bank offset in the lower.
- The `_32_bank` in the `*_buffer_32_bank` wrappers is historical; all of
  them are fully parameterized on bank count.
- The GELU table is not one of these IPs: it is 16 inferred 256 x 8 ROMs
  (one per output lane, in `rtl/activation/rtl/gelu_int8.sv`), initialised
  by `$readmemh` from `accel_top`'s `GELU_LUT_FILE` parameter - a path
  relative to the simulator's or Quartus project's working directory.
  `tb/Makefile` passes an absolute one.
- `accel_top` overrides every wrapper's `BANK_ADDR_WIDTH`. Size the
  generated IPs from the `*_BANK_ADDR_WIDTH` localparams there, not from
  the module defaults.
- Regenerating any IP at a different depth means updating the matching
  `*_SIZE` / `*_BANK_ADDR_WIDTH` localparams in `rtl/accel_top.sv`.

### DSP usage

Every multiplier carries a `(* multstyle = "logic" *)` attribute (`pe.v`,
`output_loader.v`, `weight_ctrl.v`) forcing **soft logic** rather than DSP
blocks - a full array needs far more DSP blocks than a Cyclone V has. Set it
back to `"dsp"` only on a device with the budget, or override project-wide
with `set_global_assignment -name DSP_BLOCK_BALANCING "LOGIC ELEMENTS"`.

---

## Simulation

cocotb + Icarus Verilog:

```sh
cd tb && make          # run the suite
make clean
```

| Test | Covers |
|---|---|
| `test_single_block_matmul` | One full block |
| `test_multiblock_matmul` | 2x2 blocks - multi-contraction-block accumulation (full 32-bit, one rescale) and multi-output-block addressing |
| `test_partial_row_block_matmul` | Partial last **output-row** block (`K` not a multiple of `ARRAY_ROWS`): `valid_row` masking, `weight_loader` zero-padding, `output_loader` row gating |
| `test_back_to_back_matmul_no_reset` | Two matmuls with no reset between - regression test for stale `STATUS.done`/`busy` |
| `test_128x128x128_matmul` | `W` (128x128) @ `X` (128x128): 64 blocks, fills every weight and input bank to its full 1 KB, and is the only test with `M` larger than the 32-entry input FIFOs. Reads the result back in one 2047-beat burst, the 11-bit `avs_burstcount` maximum. Logs cycles from `matmul_start` to `STATUS.done`. Takes about half a minute in Icarus. |
| `test_tiled_canvas_use_prev_out` | `Y` (256x256) as 2x2 tiles of 128x128, each tile's contraction split into two `use_prev_out` matmuls (the first halves with `ACT_CFG.out_raw`): eight matmuls filling the whole output buffer, read back once. |
| `test_tiled_unaligned_col_offset` | 2x2 tiles of 16x37 (two contraction blocks) on a 74-wide canvas: rows padded to 80 bytes (5 words, not a power of two), and column tiles start mid-word, sharing bus words with their left neighbour |
| `test_tiled_edge_tiles` | A 40x37 result on a 32x22 tile grid (`TILE_ROWS`/`TILE_COLS`): smaller bottom and right edge tiles land on the grid, and the right tiles start mid-word |
| `test_bias_gelu` | `ACT_CFG.bias_en` + `gelu_en` on 3 output-row x 3 contraction blocks - every output-row block gets its bias and GELU exactly once, on its last pass - then a bias-only tile at tile row 1, checking the bias is indexed by canvas row |
| `test_prev_int8_out_raw` | `ACT_CFG.out_raw` (raw 32-bit sums, bias ignored), a `use_prev_out` matmul continuing them to the int8 result of the whole contraction, then a two-contraction-block `use_prev_out` + `prev_int8` (+ bias, GELU) matmul adding onto that int8 result |
| `test_done_set_wins_over_status_read` | Times a `STATUS` read to land exactly on the done pulse; the done event must survive. Guards the patch `rdl/python_gen.py` applies to the generated register file. |

Test dimensions derive from `ARRAY_ROWS`/`ARRAY_COLS` (top of
`tb/test_accel_top.py`, which must match `accel_top.sv`), so the suite
follows an array resize. The GELU tests read the same table the hardware
loads, `rtl/activation/example/gelu_int8.hex`.

---

## Known issues

- **`WEIGHT_COLS` must be a multiple of `ARRAY_COLS`.** The design masks a
  partial last block only on the output-row axis (`valid_row` ->
  `output_loader`'s `row_live`). There is no equivalent masking of a partial
  *contraction*: `weight_ctrl` computes `valid_col`/`valid_n`, but nothing
  gates the contribution of input banks past it. Pad `N` up to a multiple of
  `ARRAY_COLS` with zeros.
- **The array is square-only.** `pe_array` indexes `a_in`/`a_en` by column
  while declaring them `ARRAY_ROWS` wide, and indexes `b_en` by row while
  declaring it `ARRAY_COLS` wide; `output_loader` banks per `p_out` lane
  while naming that axis `ARRAY_COLS`. Widths line up only while
  `ARRAY_ROWS == ARRAY_COLS`. Resize both together.
- **`weight_loader`'s out-of-range PE zeroing does not work as documented.**
  Its header says out-of-range PEs "get an explicit 0", but
  `w_en = rd_valid_d` is low for exactly those PEs, so the
  `w_data = ... : 0` branch never reaches them - they retain the previous
  block's weight. Harmless on the output-row axis (a stale row only feeds an
  output lane that `row_live` masks off), but it means the zero-fill cannot
  be relied on if contraction-axis masking is ever added.
- **Raw 32-bit sums are not host-readable.** They live in the internal
  `psum_buffer`; a matmul run with `ACT_CFG.out_raw` is only useful to a
  later `use_prev_out` matmul.
- **Tile placement is not checked.** A tile whose first row
  (`tile_r * TILE_ROWS`) is not a multiple of 16 is silently moved up to
  the previous multiple, and a tile larger than its grid cell overwrites
  its neighbour.
- **Stale axis comments.** `rtl/weight_ctrl.v`'s and `rdl/ctrl_reg.rdl`'s
  headers describe `WEIGHT_ROWS` as the contraction dimension. They are
  wrong; see [Data flow and axis convention](#data-flow-and-axis-convention).

See also [`rdl/README.md`](rdl/README.md) for the register file's own notes.

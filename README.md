# matmul-accel

An int8 weight-stationary systolic matrix-multiply accelerator with an
Avalon-MM slave interface, targeting a DE1-SoC (Cyclone V). The array is
`ARRAY_ROWS x ARRAY_COLS` (currently **16x16**, set in
[`rtl/accel_top.sv`](rtl/accel_top.sv)); larger matrices are tiled into
blocks and sequenced automatically by hardware.

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
| `WBUF_BASE` | `0x0000_1000` | `0x4000` (16 KB) | **write-only**, byte | `weight_buffer` (banked) |
| `IBUF_BASE` | `0x0000_5000` | `0x4000` (16 KB) | **write-only**, byte | `input_buffer` (banked) |
| `OBUF_BASE` | `0x0000_9000` | `0x4000` (16 KB) | **read-only**, byte | `output_buffer` (banked) |

The whole map spans `0x0000_0000`-`0x0000_CFFF`, so the slave needs at
least a 16-bit byte address. `OBUF_BASE` moved from `0x7000` to `0x9000`
when `IBUF` grew to 16 KB.

Access conventions:

- **All buffer accesses are full 128-bit words.** The buffers are
  *mixed-width*: the host-facing port is `AVS_DATA_WIDTH` (128) wide, the
  core-facing port stays byte-wide. So one access moves 16 bytes and the
  byte address advances by **16**, not 1. `avs_byteenable` is **not**
  honoured for buffer accesses - there is no byte-granular write.
- **Byte order is little-endian within a word**: byte address `16n+j` sits
  at `avs_writedata[8*j +: 8]`, matching Avalon byte lanes.
- **Write whole contiguous images.** Write `W` and `X` each as one
  row-major image with every row zero-padded to a multiple of 16 bytes (see
  [Buffer layouts](#buffer-layouts)); hardware scatters the rows to banks.
- **Reads may need covering-word arithmetic.** Output segments start at
  `k_blk*M`, which is only word-aligned when `M` is a multiple of 16. To
  read an unaligned run, burst the words that cover it and unpack.
- **Each register is its own 128-bit bus word** at the 16-byte-aligned
  offsets below; its value sits in the low bytes (lanes 0..3), so a write
  needs `avs_byteenable` bits 0..3 set.
- `output_buffer` reads have one cycle of latency and assert
  `avs_waitrequest`; `avs_readdatavalid` marks valid beats.
- **Burst reads are supported** for `CTRL` and `OBUF`: drive
  `avs_burstcount`, hold `avs_read` until `avs_waitrequest` drops (that
  accepting cycle already carries the first beat), then take one beat per
  `avs_readdatavalid` pulse, address auto-incrementing.
- At 16x16 every buffer is 16 banks x 1 KB, which fills its 16 KB window
  exactly. Growing a bank depth or the array means growing the matching
  window in `rtl/accel_top.sv`; elaboration fails if a window is too small
  to reach every bank.

---

## Register map

Defined in [`rdl/ctrl_reg.rdl`](rdl/ctrl_reg.rdl), compiled to
`ctrl_rf_rf.sv` by `rdl/python_gen.py` (PeakRDL). Offsets are relative to
`CTRL_BASE` (`0x0000_0000`).

| Offset | Register | Field | Bits | Access | Description |
|---|---|---|---|---|---|
| `0x00` | `CONTROL` | `matmul_start` | `[0]` | W, **self-clearing pulse** | Starts a matmul: latches `WEIGHT_ROWS`/`WEIGHT_COLS` and resets the block position to (0,0). |
| `0x10` | `WEIGHT_ROWS` | `value` | `[15:0]` | W | `K` - **output rows**. May be any value; a non-multiple of `ARRAY_ROWS` is handled by masking. |
| `0x20` | `WEIGHT_COLS` | `value` | `[15:0]` | W | `N` - **contraction size**. **Must be a multiple of `ARRAY_COLS`** (see [Known issues](#known-issues)). |
| `0x30` | `INPUT_MAX_ADDR` | `value` | `[9:0]` | W | `M` - number of reads issued per input band. Set equal to `INPUT_COLS`. |
| `0x40` | `INPUT_COLS` | `value` | `[15:0]` | W | `M` - output/activation column count. Becomes `output_loader.total_rows`, the per-bank segment stride. |
| `0x50` | `STATUS` | `busy` | `[0]` | R | Live level. Compute in progress: `weight_ctrl` sequencing **or** `output_loader` still draining. |
| `0x50` | `STATUS` | `done` | `[1]` | R, **read-to-clear** | Sticky. Set once per matmul, when every block is committed **and** the last one has drained. Cleared by the read that observes it. |
| `0x60` | `OUTPUT_SCALE` | `value` | `[15:0]` | W | Q0.16 unsigned rescale factor; actual scale = `value/65536`, range `[0,1)`. |

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
cannot be lost to that collision.

### `OUTPUT_SCALE` and the quantization scheme

The array accumulates at `ACC_WIDTH` (32-bit) so a full contraction of
int8 x int8 products cannot overflow, but `output_buffer` stores int8.
`output_loader` therefore rescales every psum:

```
scaled = saturate_int8( (psum * OUTPUT_SCALE) >>> 16 )
```

Under per-tensor symmetric int8 quantization with scales `w_scale`,
`x_scale` and `y_scale`, the value to program is the single combined factor

```
OUTPUT_SCALE = round( w_scale * x_scale / y_scale * 65536 )
```

which must land in `[0, 65536)` - i.e. the combined scale must be `< 1.0`.

> When `N` spans more than one contraction block, each block's psum is
> rescaled and **saturated to int8 before** being accumulated into the
> previous block's result, which is then saturated again. The accelerator
> does not carry the full-precision 32-bit accumulator across block
> boundaries. Requantization error therefore compounds across contraction
> blocks, and a golden model must reproduce this ordering exactly (see
> `test_multiblock_matmul`).

---

## Programming the accelerator

```
 1. (optional) poll STATUS.busy == 0
 2. write WEIGHT_ROWS    = K
    write WEIGHT_COLS    = N         # must be a multiple of ARRAY_COLS
    write INPUT_MAX_ADDR = M
    write INPUT_COLS     = M
    write OUTPUT_SCALE   = round(w_scale * x_scale / y_scale * 65536)
 3. write weight bytes into WBUF  (layout below)
    write input  bytes into IBUF  (layout below)
 4. write CONTROL.matmul_start = 1  # single pulse, self-clearing
 5. poll STATUS until bit[1] (done) reads 1
    # NB: that read consumes done - keep the value, don't re-read
 6. read result bytes from OBUF   (layout below)
```

Step 4 is the **only** software trigger. From there `weight_ctrl` sequences
the entire matmul itself - per block it prefetches weights, preloads the
input band, commits the weights into the array, streams, and advances to the
next block - with nothing left for software to poke mid-matmul.

Steps 2 and 3 may be done in either order, but both must complete before
step 4, because `matmul_start` latches the dimension registers.

> **Do not skip step 5's poll before reading.** `output_loader` keeps
> writing `output_buffer` for some cycles after the last block is committed;
> `STATUS.done` accounts for that drain, and reading early returns
> partly-stale data.

---

## Buffer layouts

`ARRAY_ROWS`/`ARRAY_COLS` below are the array dimensions (16), and
`OBUF_BANK_DEPTH` = 1024 is the per-bank output RAM depth. All three
buffers have 1024-byte banks. `pitch(c) = roundup(c, 16)` is a row padded
to whole 128-bit words.

### `weight_buffer` and `input_buffer` - contiguous padded images

Software writes each matrix as one row-major image, rows padded to whole
words. `W` is stored exactly as written on paper, with no transpose:

```
addr(k, n) = WBUF_BASE + k*pitch(N) + n     k in [0,K), n in [0,N)
addr(p, m) = IBUF_BASE + p*pitch(M) + m     p in [0,N), m in [0,M)
```

[`rtl/buf_addr_remap.v`](rtl/buf_addr_remap.v) scatters row `r` to bank
`r % 16`, at offset `(r / 16) * pitch` within the bank. It uses the live
`WEIGHT_COLS` and `INPUT_MAX_ADDR` values, so program those registers
**before** writing either buffer.

### `output_buffer` - banked by output row

One bank per PE row. Output row `k` goes to bank `k % ARRAY_ROWS`, with
output-block `k / ARRAY_ROWS` occupying an `M`-sized segment appended within
that bank:

```
addr(k, m) = OBUF_BASE
           + (k % ARRAY_ROWS) * OBUF_BANK_DEPTH     # bank select
           + (k / ARRAY_ROWS) * M                   # output-block segment
           + m
```

Results are **int8**; sign-extend the byte you read back.

Note this means an output row is *not* one contiguous run in the buffer - it
is split one-`ARRAY_ROWS`-th per bank, so reading a full row means striding
across banks.

---

## Capacity limits

All three buffers are fixed-size, so software must check these before
programming a matmul:

| Constraint | Bound |
|---|---|
| `ceil(K / ARRAY_ROWS) * pitch(N) <= 1024` | weight_buffer per-bank depth |
| `ceil(N / ARRAY_COLS) * pitch(M) <= 1024` | input_buffer per-bank depth |
| `ceil(K / ARRAY_ROWS) * M <= 1024` | output_buffer per-bank depth |
| `M <= 1023` | `INPUT_MAX_ADDR` is a 10-bit field |

`K = N = M = 128` fills all three buffers exactly
(`test_128x128x128_matmul`). Beyond that the bounds trade off against each
other: larger `K` or `N` needs a smaller `M`, and vice versa.

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

- **`held_k_blk_idx`** - the write address within the bank:
  `cur_addr = held_k_blk_idx * total_rows + row_count[r]`, i.e. different
  output-row blocks write **disjoint** segments (concatenation, not
  accumulation).
- **`held_n_blk_idx`** - write vs. read-modify-write accumulate:
  `new_val = (held_n_blk_idx == 0) ? scaled_psum : acc_sat`. The first
  contraction block writes; every later one accumulates into what is already
  there, since the contraction may need several passes.

> `committed_first_k_blk` / `held_first_k_blk` are wired through and latched
> but **never read** - the accumulate decision uses `held_n_blk_idx == 0`
> instead. Dead signal.

---

## Buffer RAM IPs

`weight_buffer`, `input_buffer`, and `output_buffer` are placeholder leaf
RAMs (no `rtl/*.v` source checked in - only instantiated) standing in for
Quartus-generated On-Chip Memory IPs.
[`tb/models/buffer_ram_models.v`](tb/models/buffer_ram_models.v) provides
simulation-only behavioral stand-ins with matching names and ports.

All three are **mixed-width simple dual-port** RAMs - independent
`rdaddress`/`wraddress`, each with its own `rden`/`wren`, one cycle of
registered read latency - with the host-facing port 128 bits wide and the
core-facing port byte wide. Port shapes below are per bank:

| Buffer | Instances | Host port (Avalon) | Core port | Total | AVS region |
|---|---|---|---|---|---|
| `weight_buffer` | `ARRAY_ROWS` (16) | **write** 64 x 128, 6-bit | read 1024 x 8, 10-bit (`weight_loader`) | 16 KB | `WBUF_SIZE = 0x4000` |
| `input_buffer` | `ARRAY_ROWS` (16) | **write** 64 x 128, 6-bit | read 1024 x 8, 10-bit (`input_dispatch`) | 16 KB | `IBUF_SIZE = 0x4000` |
| `output_buffer` | `ARRAY_COLS` (16) | **read** 64 x 128, 6-bit | write 1024 x 8, 10-bit (`output_loader`) | 16 KB | `OBUF_SIZE = 0x4000` |

> The `input_buffer` and `output_buffer` IPs grew from 256 B and 128 B per
> bank to 1024 B. Regenerate both On-Chip Memory IPs at the depths above
> before building in Quartus.

The core side stays byte-wide because `weight_loader` and `input_dispatch`
consume one byte per cycle and `output_loader` produces one byte per cycle.
Configure each generated On-Chip Memory with these two port widths; byte
ordering must be little-endian (narrow address `16n+j` = wide bit `8*j`),
which is altsyncram's default.

> `output_buffer`'s single read port is shared - `output_loader`'s
> read-modify-write and the Avalon host both use it, muxed by `busy`. Since
> that port is 128 bits, `output_loader` reads the *word* containing its
> byte and selects the lane with the delayed address; its write-back still
> goes through the byte-wide write port.

Notes:

- The banked wrappers' flat addresses are derived, not hardcoded:
  `*_ADDR_WIDTH = $clog2(16) + 10` (14 bits at 16 banks), split as
  bank-select in the upper bits and per-bank offset in the lower.
- The `_32_bank` in `input_buffer_32_bank` / `output_buffer_32_bank` is
  historical; both are fully parameterized on bank count.
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
| `test_multiblock_matmul` | 2x2 blocks - multi-contraction-block accumulation and multi-output-block addressing |
| `test_partial_row_block_matmul` | Partial last **output-row** block (`K` not a multiple of `ARRAY_ROWS`): `valid_row` masking, `weight_loader` zero-padding, `output_loader` row gating |
| `test_back_to_back_matmul_no_reset` | Two matmuls with no reset between - regression test for stale `STATUS.done`/`busy` |
| `test_128x128x128_matmul` | `W` (128x128) @ `X` (128x128): 64 blocks, fills every buffer bank to its full 1 KB, and is the only test with `M` larger than the 32-entry input FIFOs. Reads the result back in one 2047-beat burst, the 11-bit `avs_burstcount` maximum, which runs past the 1024-word output window and wraps to its start. Logs cycles from `matmul_start` to `STATUS.done`. Takes about a minute in Icarus. |

Test dimensions derive from `ARRAY_ROWS`/`ARRAY_COLS` (top of
`tb/test_accel_top.py`, which must match `accel_top.sv`), so the suite
follows an array resize.

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
- **Stale axis comments.** `rtl/weight_ctrl.v`'s and `rdl/ctrl_reg.rdl`'s
  headers describe `WEIGHT_ROWS` as the contraction dimension. They are
  wrong; see [Data flow and axis convention](#data-flow-and-axis-convention).

See also [`rdl/README.md`](rdl/README.md) for the register file's own notes.

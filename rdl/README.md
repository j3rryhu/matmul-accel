# ctrl_rf register map

Defined in [ctrl_reg.rdl](ctrl_reg.rdl), compiled to `ctrl_rf_rf.sv` via
`python_gen.py` (PeakRDL-verilog). Mapped into `accel_top`'s Avalon-MM space
at `CTRL_BASE = 0x0000_0000`, size `0x100` (see `rtl/accel_top.sv`).

| Offset | Register        | Field                | Bits   | Access | Description |
|--------|-----------------|-----------------------|--------|--------|-------------|
| 0x00   | CONTROL         | `matmul_start`        | [0]    | sw=w, single-pulse | A new matmul is starting: latches `WEIGHT_ROWS`/`WEIGHT_COLS` into `weight_ctrl` and resets its block position to (0,0). |
| 0x04   | WEIGHT_ROWS     | `value`               | [15:0] | sw=w   | Contraction (reduction) dimension size - feeds pe_array's `ARRAY_ROWS` axis, which `pe.v`'s `b_in` cascade sums. Not "rows of the weight matrix on paper" - see note below. |
| 0x08   | WEIGHT_COLS     | `value`               | [15:0] | sw=w   | Output dimension size - feeds `ARRAY_COLS`; values here stay separate (one result per array column). |
| 0x0C   | INPUT_MAX_ADDR  | `value`               | [9:0]  | sw=w   | Max `input_buffer` read address, fed to `input_dispatch.i_max_addr`. |
| 0x10   | INPUT_COLS      | `value`               | [15:0] | sw=w   | Number of columns of the input (activation) matrix. Not yet consumed downstream - see Status below. |
| 0x14   | STATUS          | `out_ready`           | [0]    | sw=r, hw=w | An output result is available to read. |
|        |                 | `out_count`           | [16:1] | sw=r, hw=w | Number of output rows written so far. |
|        |                 | `weight_tile_ready`   | [17]   | sw=r, hw=w | One 32x32 tile's worth of weights has been buffered - ready to prefetch. |
|        |                 | `busy`                | [18]   | sw=r, hw=w | Compute running (loading weights, streaming, or draining). |
|        |                 | `done`                | [19]   | sw=r, hw=w | All inputs drained and all outputs written. |

Bit positions above 0 for multi-bit/later fields are assigned automatically
by PeakRDL in declaration order; re-check `ctrl_rf_rf.sv` after regenerating
if fields are reordered.

**Naming note:** `WEIGHT_ROWS`/`WEIGHT_COLS` name which `pe_array` axis a
dimension loads into, not "row/col of the weight matrix as you'd write it
on paper." If your weight matrix is conventionally `(out_features x
in_features)`, program `WEIGHT_ROWS = in_features` and `WEIGHT_COLS =
out_features`, regardless of which one is literally "rows" in that
convention - `pe.v`'s `b_in` cascade always sums over whatever occupies
`ARRAY_ROWS` (`WEIGHT_ROWS`), so that's always the contraction dimension.
See `rtl/weight_ctrl.v`'s header comment for the full reasoning.

There is no `INPUT_MAX_ROW`/M register: the valid row range for whichever
32x32 block is currently loaded is derived by `weight_ctrl` from
`WEIGHT_ROWS` and the block's position, since the contraction dimension and
activation rows share the same array-row extent.

## Programming sequence

1. Program `WEIGHT_ROWS` / `WEIGHT_COLS`.
2. Write weight bytes into `weight_buffer` (Avalon writes at `WBUF_BASE`)
   and input bytes into `input_buffer` (Avalon writes at `IBUF_BASE`);
   program `INPUT_MAX_ADDR` / `INPUT_COLS` as needed.
3. Pulse `CONTROL.matmul_start`. This is the only step-by-step software
   pulse left - it latches `WEIGHT_ROWS`/`WEIGHT_COLS` into `weight_ctrl`,
   resets its block position to (0,0), and kicks off the first block's
   weight prefetch and input preload. From there `weight_ctrl` sequences
   everything itself: once a block's weights are prefetched and its input
   band is preloaded, it commits the weights, starts streaming, and (per
   its column-major traversal) prefetches/preloads the next block, until
   the whole matmul is done. The access pattern is fixed, so there's
   nothing left for software to trigger mid-matmul.
4. Poll `STATUS.busy` / `STATUS.done`; read results from `output_buffer`
   (Avalon reads at `OBUF_BASE`) once done.

## Status

`CONTROL` (`matmul_start` only), `WEIGHT_ROWS`, `WEIGHT_COLS`,
`INPUT_MAX_ADDR`, and `INPUT_COLS` are wired up in `rtl/accel_top.sv`, and
`weight_ctrl` fully drives `input_dispatch`'s preload/compute-start
handshake (`i_band_start`/`i_start_compute`/`fifos_primed`/`band_done`) -
no software pulses needed once `matmul_start` fires. `INPUT_COLS` feeds
`input_dispatch`'s row-major address stride but isn't otherwise consumed
yet. Still TODO: the byte-counter logic that computes `weight_tile_ready`,
`output_loader` (partial-sum -> `output_buffer`), multi-block output
accumulation (when the contraction dimension needs more than one 32-block
pass), and folding activation streaming/drain busy into
`STATUS.busy`/`STATUS.done`.

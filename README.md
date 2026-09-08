# Buffer RAM IPs

`weight_buffer`, `input_buffer`, and `output_buffer` are placeholder leaf RAMs
(no `rtl/*.v` source checked in - just instantiated) standing in for
Quartus-generated On-Chip Memory IPs. These are the depth/width params those
generated IPs need to match:

| Buffer | Leaf module | Instances | Per-instance depth x width | Address width | Total capacity | Matches AVS region |
|---|---|---|---|---|---|---|
| `weight_buffer` | `weight_buffer` (single instance, `rtl/accel_top.sv:81`) | 1 | 16384 x 8 | 14-bit | 16384 B (16 KB) | `WBUF_SIZE = 0x4000` |
| `input_buffer` | `input_buffer` (via `input_buffer_32_bank.v`) | 32 (one per `ARRAY_ROWS`) | 256 x 8 | 8-bit | 32 x 256 B = 8192 B (8 KB) | `IBUF_SIZE = 0x2000` |
| `output_buffer` | `output_buffer` (via `output_buffer_32_bank.v`) | 32 (one per `ARRAY_COLS`) | 128 x 8 | 7-bit | 32 x 128 B = 4096 B (4 KB) | `OBUF_SIZE = 0x1000` |

All three are byte-wide (8-bit data, matching `PE_DATA_WIDTH`/int8) simple
dual-port RAMs - independent `rdaddress`/`wraddress` each with their own
`rden`/`wren`, one cycle of registered read latency.

Notes:
- `input_buffer`'s Avalon write address is 13 bits total
  (`IBUF_ADDR_WIDTH`), split as bank-select (upper 5 bits, picks 1 of 32
  banks) + 8-bit offset within the bank - see
  `input_buffer_32_bank.v:34-37`.
- `output_buffer`'s per-bank address width is a module parameter
  (`output_buffer_32_bank.v`'s `BANK_ADDR_WIDTH`, default 10) but
  `accel_top` overrides it to 7 (`OBUF_BANK_ADDR_WIDTH`, `rtl/accel_top.sv:124`)
  to land on the 128-entries/bank depth above; use the overridden value,
  not the module default, when sizing the generated IP.
- If wrapping the whole `accel_top` into one Quartus IP (systolic array +
  these three RAMs), each generated On-Chip Memory IP must be a simple
  dual-port RAM (separate read/write ports) at the depth/width above -
  regenerating at a different depth requires updating the corresponding
  `*_SIZE`/`*_ADDR_WIDTH` localparams in `rtl/accel_top.sv` to match.

# Block indexing (`k_blk_idx` / `n_blk_idx`)

Notes on how the weight matrix's block/tile position is tracked and handed
off between `weight_ctrl`, `pe_array`, and `output_loader`.

## Two independent block indices

Tracked in [`rtl/weight_ctrl.v`](rtl/weight_ctrl.v):

| Index | Axis it feeds | What it means |
|---|---|---|
| `k_blk_idx` | `ARRAY_ROWS` (the axis `pe.v`'s `b_in` cascades/sums down) | **contraction** dimension block - a 32-wide slice of `WEIGHT_ROWS` |
| `n_blk_idx` | `ARRAY_COLS` (one accumulated result per column) | **output** dimension block - a 32-wide slice of `WEIGHT_COLS` |

## Traversal order

Column-major over the weight matrix's block grid (`rtl/weight_ctrl.v:24-33`):

- `k_blk_idx` is the **fast/inner** index, `n_blk_idx` the **slow/outer** one.
- For a fixed `n_blk_idx`, every `k_blk_idx` (`0..num_k_blocks-1`) runs and
  gets summed into that output block *before* moving to the next
  `n_blk_idx`.
- Why: only one output block's accumulation needs to stay "live" at a time,
  instead of needing the whole output width resident simultaneously.

`is_last_k_blk` / `is_last_n_blk` / `is_last_block`
(`rtl/weight_ctrl.v:116-118`) gate the advance logic in `WC_WAIT`
(`rtl/weight_ctrl.v:188-197`):

```
if not last block:
    if is_last_k_blk:  k_blk_idx = 0;  n_blk_idx += 1   # wrap k, bump n
    else:               k_blk_idx += 1                   # just advance k
```

## How `output_loader` learns "which block is this"

`weight_ctrl` **commits** a block (asserts `commit_pulse`/`w_load`) once
that block's weights are prefetched and its input band is primed
(`rtl/weight_ctrl.v:180-199`). At that exact commit, it latches two signals
*for `output_loader`*:

```verilog
committed_n_blk_idx   <= n_blk_idx;              // which output block this pass belongs to
committed_first_k_blk <= (k_blk_idx == '0);       // write vs accumulate
```

These are **snapshots at commit time**, not the live `k_blk_idx`/`n_blk_idx`
- because by the time `output_loader` is done draining psums for block N,
`weight_ctrl` has usually already moved on to prefetching block N+1
(`rtl/weight_ctrl.v:82-90`).

`rtl/output_loader.v` then samples these once, at its own `start_edge`
(first rising edge of `a_en_last`, i.e. row 0's `a_en`), and holds them
(`held_n_blk_idx`, `held_first_k_blk`) for that block's entire drain window
(`rtl/output_loader.v:108-135`). The safety argument for why this can't get
clobbered is in the header comment (`rtl/output_loader.v:24-32`): the
*next* commit can't happen until the current block's input band has fully
drained, which is always after `a_en_last` has already risen.

## What those two values are used for

Inside `output_loader`'s per-column logic (`rtl/output_loader.v:152-208`):

- **`held_n_blk_idx`** picks the write *address* within the bank:

  ```verilog
  cur_addr = held_n_blk_idx * total_rows + col_count[c]
  ```

  Bank `c` is reused across every `n_blk_idx` pass; each pass's
  `total_rows`-sized segment is appended head-to-tail after the previous
  `n_blk_idx`'s segment. (Row-major *within* a bank, not globally across the
  buffer.)

- **`held_first_k_blk`** picks write vs. **read-modify-write** accumulate:

  ```verilog
  new_val = held_first_k_blk ? psum_d : (old_val + psum_d)
  ```

  First `k_blk_idx` for a given `n_blk_idx` -> the target slot is
  stale/uninitialized, so just write. Every subsequent `k_blk_idx` (same
  `n_blk_idx`) -> add to what's already there, since the contraction
  dimension may need multiple 32-blocks to fully sum.

See also [`rdl/README.md`](rdl/README.md) for the register-level
programming sequence.

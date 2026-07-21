# Compute Datapath TODO

Context: `pe_array` in [rtl/accel_top.sv](rtl/accel_top.sv) is instantiated but tied to
constants — none of `weight_buffer`/`input_buffer`/`output_buffer` are wired to it yet.
This is the work to close that gap.

## 1. Valid col/row range bypass

- Add a valid-range register pair (row_start/row_end, col_start/col_end, or
  base+size — pick one convention) to the control regfile so software can
  constrain which rows/cols of the 32x32 `pe_array` are considered live for
  the current op.
- Add bypass logic on the `p_out`/`a_out` edges (or at the output_buffer
  write path) that masks/drops any row or column outside the programmed
  range instead of writing it to `output_buffer`.
- Decide bypass semantics: hold last value, force zero, or simply gate
  `wren` for out-of-range lanes — needs to match whatever the writeback
  logic in item 4 expects.

## 2. Input dispatcher + per-row FIFO

- Build a dispatcher that reads activation data out of `input_buffer` and
  feeds `pe_array.a_in` / `a_en`, one lane per row (`ARRAY_ROWS` = 32 lanes).
- Each row needs its own small FIFO so rows can be filled independently and
  staggered into the systolic array without stalling the whole array on a
  single shared read port into `input_buffer`.
- Handle the `input_buffer` single read port: dispatcher must arbitrate
  across the 32 row FIFOs to fill them from the one `rdaddress`/`q` port.
- Replace the tied-off `CONTROL_input_ready_q` in
  [rtl/accel_top.sv:133](rtl/accel_top.sv#L133) with a real ready signal
  once the FIFOs can accept data.

## 3. Weight load + prefetch logic

- Sequencer that reads `weight_buffer` and drives `pe_array.w_in`/`w_en`
  (flattened, `row*ARRAY_COLS+col` indexed per
  [rtl/pe_array.v:70](rtl/pe_array.v#L70)) to prefetch the next weight tile
  into each PE's shadow register while the current tile is still computing.
- Drive `w_load` as a single global pulse (per the comment in
  [rtl/pe_array.v:8-9](rtl/pe_array.v#L8-L9)) once the whole 32x32 tile has
  been prefetched — needs a counter to know prefetch is complete.
- Replace the tied-off `CONTROL_weight_ready_q` in
  [rtl/accel_top.sv:134](rtl/accel_top.sv#L134) similarly.

## 4. Block-done counter + ctrl regfile status bit

- Add a counter that tracks progress of one 32x32 block compute (e.g. count
  activation cycles in / drain cycles through the systolic pipeline) and
  flags when a block is `done` vs `ongoing`.
- Wire this into a new `block_done` (or similar) status field in the ctrl
  regfile — add it to [rdl/ctrl_reg.rdl](rdl/ctrl_reg.rdl) and regenerate
  `ctrl_rf_rf.sv` (don't hand-edit the generated file directly, see
  `STATUS_out_ready`/`STATUS_out_count` in
  [rdl/ctrl_reg.sv/ctrl_rf_rf.sv:196-229](rdl/ctrl_reg.sv/ctrl_rf_rf.sv#L196-L229)
  for the existing pattern of a HW-driven status field).
- This counter is likely the shared timing reference that items 1-3 all key
  off (when to sample valid-range bypass, when dispatcher/prefetch can
  start the next tile, when output_buffer writeback is valid).

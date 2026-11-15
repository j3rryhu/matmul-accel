# Standalone int8 GELU RTL

`gelu_int8.sv` is a 256x8 ROM with a one-entry ready/valid output buffer.
An input is accepted on a rising edge when in_valid && in_ready. Its result is
available after that edge. Output transfers on out_valid && out_ready; output
data and valid remain stable while stalled. Peak throughput is one byte per
clock. Active-low reset discards pending output. It does not reset the ROM.

Run from the matmul-soc repository root:

```powershell
vsim -c -do activation/rtl/run_modelsim.do
```

The testbench checks every input byte against the Python-generated HEX, then
checks bubbles, downstream backpressure, and reset with a pending output.
Python tests separately verify the table's GELU arithmetic. The default LUT
uses demonstration scales of 1/32, not accepted TinyViT deployment scales.
Override LUT_FILE for a calibrated table. Add the chosen HEX as a Quartus
project file and verify ROM initialization and synthesis reports during integration.

## Implemented same-scale bias boundary

`bias_gelu_int8.sv` is LANES wide (default 16 x int8 = 128 bits). Lane l uses
bits [8l+7:8l] of `matmul_data`, `bias_data` and `out_data`. Per lane it accepts
paired signed int8 matmul and bias operands, explicitly sign-extends both to 9
bits, adds, clips to [-128,127], and feeds that lane's GELU ROM.
Both operands MUST have the same scale as the LUT input (zero points zero).
This is the user-selected contract; deployment scales still need calibration.
No bias storage or channel selection is included: upstream must supply the
correct bias per lane and hold operands, `in_mask` and in_valid stable while
in_ready is low. All lanes share one ready/valid handshake with the same
latency and one-entry buffering as gelu_int8: peak one 16-byte word per clock.

`in_mask[l]` selects per-lane activation. A set lane outputs GELU(sat(matmul+bias));
a clear lane passes its raw matmul byte through unchanged (its bias is
don't-care, X is allowed). `out_mask` is the accepted `in_mask`, aligned with
`out_data`. Reset clears `out_valid`, `out_mask` and `out_data`. Each lane has
its own 256x8 ROM with a registered, clock-enabled, non-reset read so it can
infer block RAM; the raw byte has its own register and the lane select is after
both registers. Check the Quartus fitter report for 16 ROMs.

`tb_bias_gelu_int8.sv` makes every lane walk all 65,536 operand pairs (full
mask), then checks raw pass-through with all-zero, one-hot, inverted one-hot and
random masks (X bias on bypassed lanes), plus bubbles, multi-cycle backpressure
and reset with a stalled
output. The simulation script runs both standalone GELU and bias/GELU tests.

Examples of values BEFORE GELU: 120+20 -> 127; -120-20 -> -128; 50-20 -> 30.

The confirmed order is matmul int8 -> bias -> GELU. This module starts AFTER
bias addition and conversion to the GELU input scale. The wrapper implements
only the selected same-scale int8 case; different scales require rescaling.

Mathematical reference (zero points zero):

```
x = matmul_int8 * matmul_output_scale + bias_real
gelu_input_int8 = clip(round(x / gelu_input_scale), -128, 127)
gelu_output_int8 = ROM[gelu_input_int8 & 255]
```

If bias is already quantized to the SAME scale as matmul output and GELU input,
sign-extend both int8 values, add in at least signed 9 bits, and saturate ONCE
to int8. Do not allow int8 wraparound: 120 + 20 must clamp to 127, not -116.
Other bias widths require a sufficiently wide adder and documented rescaling.
Do not assume that the matmul Q0.16 multiplier is its real-value output scale.

Next integration decisions: matmul output scale, bias units/width, GELU input
and output scales, rounding, bias channel ordering, and stream handshake.
A 128-bit interface carries 16 bytes; `bias_gelu_int8` processes all 16 lanes
per cycle (`gelu_int8` remains the single-byte core).

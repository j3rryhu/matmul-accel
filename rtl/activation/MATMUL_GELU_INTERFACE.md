# Matmul-to-GELU Interface: Evidence and Open Decisions

These are source-code observations, not an interface agreement with the matmul
owner. No teammate was contacted during this experiment.

## Current source evidence

The adjacent `matmul-soc` repository contains:

- `matmul_accel/accel_top.sv`: default operand width 8 and accumulator width 32;
  `output_scale` is an unsigned Q0.16 register.
- `matmul_accel/output_loader.v`: partial sums are multiplied by output_scale,
  arithmetic-shifted right by 16, and saturated to signed int8. The stored
  previous int8 value and a rescaled partial sum are added with another saturation.
- `test_program/matmul_test_32.c`: the CPU golden model explicitly reproduces
  per-block requantization, saturation, and int8 accumulation. Its scale formula is
  `OUTPUT_SCALE = round(weight_scale * input_scale / output_real_scale * 65536)`.
- A text search of matmul_accel Verilog/SystemVerilog found no `bias` reference.
  This does not establish where bias is handled elsewhere in the system.

Arithmetic right shift is not the ties-to-even rule used by this GELU experiment.
Repeated block rounding and saturation can also differ from a full wide sum
followed by one requantization, especially when partial terms cancel.

## Decisions to confirm with the matmul owner

1. Is the delivered value a raw accumulator or an already requantized int8 value?
2. What real-value scale and zero point does the delivered value use? A Q0.16
   rescale multiplier is not itself the real-value GELU input scale.
3. Are weight scales per tensor or per output channel? Can the hardware support
   the required multiplier granularity?
4. Is the complete contraction sum retained in a wide accumulator before
   requantization, or is the current per-block int8 accumulation intentional?
5. Where is bias added, in what units, and how is bias rounded? It must be included
   before GELU and should not be added repeatedly for each partial block.
6. What rounding and saturation rules are required at the matmul-to-GELU boundary?
7. What scale does the next Linear layer expect for the GELU output?

For a symmetric scheme with a full wide accumulator, a possible mathematical
contract is:

```text
accumulator_scale = input_scale * weight_scale
q_gelu_input = clip_int8(round((accumulator * accumulator_scale + bias_real)
                              / gelu_input_scale))
gelu_output_real = LUT[q_gelu_input & 255] * gelu_output_scale
```

This is a proposed reference contract, not a description of the current hardware.
Per-channel weight scales require corresponding per-channel rescaling. The
software model-level experiment includes floating-point bias and quantizes the
completed floating-point Linear 1 output; it does not simulate the existing
integer matmul pipeline.

After these decisions are agreed, model the actual integer boundary in software
and compare it against the reference data before implementing or connecting RTL.

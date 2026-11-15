# HPS matmul + bias/GELU test

Build/program the newly regenerated FPGA system first. Copy these two files to
the same directory on the board:

- `test_program/matmul_gelu_test.c`
- `activation/example/gelu_int8.hex` (must be the table in that FPGA build)

```bash
gcc -std=gnu99 -O2 -Wall -o matmul_gelu_test matmul_gelu_test.c -lrt
./matmul_gelu_test --self-test gelu_int8.hex
./matmul_gelu_test gelu_int8.hex
```

Hardware execution requires root, enabled HPS-FPGA bridges and an idle/reset
accelerator. Uses only 32-bit PIO, no NEON/DMA/staging DDR. Register stride is
16 bytes, matching current RTL. Bias and enable offsets are NOT multiplied.

The test runs 32x32 identity matmul, validates every raw byte, loads 32 distinct
row biases (-16..15), enables readout processing, and compares all 1024 outputs
to signed saturating addition followed by the supplied table. It then disables
processing and checks raw RAM contents are unchanged. This validates both row
segments, all int8 input codes, positive/negative bias clipping and 32-bit
accesses across all 16 wide-bus lanes. It does not benchmark, validate real
TinyViT scales, or exercise Avalon burst reads/DMA.

The expected result is based on the FPGA HEX, not on GELU evaluated in C, to
avoid platform floating-point rounding differences. Python/RTL tests separately
validate table arithmetic. The offline self-test checks only C reference logic
and file parsing; it is not evidence that hardware passed.

Expected final line: `PASS: HPS matmul + bias + GELU readout integration`.
An old bitstream should fail Bias + GELU even if raw identity passes. New config
registers are write-only, so the test verifies their effects rather than reading
them. The enable bit is disabled on idle exit; loaded bias values remain.

## Board result reported on 2026-10-11

The user compiled with the command above and reported:

```text
PASS: LUT parsing and 65536 bias reference combinations (no FPGA access)
Initial STATUS = 0x00000000
Raw identity matmul: PASS (0/1024 mismatches)
Bias + GELU: PASS (0/1024 mismatches)
Bypass preserves raw output: PASS (0/1024 mismatches)
PASS: HPS matmul + bias + GELU readout integration
```

This validates board integration with the example LUT, not real-model accuracy
or timing/resource signoff. The bitstream hash was not provided.

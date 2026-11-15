# Matmul readout integration

Resource optimization: row mapping and bias read are shared by all bytes of a
wide word; the divider is bank-address-width unsigned, not one integer divider
per lane. Addition is explicitly signed 9-bit. Enabled processing requires M
to be a multiple of DATA_WIDTH/8 (16 for the current 128-bit interface), so a
word cannot straddle output-row segments. Unsupported geometry bypasses the
activation; software must enforce the restriction. The existing M=32 test meets
it. Actual fit/timing improvement must be measured by recompiling Quartus.

New logic is isolated in `matmul_gelu_readout.v`. Existing edits are limited to
one instance and the output read mux in `accel_top.sv`, plus two component
fileset entries in `matmul_accel_hw.tcl`. The output loader and RAM are unchanged.

Why readout: the output RAM stores partial-block sums as well as final results.
Applying nonlinear GELU on intermediate writes would corrupt later accumulation.
The host must wait for matmul done, then read the final output. The same output
addresses now return bias/GELU when enabled, raw matmul bytes when disabled.
No additional read cycle is added. Processing is combinational across every
byte of the wide RAM read; registered address tracking matches RAM latency.
Quartus timing/resource checks are still required (parallel ROM lookups and
runtime division for row mapping may be expensive). This is not a streaming
connection to the next Linear layer and does not write activated values into RAM.

## New byte-addressed host writes

All offsets are relative to accelerator base `0xC0000000`, without --reg16 scaling.

| Offset | Write behavior |
|---|---|
| 0x8000..0x80ff | 256 signed int8 biases, indexed by output matrix row |
| 0x8100 | Bit 0 enables bias + GELU; zero bypasses both |

Reset disables processing and clears biases. Configuration writes are ignored
while matmul busy. These registers are write-only; reads are not implemented.
Do not change dimensions, biases or enable state during output reads. Bias is
constant across each output row in the existing W[K,N] @ X[N,M] layout.
Upstream software must transpose/map its model tensors to that layout.

Use 32-bit aligned host writes (Avalon byteenable selects the correct wide lane):

```c
/* accel is your existing mmap pointer; bias contains K same-scale int8 values. */
*(volatile uint32_t *)(accel + 0x8100) = 0; /* disable, while idle */
for (int row = 0; row < K; row += 4) {
    uint32_t packed = 0;
    for (int j = 0; j < 4 && row+j < K; j++)
        packed |= (uint32_t)(uint8_t)bias[row+j] << (8*j);
    *(volatile uint32_t *)(accel + 0x8000 + row) = packed;
}
__sync_synchronize();
*(volatile uint32_t *)(accel + 0x8100) = 1;
__sync_synchronize();
/* Configure/start matmul as usual, wait for done, then read OBUF. */
```

Default LUT input/output scales are both 1/32. Bias and matmul bytes MUST have
that input scale for this table. These remain demonstration scales, not selected
TinyViT deployment parameters. A different table requires rebuilding the FPGA.

Expected host result per element:
`table[clip(matmul_golden[k,m] + bias[k], -128, 127) & 255]`.
The old identity test expects raw matmul, so it must run with activation disabled.
For activation testing first read raw output with enable=0, then enable=1 while
idle and reread identical addresses; compare every byte against this reference.
Disable again to verify the original raw data has not changed.

## Build and verification

Regenerate Platform Designer synthesis HDL after refreshing the component.
Check that the generated fileset includes the new Verilog and LUT HEX and that
the ROM initialization path resolves. The current root-relative LUT path assumes
Quartus is launched from the project root. Compile Quartus and inspect timing
before programming. Existing generated Qsys HDL was not regenerated here.

Run `vsim -c -do activation/rtl/run_modelsim.do` from the project root.
Tests cover standalone GELU, all 65,536 int8 bias pairs, and 1,024 readout bytes
over both 16-row segments with distinct biases, byte-enabled writes, bypass and
reset. The readout test models synchronous RAM; it is not a full Avalon/HPS or
full matmul simulation. Board and full-system verification remain necessary.

# GELU Python reference and int8 LUT

This prototype processes one signed int8 element at a time. It uses exact
erf-based GELU, symmetric quantization (zero points are zero), nearest rounding
with ties to even, and saturation to [-128, 127]. No third-party packages needed.

```
x_real = input_int8 * input_scale
y_real = GELU(x_real)
output_int8 = clamp(round(y_real / output_scale), -128, 127)
```

From the repository root:

```bash
python activation/gelu_lut.py --input-scale 0.03125 --output-scale 0.03125 --out-dir activation/example
python -m unittest discover -s activation -p "test_*.py"
```

The example scales represent an input range [-4, 3.96875]. They are demonstration
values, not calibrated TinyViT scales. Replace them with measured scales for the
chosen layer. The matmul OUTPUT_SCALE register is a combined multiplier and is
not itself the real-value scale needed by this LUT.

Outputs:

- `gelu_int8.hex`: 256 output bytes in ROM address order; suitable for `$readmemh`.
- `gelu_vectors.csv`: all signed input codes and expected outputs for hardware tests.
- `gelu_metadata.json`: scales, rounding, addressing, and error summary.

ROM address is `input_int8 & 0xff`: addresses 0..127 represent 0..127;
addresses 128..255 represent -128..-1. Interpret output bytes as signed int8.
For the example scales, input -32 represents -1 and maps to -5 (real -0.15625);
input 32 represents 1 and maps to 27 (real 0.84375).

Add bias and finish accumulation before applying GELU. This LUT assumes its
input has already been quantized to the documented input scale. It cannot
recover information lost during matmul's partial-sum saturation. It does not
modify or connect to the existing FPGA datapath. A 128-bit stream would contain
16 independent input elements; ROM replication or serialization is a later RTL
design choice. Validate layer outputs and model predictions before integration.

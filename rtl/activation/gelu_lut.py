"""Exact GELU reference and symmetric int8 LUT, using only Python's stdlib."""

import argparse
import csv
import json
import math
from pathlib import Path


def gelu(x):
    """GELU with the exact erf formulation (no tanh approximation)."""
    return 0.5 * x * (1.0 + math.erf(x / math.sqrt(2.0)))


def validate_scale(scale):
    if not math.isfinite(scale) or scale <= 0:
        raise ValueError("scale must be finite and positive")


def quantize_int8(value, scale):
    """Round to nearest, ties to even; saturate to [-128, 127]."""
    validate_scale(scale)
    return max(-128, min(127, round(value / scale)))


def make_lut(input_scale, output_scale):
    """Address is the raw two's-complement byte, NOT q + 128."""
    validate_scale(input_scale)
    validate_scale(output_scale)
    return [quantize_int8(gelu((a if a < 128 else a - 256) * input_scale),
                          output_scale) for a in range(256)]


def lookup(q, lut):
    if not isinstance(q, int) or not -128 <= q <= 127:
        raise ValueError("input must be an integer in [-128, 127]")
    return lut[q & 0xff]


def export(input_scale, output_scale, destination):
    lut = make_lut(input_scale, output_scale)
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    # One raw output byte per address, ready for an 8-bit $readmemh ROM.
    (destination / "gelu_int8.hex").write_text(
        "".join(f"{v & 0xff:02x}\n" for v in lut), encoding="ascii")
    rows = []
    for q in range(-128, 128):
        x = q * input_scale
        reference = gelu(x)
        result = lookup(q, lut)
        rows.append([q & 0xff, q, x, reference, result,
                     result * output_scale, abs(result * output_scale - reference)])
    with (destination / "gelu_vectors.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["address", "input_int8", "input_real", "gelu_reference",
                         "output_int8", "output_real", "absolute_error"])
        writer.writerows(rows)
    errors = [r[-1] for r in rows]
    clipped = sum(not -128 <= round(r[3] / output_scale) <= 127 for r in rows)
    metadata = {
        "function": "0.5*x*(1+erf(x/sqrt(2)))",
        "input_scale": input_scale, "output_scale": output_scale,
        "input_zero_point": 0, "output_zero_point": 0,
        "rounding": "nearest_ties_to_even", "entries": 256,
        "addressing": "input_int8 & 0xff; output byte is two's complement",
        "max_absolute_error": max(errors),
        "mean_absolute_error": sum(errors) / 256,
        "saturated_reference_count": clipped,
    }
    (destination / "gelu_metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    return metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-scale", required=True, type=float)
    parser.add_argument("--output-scale", required=True, type=float)
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()
    try:
        metadata = export(args.input_scale, args.output_scale, args.out_dir)
    except ValueError as error:
        parser.error(str(error))
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()

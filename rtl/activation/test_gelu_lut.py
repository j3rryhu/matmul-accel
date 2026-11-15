import math
import tempfile
import unittest
from pathlib import Path

from gelu_lut import export, gelu, lookup, make_lut, quantize_int8


class GeluTests(unittest.TestCase):
    def test_known_values_and_symmetry(self):
        self.assertEqual(gelu(0), 0)
        self.assertAlmostEqual(gelu(1), 0.8413447460685429)
        self.assertAlmostEqual(gelu(-1), -0.15865525393145707)
        for x in (0.1, 0.5, 1, 2, 4):
            self.assertAlmostEqual(gelu(x) - gelu(-x), x)

    def test_rounding_and_clipping(self):
        self.assertEqual([quantize_int8(x, 1) for x in (0.5, 1.5, -0.5, -1.5)],
                         [0, 2, 0, -2])
        self.assertEqual(quantize_int8(999, 1), 127)
        self.assertEqual(quantize_int8(-999, 1), -128)

    def test_all_codes_and_unsaturated_error_bound(self):
        si, so = 0.03125, 0.03125
        lut = make_lut(si, so)
        for q in range(-128, 128):
            y = gelu(q * si)
            expected = max(-128, min(127, round(y / so)))
            self.assertEqual(lookup(q, lut), expected)
            self.assertLessEqual(abs(lookup(q, lut) * so - y), so / 2 + 1e-12)
        self.assertEqual(lookup(-128, lut), lut[128])
        self.assertEqual(lookup(-1, lut), lut[255])

    def test_export_raw_bytes(self):
        with tempfile.TemporaryDirectory() as directory:
            export(0.03125, 0.03125, directory)
            words = (Path(directory) / "gelu_int8.hex").read_text().splitlines()
            self.assertEqual(len(words), 256)
            decoded = [v if v < 128 else v - 256 for v in map(lambda w: int(w, 16), words)]
            self.assertEqual(decoded, make_lut(0.03125, 0.03125))

    def test_invalid_scales(self):
        for scale in (0, -1, math.inf, math.nan):
            with self.assertRaises(ValueError):
                make_lut(scale, 1)


if __name__ == "__main__":
    unittest.main()

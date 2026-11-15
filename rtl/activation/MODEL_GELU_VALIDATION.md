# Model-Level GELU Validation

The confirmed FPGA target is **TinyViT 21M**, not the 5M model used for the
earlier activation-only experiment. This experiment replaces one MLP GELU in
the actual 21M model and compares final prediction classes and logits.

Run from the TinyVit-on-FPGA repository root:

```powershell
python tools/compare_gelu_model.py
```

The default local checkpoint is
`model_checkpoints/tiny_vit_21m_22kto1k_distill.pth`, downloaded from the official
model-zoo URL referenced by `tiny-vit/models/tiny_vit.py`. The script itself
does not download checkpoints. The output directory is `gelu_model_validation_21m`.

For a larger dataset and explicit ground-truth labels:

```powershell
python tools/compare_gelu_model.py --images "C:/datasets/example" --labels-json "C:/datasets/labels.json" --out gelu_model_validation_large
```

The image folder must directly contain the images. Optional labels are a JSON
object mapping each image filename to a zero-based ImageNet class index, for
example `{"image001.jpg": 1}`. Without explicit labels, the report measures
agreement with the floating-point model, not ground-truth classification accuracy.

## Experiment

Original:

```text
float32 Linear 1 + bias -> PyTorch GELU -> float32 Linear 2 -> final logits
```

Replacement:

```text
float32 Linear 1 + bias -> int8 quantization -> LUT -> real-value reconstruction
                       -> float32 Linear 2 -> final logits
```

Only `layers.1.blocks.0.mlp.act` is replaced. All other model operations remain
floating point. RGB preprocessing uses bicubic short-side resize to 256,
center crop to 224, and ImageNet normalization. The original model is restored
after each option and its original logits are checked for exact reproduction.

Images are sorted by filename. The first 70% are used for calibration and the
rest for validation. Duplicate image contents are rejected. Default data has
only 10 images, so the 7/3 split is preliminary, not a representative benchmark.

Four options are evaluated:

- Original LUT: existing input/output scale of 0.03125.
- Minimum calibration MAE: the earlier selection method, recalibrated for 21M.
- Minimum calibration RMSE: penalizes large errors more strongly, using the same grid.
- Calibration range coverage: input/output scales cover the measured calibration
  ranges and outputs at the observed quantized calibration inputs. This does not
  guarantee coverage of unseen validation data or every unused LUT entry.

Grid candidates use the calibration absolute-value percentiles
99/99.5/99.9/100 divided by 127, plus the original scale. No held-out values are
used to select scales. The report compares MAE, RMSE, maximum error, positive
input clipping, positive reference-output clipping, actual LUT saturation,
top-1 agreement, top-5 set agreement, and final logit differences.

Positive reference-output clipping is the fraction of **positive reference
outputs** whose rounded code exceeds 127 at the chosen output scale. Actual LUT
output saturation is counted separately using outputs at the quantized inputs.
These metrics have different meanings and are not interchangeable.

## Artifacts and scope

- `RESULTS.md`: readable English results and limitations.
- `report.json`: complete calibration/validation metrics, scales, checkpoint and
  image hashes, preprocessing, and framework versions.
- `predictions.csv`: original/replacement classes and logit errors for every image.
- `logits.npz`: full baseline and replacement logits, with image order.
- `*_reference.npz`: 21M Linear 1 outputs including bias and PyTorch GELU outputs.
- `luts/`: software comparison HEX/CSV tables; no original hardware files are changed.

The LUT module is checked against the software convention for all 256 input
codes, representative clipping/tie cases, and real activations. Quantization uses
float64 division and ties-to-even rounding; reconstruction is cast back to float32.
Future fixed-point hardware must be verified against this convention or explicitly
document a different one.

No acceptance threshold has been supplied. Even unchanged predictions on the
three held-out images do not prove deployment readiness, full-model int8 accuracy,
or the behavior of several LUT replacements together.

See `MATMUL_GELU_INTERFACE.md` for the current interface evidence and questions
to resolve with the matmul owner before hardware integration.

# TinyViT 21M: Model-Level GELU Validation

Target: `tiny_vit_21m_224`, `22kto1k_distill`, CPU float32. Replaced only `layers.1.blocks.0.mlp.act`.

Dataset: 10 local images; 7 calibration and 3 held out. Scales were fixed before held-out evaluation.

## Held-out activation errors

| LUT option | Input scale | Output scale | MAE | RMSE | Max error | Positive reference output clipping |
|---|---:|---:|---:|---:|---:|---:|
| original | 0.03125 | 0.03125 | 0.00851507 | 0.00999788 | 0.392292 | 0.0015% |
| minimum_calibration_mae | 0.025640268 | 0.0141573309 | 0.00455291 | 0.0238066 | 2.56306 | 0.9427% |
| minimum_calibration_rmse | 0.025640268 | 0.0331772294 | 0.00768286 | 0.00976887 | 1.10967 | 0.0006% |
| calibration_range_covering | 0.0591312296 | 0.0331772327 | 0.00855735 | 0.010941 | 0.147533 | 0.0006% |

Positive reference output clipping means the reference output's rounded code exceeds 127; its denominator is the number of positive reference outputs. Actual LUT saturation counts and positive input clipping are separately recorded in report.json.

## Held-out model predictions and logits

| LUT option | Top-1 agreement | Top-5 set agreement | Logit MAE | Logit RMSE | Max logit error |
|---|---:|---:|---:|---:|---:|
| original | 3/3 | 2/3 | 0.0278889 | 0.0412169 | 0.235259 |
| minimum_calibration_mae | 3/3 | 2/3 | 0.0265797 | 0.0358657 | 0.243121 |
| minimum_calibration_rmse | 3/3 | 3/3 | 0.0124043 | 0.0166008 | 0.0788904 |
| calibration_range_covering | 3/3 | 3/3 | 0.0104576 | 0.0140319 | 0.10296 |

## Per-image held-out top-1 predictions

| Image | Original PyTorch model | LUT option | Modified prediction | Agrees |
|---|---|---|---|---|
| n02085620_Chihuahua.JPEG | Chihuahua | original | Chihuahua | True |
| n02086079_Pekinese.JPEG | Pekinese | original | Pekinese | True |
| n02090379_redbone.JPEG | vizsla | original | vizsla | True |
| n02085620_Chihuahua.JPEG | Chihuahua | minimum_calibration_mae | Chihuahua | True |
| n02086079_Pekinese.JPEG | Pekinese | minimum_calibration_mae | Pekinese | True |
| n02090379_redbone.JPEG | vizsla | minimum_calibration_mae | vizsla | True |
| n02085620_Chihuahua.JPEG | Chihuahua | minimum_calibration_rmse | Chihuahua | True |
| n02086079_Pekinese.JPEG | Pekinese | minimum_calibration_rmse | Pekinese | True |
| n02090379_redbone.JPEG | vizsla | minimum_calibration_rmse | vizsla | True |
| n02085620_Chihuahua.JPEG | Chihuahua | calibration_range_covering | Chihuahua | True |
| n02086079_Pekinese.JPEG | Pekinese | calibration_range_covering | Pekinese | True |
| n02090379_redbone.JPEG | vizsla | calibration_range_covering | vizsla | True |

## Interpretation and limits

All options are comparisons, not accepted deployment scales. The original LUT is retained as a reference; the previous 5M calibrated candidates were not reused. MAE/RMSE candidates and range coverage were recalibrated using only the 21M calibration activations.

In this run, every option preserves the baseline top-1 class on all three held-out images. The calibration-range option reduces maximum activation error from 0.392292 to 0.147533, has no held-out input clipping, and preserves all three top-5 class sets. However, two positive reference outputs still exceed its output range: covering calibration values does not guarantee coverage of unseen values. The minimum-RMSE option has the lowest maximum logit error of these comparisons; the range-covering option has the lowest mean logit error. No option is declared accepted from this small sample.

The lowest-MAE option still has a much larger maximum activation error (2.56306) and positive reference-output clipping fraction (0.9427%). Preserved top-1 classes on three images do not remove that concern.

The original model predicts `vizsla` for the file named `n02090379_redbone.JPEG`, and all replacements retain it. This illustrates why agreement with the baseline must not be described as correct classification. No authoritative ground-truth label mapping was supplied for this evaluation.

- Only 10 local example images in the default run; not a representative accuracy benchmark
- One MLP GELU replaced; matmul, attention, other activations and other weights remain floating point
- Prediction agreement is not ground-truth accuracy; explicit labels are required for accuracy
- Integer matmul scales, bias representation, accumulator width and requantization remain unconfirmed
- No numerical or task accuracy acceptance thresholds supplied; no hardware integration approval implied

The LUT module matched software quantization checks for all 256 codes, clipping/tie probes, and real inputs. Restoring the original GELU reproduced the original logits exactly.

Artifacts: report.json includes checkpoint/source/image hashes and both split metrics; predictions.csv contains per-image classes and logit differences; logits.npz contains complete baseline/modified logits; image reference NPZ files contain paired Linear 1 and PyTorch GELU outputs; luts/ contains generated software comparison tables.

Next: define acceptance criteria and a representative labeled dataset; confirm matmul accumulator scale, accumulator width, bias addition/quantization and output requantization with the matmul owner. This experiment does not connect a LUT to FPGA or implement RTL.

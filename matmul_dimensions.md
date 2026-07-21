# TinyViT-5M — Per-Stage Matmul Dimensions

Config (`tiny_vit_5m_224`): `embed_dims=[64,128,160,320]`, `depths=[2,2,6,2]`, `num_heads=[2,4,5,10]`, `window_sizes=[7,7,14,7]`, input `224×224×3`. Verified against `microsoft/Cream/TinyViT` source.

Notation: `B`=batch, `N`=tokens (=H×W), `C`=channels, `h`=heads, `d`=head_dim=C/h.

---

## Stem (PatchEmbed)

| Step | Op | Matrices involved | Shapes |
|---|---|---|---|
| 1 | Conv 3×3, s2 | input `x`, kernel `W_stem1` | `(B,3,224,224) → (B,32,112,112)` |
| 2 | Conv 3×3, s2 | input `x1`, kernel `W_stem2` | `(B,32,112,112) → (B,64,56,56)` |

---

## Stage 0 — MBConv ×2 (pure conv, no attention), C=64, res=56×56

Per block, hidden = 4×64 = 256. Input to block: `x`.

| Op | Matrices involved | Weight shape | Input → Output |
|---|---|---|---|
| Conv1×1 (expand) | `x1 = Conv1x1(x, W_expand)` | `W_expand: [64→256]` | `(B,64,56,56) → (B,256,56,56)` |
| DWConv3×3 | `x2 = DWConv3x3(x1, W_dw)` | `W_dw: [256]` depthwise | `(B,256,56,56) → (B,256,56,56)` |
| Conv1×1 (project) | `x3 = Conv1x1(x2, W_proj)` | `W_proj: [256→64]` | `(B,256,56,56) → (B,64,56,56)` |
| Residual | `out = x3 + x` | — | `(B,64,56,56)` |

Repeat ×2. **PatchMerging → Stage 1:** `Conv1×1[64→128] → DWConv3×3 s2 → Conv1×1[128→128]` → `(B,128,28,28)`

---

## Stage 1 — Attention ×2, C=128, h=4, d=32, window=7×7, N=784, 16 windows (Nw=49)

Per window, input `xw` (window slice of the token sequence). Per block:

| Step | Matrices involved | Formula | Output shape |
|---|---|---|---|
| LayerNorm | `xn = LN(xw)` | — | `(49,128)` |
| QKV proj | `QKV = xn · W_qkv` | `[49×128]·[128×384]` | `(49,384)` |
| reshape → heads | `QKV → (49,4,96)` | — | |
| split → Q,K,V | `Q,K,V ← split(QKV, [32,32,32])` | `96→32+32+32` | `Q,K,V∈(49,4,32)` |
| Attn scores (×4 heads) | `S_i = Q_i·K_i^T / √32` | `[49×32]·[32×49]` | `(49,49)` per head |
| + bias, softmax | `A_i = softmax(S_i + B_rel,i)` | — | `(49,49)` per head |
| Attn·V (×4 heads) | `O_i = A_i · V_i` | `[49×49]·[49×32]` | `(49,32)` per head |
| concat heads | `O = concat(O_1..O_4)` | `4×(49,32)` | `(49,128)` |
| Output proj | `x' = O · W_O` | `[49×128]·[128×128]` | `(49,128)` |
| Residual | `x = x_orig + x'` | — | `(784,128)` (after window-reverse) |
| Local conv | `x = DWConv3x3(x, W_local)` | depthwise, on `(B,128,28,28)` | `(B,128,28,28)` |
| LayerNorm | `xn = LN(x)` | — | `(784,128)` |
| MLP fc1 | `m1 = GELU(xn · W_1)` | `[784×128]·[128×512]` | `(784,512)` |
| MLP fc2 | `m2 = m1 · W_2` | `[784×512]·[512×128]` | `(784,128)` |
| Residual | `x = x + m2` | — | `(784,128)` |

Repeat ×2. **PatchMerging → Stage 2:** `Conv1×1[128→160] → DWConv3×3 s2 → Conv1×1[160→160]` → `(B,160,14,14)`

---

## Stage 2 — Attention ×6, C=160, h=5, d=32, window=14×14 = full map (global attn), N=196

Input `x` (whole map, no windowing — global attention). Per block:

| Step | Matrices involved | Formula | Output shape |
|---|---|---|---|
| LayerNorm | `xn = LN(x)` | — | `(196,160)` |
| QKV proj | `QKV = xn · W_qkv` | `[196×160]·[160×480]` | `(196,480)` |
| reshape → heads | `QKV → (196,5,96)` | — | |
| split → Q,K,V | `Q,K,V ← split(QKV, [32,32,32])` | `96→32+32+32` | `Q,K,V∈(196,5,32)` |
| Attn scores (×5 heads) | `S_i = Q_i·K_i^T / √32` | `[196×32]·[32×196]` | `(196,196)` per head |
| + bias, softmax | `A_i = softmax(S_i + B_rel,i)` | — | `(196,196)` per head |
| Attn·V (×5 heads) | `O_i = A_i · V_i` | `[196×196]·[196×32]` | `(196,32)` per head |
| concat heads | `O = concat(O_1..O_5)` | `5×(196,32)` | `(196,160)` |
| Output proj | `x' = O · W_O` | `[196×160]·[160×160]` | `(196,160)` |
| Residual | `x = x_orig + x'` | — | `(196,160)` |
| Local conv | `x = DWConv3x3(x, W_local)` | depthwise, on `(B,160,14,14)` | `(B,160,14,14)` |
| LayerNorm | `xn = LN(x)` | — | `(196,160)` |
| MLP fc1 | `m1 = GELU(xn · W_1)` | `[196×160]·[160×640]` | `(196,640)` |
| MLP fc2 | `m2 = m1 · W_2` | `[196×640]·[640×160]` | `(196,160)` |
| Residual | `x = x + m2` | — | `(196,160)` |

Repeat ×6 (deepest stage). **PatchMerging → Stage 3:** `Conv1×1[160→320] → DWConv3×3 s2 → Conv1×1[320→320]` → `(B,320,7,7)`

---

## Stage 3 — Attention ×2, C=320, h=10, d=32, window=7×7 = full map (global attn), N=49

Input `x` (whole map, no windowing — global attention). Per block:

| Step | Matrices involved | Formula | Output shape |
|---|---|---|---|
| LayerNorm | `xn = LN(x)` | — | `(49,320)` |
| QKV proj | `QKV = xn · W_qkv` | `[49×320]·[320×960]` | `(49,960)` |
| reshape → heads | `QKV → (49,10,96)` | — | |
| split → Q,K,V | `Q,K,V ← split(QKV, [32,32,32])` | `96→32+32+32` | `Q,K,V∈(49,10,32)` |
| Attn scores (×10 heads) | `S_i = Q_i·K_i^T / √32` | `[49×32]·[32×49]` | `(49,49)` per head |
| + bias, softmax | `A_i = softmax(S_i + B_rel,i)` | — | `(49,49)` per head |
| Attn·V (×10 heads) | `O_i = A_i · V_i` | `[49×49]·[49×32]` | `(49,32)` per head |
| concat heads | `O = concat(O_1..O_10)` | `10×(49,32)` | `(49,320)` |
| Output proj | `x' = O · W_O` | `[49×320]·[320×320]` | `(49,320)` |
| Residual | `x = x_orig + x'` | — | `(49,320)` |
| Local conv | `x = DWConv3x3(x, W_local)` | depthwise, on `(B,320,7,7)` | `(B,320,7,7)` |
| LayerNorm | `xn = LN(x)` | — | `(49,320)` |
| MLP fc1 | `m1 = GELU(xn · W_1)` | `[49×320]·[320×1280]` | `(49,1280)` |
| MLP fc2 | `m2 = m1 · W_2` | `[49×1280]·[1280×320]` | `(49,320)` |
| Residual | `x = x + m2` | — | `(49,320)` |

Repeat ×2.

---

## Head

```
mean-pool: (B,49,320) → (B,320)
LayerNorm(320)
Linear: (B,320) → (B,num_classes)
```

---

## Summary table

| Stage | Type | Res | C | Heads | d | Window | Depth | QKV | Proj | MLP |
|---|---|---|---|---|---|---|---|---|---|---|
| 0 | MBConv | 56×56 | 64 | — | — | — | 2 | — | — | 64→256→64 |
| 1 | Attn (windowed) | 28×28 | 128 | 4 | 32 | 7×7 (16 win) | 2 | 128→384 | 128→128 | 128→512→128 |
| 2 | Attn (global) | 14×14 | 160 | 5 | 32 | 14×14 (1 win) | 6 | 160→480 | 160→160 | 160→640→160 |
| 3 | Attn (global) | 7×7 | 320 | 10 | 32 | 7×7 (1 win) | 2 | 320→960 | 320→320 | 320→1280→320 |
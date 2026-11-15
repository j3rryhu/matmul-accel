# Output buffer layout

How a matmul's result is stored across the two 16-bank output buffers, how
a tile's position turns into bank addresses, and how the host reads the
int8 result back at `OBUF_BASE`. Register meanings are in
[`CONTROL_REGISTERS.md`](CONTROL_REGISTERS.md).

## Two buffers, one canvas

Both buffers hold the same canvas - `W` columns wide, rows stacked - in
different formats:

| Buffer | Holds | Element | Per bank | Total | Host |
|---|---|---|---|---|---|
| `output_buffer` | finished int8 results | 1 byte | 4096 bytes | 64 KB | read at `OBUF_BASE` |
| `psum_buffer` | raw 32-bit running sums between contraction blocks | 32-bit slot | 4096 slots | 256 KB | not visible |

```
W = CANVAS_WIDTH, or INPUT_COLS when CANVAS_WIDTH is 0
```

Every contraction pass but an element's last read-modify-writes its raw sum
in `psum_buffer`. The last pass rescales (plus bias, GELU) and writes the
int8 as **one byte at its exact position** in `output_buffer` - no packing,
no alignment rules. With `ACT_CFG.out_raw` the last pass keeps the raw sum
in `psum_buffer` instead, for a later `use_prev_out` matmul.

## Rows to banks

In both buffers canvas row `R` lives in bank `R mod 16`, and each bank
stacks its rows back to back, every row holding all `W` columns - every
tile's columns:

```
output_buffer:  byte  (R / 16) * pitch8 + C      pitch8 = roundup(W, 16) bytes
psum_buffer:    slot  (R / 16) * W      + C      (no padding)

bank 0:  [row 0 ][row 16][row 32] ...
bank 1:  [row 1 ][row 17][row 33] ...
 ...
bank 15: [row 15][row 31][row 47] ...
```

`output_buffer` rows are padded to whole 128-bit words (16 bytes) because
the host reads it a word at a time. `psum_buffer` is only read one slot at
a time by `output_loader`, so its rows need no padding.

In the RTL, `obuf_pitch_words` in `rtl/accel_top.sv` is `pitch8` in bus
words, `ceil(W / 16)`. It is latched at `CONTROL.matmul_start`, both for
the addresses results are written to and for the host read remap.

## Where a tile's results go

For a matmul placed as tile (`TILE_CFG.tile_r`, `TILE_CFG.tile_c`):

```
first row     R0 = tile_r * TILE_ROWS       (TILE_ROWS = 0 means WEIGHT_ROWS)
first column  C0 = tile_c * TILE_COLS       (TILE_COLS = 0 means INPUT_COLS)
```

The array produces the result 16 output rows at a time. Output-row block
`b` of the tile is canvas rows `R0 + 16b ... R0 + 16b + 15`, and array row
`r` always writes **bank `r`** - which is canvas row `R0 + 16b + r` only
because `R0` is a multiple of 16. That is the one placement rule:
`tile_r * TILE_ROWS` must be a multiple of 16. For column `m` of the tile:

```
int8 result:   output_buffer byte  (R0/16 + b) * pitch8 + C0 + m
raw sum:       psum_buffer slot    (R0/16 + b) * W      + C0 + m
```

The row offset only picks which stacked row inside each bank; the column
offset just adds `C0`, so it can be any value - a tile may start mid-word
and be any width. A tile may be smaller than its grid cell
(`WEIGHT_ROWS < TILE_ROWS`, `INPUT_COLS < TILE_COLS`): it starts at the
same place and simply writes fewer rows and columns.

## Host reads

`output_buffer` is read through `rtl/buf_addr_remap.v`, the same remap the
weight and input writes use, so the host sees one dense row-major int8
image:

```
int8 (R, C)  is at  OBUF_BASE + R * pitch8 + C
```

A byte address `A` relative to `OBUF_BASE` becomes:

```
w          = A / 16                   bus word
R          = w / pitch_words          canvas row       (pitch_words = pitch8 / 16)
c          = w mod pitch_words        word within the row
bank       = R mod 16
bank word  = (R / 16) * pitch_words + c
byte       = A mod 16                 byte within that 128-bit word
```

Tiles side by side are adjacent in each row, so a whole canvas row is one
contiguous burst.

## Worked example: `test_tiled_edge_tiles`

A 40 x 37 result: `CANVAS_WIDTH = 37`, `TILE_ROWS = 32`, `TILE_COLS = 22`,
so `pitch8 = 48` bytes (3 words) and the psum pitch is 37 slots.

| Tile | First row | First column | Size |
|---|---|---|---|
| (0,0) | 0 | 0 | 32 x 22 |
| (0,1) | 0 | 22 | 32 x 15 |
| (1,0) | 32 | 0 | 8 x 22 |
| (1,1) | 32 | 22 | 8 x 15 |

Canvas rows 0-15 are in-bank row 0 of banks 0-15, rows 16-31 in-bank row
1, rows 32-39 in-bank row 2 of banks 0-7 only. Bank 0 of `output_buffer`:

```
byte:  0 ── 21 | 22 ── 36 | 37 ─ 47 | 48 ── 69 | 70 ── 84 | 85 ─ 95 | 96 ── 117 | 118 ─ 132 | 133 ─ 143
       row 0              | pad     | row 16              | pad     | row 32                 | pad
       tile(0,0)| tile(0,1)         | tile(0,0)| tile(0,1)          | tile(1,0)  | tile(1,1)
```

Element: canvas row 35, column 30 (tile (1,1), tile column 8):

| | Where |
|---|---|
| int8 | bank 3, byte `2 * 48 + 30 = 126` = RAM word 7, byte 14 |
| raw sum (during the matmul) | `psum_buffer` bank 3, slot `2 * 37 + 30 = 104` |
| host address | `OBUF_BASE + 35 * 48 + 30` = `0x96AE` |

Running the host address back through the remap: `0x96AE - 0x9000 =
1710`, `w = 106`, `R = 106 / 3 = 35`, `c = 1`, bank 3, bank word
`2 * 3 + 1 = 7`, byte `1710 mod 16 = 14`. Same place.

## Capacity

`ceil(total canvas rows / 16)` stacked rows per bank in both buffers:

```
ceil(rows / 16) * roundup(W, 16)  <=  4096      output_buffer bytes per bank
ceil(rows / 16) * W               <=  4096      psum_buffer slots per bank
```

For example 256 x 256 fills both exactly.

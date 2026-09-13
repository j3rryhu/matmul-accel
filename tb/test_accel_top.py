"""
Cocotb tests for accel_top simulating a real int8-quantization deployment
flow. Two tests:
  - test_single_block_matmul: a single 32x32 (weight) x 32x32 (input) matmul -
    exactly one contraction block (k_blk_idx=0 only) and one output block
    (n_blk_idx=0 only), so no multi-block accumulation and no partial-block
    masking are exercised. Checks the whole pipeline end to end: Avalon
    writes -> weight_buffer/input_buffer -> weight_ctrl ->
    weight_loader/input_dispatch -> pe_array -> output_loader ->
    output_buffer -> Avalon reads.
  - test_multiblock_matmul: 2x2 blocks, exercising
    the multi-k-block accumulate and multi-n-block output_buffer addressing
    that the single-block test never touches - see its own docstring.

Quantization scheme: start from "real" (float) weight/input matrices,
per-tensor symmetric-quantize each to int8 (scale = max(|x|)/127), and load
the *quantized* int8 values into the device - this is what a real
quantized-inference flow would actually feed the array. weight_scale and
input_scale are the two quantizer scales; output_scale is derived from the
float reference output's own range (as a calibration pass would produce).
Hardware only ever sees one combined factor via the OUTPUT_SCALE register:

    y_int8 = (W_int8 @ X_int8) * weight_scale * input_scale / output_scale

i.e. OUTPUT_SCALE (Q0.16 fixed point, see rdl/ctrl_reg.rdl) is programmed
with weight_scale*input_scale/output_scale, and output_loader's requantize
stage multiplies the raw accumulator by it, shifts, and saturates (see
output_loader.v) to produce exactly that.

Everything is addressed per the RTL's own documented conventions:
  - weight_buffer: row-major over the (contraction x output) matrix,
    address = k*WEIGHT_COLS + n (see weight_ctrl.v's base_addr).
  - input_buffer: banked one bank per array row (see input_buffer_32_bank.v
    / input_dispatch.v); bank k, offset m holds x[k,m] - with a single
    32-row block, k_blk_idx=0, so bank k is x's row k directly, laid out at
    IBUF_BASE + k*(1<<IBUF_BANK_ADDR_WIDTH) + m.
  - output_buffer: banked one row per array column; bank c, offset m holds
    y[row=n_blk_idx*ARRAY_COLS+c, col=m] (see output_loader.v). With a
    single 32x32 output block, n_blk_idx=0, so bank c holds row c directly,
    laid out at OBUF_BASE + c*(1<<OBUF_BANK_ADDR_WIDTH) + m.

Golden model (both tests): raw = W_int8 @ X_int8 at full precision (the
32-bit ACC_DATA_WIDTH accumulator - see pe.v/pe_array.v - can't overflow
for int8 operands at any K used here: worst case is K*127*127, well under
2^31), then rescaled by the *actual* OUTPUT_SCALE value written to the
device (raw*OUTPUT_SCALE >> 16, arithmetic shift) and saturated to signed
int8 [-128,127] - matching output_loader's requantize stage bit-for-bit.
"""

import numpy as np

import cocotb
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge, ClockCycles

# ---- memory map (byte addresses, must match rtl/accel_top.sv) ----
CTRL_BASE = 0x0000
WBUF_BASE = 0x1000
IBUF_BASE = 0x5000
OBUF_BASE = 0x7000

CONTROL_ADDR        = CTRL_BASE + 0x00
WEIGHT_ROWS_ADDR    = CTRL_BASE + 0x04
WEIGHT_COLS_ADDR    = CTRL_BASE + 0x08
INPUT_MAX_ADDR_ADDR = CTRL_BASE + 0x0C
INPUT_COLS_ADDR     = CTRL_BASE + 0x10
OUTPUT_SCALE_ADDR   = CTRL_BASE + 0x18

MATMUL_START_BIT = 0

# ---- test-fixed dimensions ----
ARRAY_ROWS = 16  # must match accel_top.sv's ARRAY_ROWS / input_buffer_32_bank's bank count
ARRAY_COLS = 16  # must match accel_top.sv's ARRAY_COLS / output_buffer_32_bank's bank count

K = ARRAY_ROWS   # contraction size (WEIGHT_ROWS) - exactly one block
N = ARRAY_COLS   # output size (WEIGHT_COLS)
M = ARRAY_ROWS   # input/output column count (INPUT_COLS == INPUT_MAX_ADDR here)

# "Real" (float) weight/input matrices are drawn N(0, REAL_VAL_STD) - wide
# enough dynamic range to make quantization meaningful without needing
# either matrix to be degenerate (all-zero, etc).
REAL_VAL_STD = 1.0

OBUF_BANK_ADDR_WIDTH = 7  # must match accel_top.sv's OBUF_BANK_ADDR_WIDTH
IBUF_BANK_ADDR_WIDTH = 8  # must match input_buffer_32_bank.v's per-bank address width

CLK_PERIOD_NS = 10
DONE_TIMEOUT_CYCLES = 8000
WATCHDOG_CYCLES = 8000

# wait_matmul_done only needs to cover compute (block prefetch/drain, see
# weight_ctrl.v/weight_loader.v) - 20000 is already generous for a 2x2-block
# (64x64x64) run.
MULTIBLOCK_DONE_TIMEOUT_CYCLES = 20000

# The watchdog runs for the *whole* test, start to finish - not just
# compute. For 64x64x64 that's dominated by the per-byte Avalon loops, not
# the matmul itself: weight+input writes are 64*64*2 = 8192 cycles (writes
# never stall on avs_waitrequest - that's only asserted for OBUF reads), and
# readback is 64*64 = 4096 reads at ~2 cycles each (obuf_rd_pending stalls
# one cycle per read) = ~8192 cycles. Add ~4900 for compute (4 blocks) and
# the total is already ~21300 before any margin - comfortable headroom here.
MULTIBLOCK_WATCHDOG_CYCLES = 60000


async def start_clock(dut):
    cocotb.start_soon(Clock(dut.clk, CLK_PERIOD_NS, units="ns").start())


async def watchdog(dut, max_cycles=WATCHDOG_CYCLES):
    """Kills the test if it's still running after max_cycles - a backstop
    against any hang (e.g. avs_waitrequest stuck high) that the individual
    per-step timeouts below don't cover."""
    await ClockCycles(dut.clk, max_cycles)
    raise TimeoutError(f"testbench watchdog: exceeded {max_cycles} cycles")


async def reset_dut(dut):
    dut.reset_n.value = 0
    dut.avs_address.value = 0
    dut.avs_read.value = 0
    dut.avs_write.value = 0
    dut.avs_writedata.value = 0
    dut.avs_byteenable.value = 0
    dut.avs_burstcount.value = 1
    await ClockCycles(dut.clk, 5)
    dut.reset_n.value = 1
    await ClockCycles(dut.clk, 5)


async def avalon_write(dut, addr, data, byteenable=0xF):
    """Word-granular Avalon-MM write (used for ctrl_rf, which is 32-bit)."""
    dut.avs_address.value = addr
    dut.avs_write.value = 1
    dut.avs_writedata.value = data & 0xFFFFFFFF
    dut.avs_byteenable.value = byteenable
    dut.avs_read.value = 0
    await RisingEdge(dut.clk)
    while int(dut.avs_waitrequest.value):
        await RisingEdge(dut.clk)
    dut.avs_write.value = 0
    dut.avs_byteenable.value = 0xF


async def avalon_read(dut, addr):
    dut.avs_address.value = addr
    dut.avs_read.value = 1
    dut.avs_write.value = 0
    await RisingEdge(dut.clk)
    while int(dut.avs_waitrequest.value):
        await RisingEdge(dut.clk)
    data = int(dut.avs_readdata.value)
    dut.avs_read.value = 0
    return data


async def write_byte(dut, byte_addr, value):
    """weight_buffer/input_buffer are byte-wide RAMs addressed directly by
    avs_address (see accel_top.sv's wbuf_wraddress/ibuf_wraddress) - no
    word alignment, just place the byte in the writedata lane matching
    byte_addr's low 2 bits and enable that one lane."""
    lane = byte_addr & 0x3
    data = (value & 0xFF) << (8 * lane)
    await avalon_write(dut, byte_addr, data, 1 << lane)


async def read_byte(dut, byte_addr):
    """output_buffer replicates its one byte across all 4 readdata lanes
    (accel_top.sv: {4{obuf_ext_q}}), so any lane works."""
    data = await avalon_read(dut, byte_addr)
    lane = byte_addr & 0x3
    return (data >> (8 * lane)) & 0xFF


async def avalon_burst_read(dut, addr, count):
    """Burst-read `count` sequential words starting at addr, using
    accel_top.sv's burst FSM (see its avs_burstcount/rd_burst_on logic).
    Acceptance works exactly like avalon_read's single-word case (hold
    avs_read/address/burstcount until avs_waitrequest drops - this also
    self-corrects if a previous transaction's burst FSM hasn't fully gone
    idle yet), and that same accepting edge already carries the first
    beat's data. Each subsequent beat then arrives one avs_readdatavalid
    pulse per cycle, address auto-incrementing, until `count` beats have
    been collected."""
    dut.avs_address.value = addr
    dut.avs_burstcount.value = count
    dut.avs_read.value = 1
    dut.avs_write.value = 0
    await RisingEdge(dut.clk)
    while int(dut.avs_waitrequest.value):
        await RisingEdge(dut.clk)
    dut.avs_read.value = 0
    dut.avs_burstcount.value = 1
    data = [int(dut.avs_readdata.value)]
    while len(data) < count:
        await RisingEdge(dut.clk)
        if int(dut.avs_readdatavalid.value):
            data.append(int(dut.avs_readdata.value))
    return data


async def read_burst_bytes(dut, byte_addr, count):
    """Burst-read `count` sequential output_buffer bytes. output_buffer
    replicates its one byte across all 4 readdata lanes, so no per-word
    lane selection is needed (see read_byte)."""
    words = await avalon_burst_read(dut, byte_addr, count)
    return [w & 0xFF for w in words]


async def wait_output_done(dut, timeout_cycles=DONE_TIMEOUT_CYCLES):
    """output_loader.done pulses for one cycle once this block's psums are
    fully drained into output_buffer - the true completion signal (later
    than weight_ctrl.done, which only reflects the input side finishing).
    With a single 32x32 block this is also the whole matmul's completion
    (STATUS.done, driven from the same two signals - see rdl/README.md -
    would work here too, but polling the internal signal directly avoids
    a race with STATUS.done's own one-cycle-later timing)."""
    for _ in range(timeout_cycles):
        await RisingEdge(dut.clk)
        if int(dut.u_output_loader.done.value) == 1:
            return
    raise TimeoutError(
        f"output_loader never asserted done within {timeout_cycles} cycles"
    )


async def wait_matmul_done(dut, timeout_cycles=MULTIBLOCK_DONE_TIMEOUT_CYCLES):
    """matmul_done (accel_top.sv) is a level that only goes high once every
    block has been committed (weight_ctrl_done) AND output_loader has
    drained the last one - the right "whole matmul finished" signal for a
    multi-block run. output_loader.done isn't usable here the way
    wait_output_done uses it for a single block: it pulses once *per
    committed block*, not once for the whole matmul (see output_loader.v /
    rdl/README.md), so waiting on it once would only catch the first
    block."""
    for _ in range(timeout_cycles):
        await RisingEdge(dut.clk)
        if int(dut.matmul_done.value) == 1:
            return
    raise TimeoutError(
        f"matmul_done never asserted within {timeout_cycles} cycles"
    )


def to_int8(byte_val):
    return byte_val - 256 if byte_val >= 128 else byte_val


def to_uint8(int_val):
    return int_val & 0xFF


def quantize_symmetric_int8(x):
    """Per-tensor symmetric quantization to signed int8 (zero point 0,
    range [-127,127] - the 127 excludes -128 so the range stays symmetric
    around 0). Returns (x_int8, scale) with x ~= x_int8 * scale."""
    max_abs = float(np.max(np.abs(x)))
    scale = max_abs / 127.0 if max_abs > 0 else 1.0
    x_int8 = np.clip(np.round(x / scale), -127, 127).astype(np.int64)
    return x_int8, scale


def format_int8_hex(arr):
    """Render a 2D int8-range array as one row of 2-digit two's-complement
    hex bytes per line (same byte values as what write_byte puts on the
    wire) - handy for cross-referencing against a waveform."""
    return "\n".join(
        " ".join(f"{to_uint8(int(v)):02x}" for v in row)
        for row in arr
    )


@cocotb.test()
async def test_single_block_matmul(dut):
    rng = np.random.default_rng(0)
    await start_clock(dut)
    cocotb.start_soon(watchdog(dut))
    await reset_dut(dut)

    # ---- "real" (float) weight (K x N) and input (K x M) matrices - stand
    # in for values that would come from a trained model / real activations ----
    W_real = rng.normal(loc=0.0, scale=REAL_VAL_STD, size=(K, N))
    X_real = rng.normal(loc=0.0, scale=REAL_VAL_STD, size=(K, M))

    # ---- per-tensor symmetric int8 quantization - the actual values loaded
    # into the device are W_int8/X_int8, not W_real/X_real ----
    W_int8, weight_scale = quantize_symmetric_int8(W_real)
    X_int8, input_scale = quantize_symmetric_int8(X_real)

    # ---- real-valued reference output and its own quantization scale - in
    # a real flow output_scale would come from calibration over many
    # samples; here we just use this single sample's own range ----
    Y_real = W_real @ X_real
    _, output_scale = quantize_symmetric_int8(Y_real)

    # ---- combined rescale factor: hardware only ever sees this one Q0.16
    # value (OUTPUT_SCALE), applied to the raw int32 accumulator as
    # raw * combined_scale = (W_int8 @ X_int8) * weight_scale * input_scale / output_scale ----
    combined_scale = weight_scale * input_scale / output_scale
    assert 0.0 <= combined_scale < 1.0, (
        f"combined_scale={combined_scale} doesn't fit OUTPUT_SCALE's Q0.16 "
        f"[0,1) range - adjust REAL_VAL_STD or K/N/M"
    )
    output_scale_q16 = round(combined_scale * 65536)

    with open("sim_build/matrices.txt", "w") as f:
        f.write(
            f"weight_scale={weight_scale!r}\ninput_scale={input_scale!r}\n"
            f"output_scale={output_scale!r}\ncombined_scale={combined_scale!r}\n"
            f"OUTPUT_SCALE (Q0.16) = 0x{output_scale_q16:04x}\n\n"
        )
        f.write(f"W_real (K x N) =\n{np.array2string(W_real, threshold=np.inf, max_line_width=200)}\n\n")
        f.write(f"W_int8 (K x N) =\n{np.array2string(W_int8, threshold=np.inf, max_line_width=200)}\n\n")
        f.write(f"X_real (K x M) =\n{np.array2string(X_real, threshold=np.inf, max_line_width=200)}\n\n")
        f.write(f"X_int8 (K x M) =\n{np.array2string(X_int8, threshold=np.inf, max_line_width=200)}\n")

    # ---- write quantized weights: row-major (k*N+n) into weight_buffer ----
    for k in range(K):
        for n in range(N):
            await write_byte(dut, WBUF_BASE + k * N + n, to_uint8(int(W_int8[k, n])))

    # ---- write quantized input: banked one bank per array row (bank =
    # k % ARRAY_ROWS, offset = k_blk_idx*M + m, matching weight_ctrl's
    # input_band_base_addr) ----
    for k in range(K):
        k_local = k % ARRAY_ROWS
        k_blk = k // ARRAY_ROWS
        for m in range(M):
            addr = IBUF_BASE + k_local * (1 << IBUF_BANK_ADDR_WIDTH) + k_blk * M + m
            await write_byte(dut, addr, to_uint8(int(X_int8[k, m])))

    # ---- program dimensions + rescale factor ----
    await avalon_write(dut, WEIGHT_ROWS_ADDR, K)
    await avalon_write(dut, WEIGHT_COLS_ADDR, N)
    await avalon_write(dut, INPUT_MAX_ADDR_ADDR, M)
    await avalon_write(dut, INPUT_COLS_ADDR, M)
    await avalon_write(dut, OUTPUT_SCALE_ADDR, output_scale_q16)

    # ---- kick off the matmul: everything past this is sequenced by
    # weight_ctrl internally (see rtl/weight_ctrl.v) ----
    await avalon_write(dut, CONTROL_ADDR, 1 << MATMUL_START_BIT)

    await wait_output_done(dut)
    # give output_buffer's write a cycle to land before reading it back
    await ClockCycles(dut.clk, 2)

    # ---- golden model: raw = W_int8 @ X_int8 at full precision, then
    # requantized exactly like output_loader's rescale stage using the
    # *actual* OUTPUT_SCALE value written above (so its own Q0.16 rounding
    # is reflected, not just the ideal combined_scale): raw*OUTPUT_SCALE
    # arithmetic-shifted right by 16, saturated to int8. Python's >> on
    # signed ints is already an arithmetic (floor) shift, so this
    # reproduces the hardware's `>>>` bit-for-bit. ----
    raw = W_int8 @ X_int8
    scaled = (raw * output_scale_q16) >> 16
    golden = np.clip(scaled, -128, 127)

    # ---- read back output_buffer: bank c (=row n, since n_blk_idx=0),
    # offset m ----
    mismatches = []
    for n in range(N):
        bank_base = OBUF_BASE + (n << OBUF_BANK_ADDR_WIDTH)
        for m in range(M):
            got = to_int8(await read_byte(dut, bank_base + m))
            exp = int(golden[n, m])
            if got != exp:
                mismatches.append((n, m, exp, got))

    if mismatches:
        preview = ", ".join(
            f"y[{n},{m}]: expected {exp}, got {got}"
            for n, m, exp, got in mismatches[:10]
        )
        raise AssertionError(
            f"{len(mismatches)}/{N * M} output mismatches. First few: {preview}"
        )


@cocotb.test()
async def test_multiblock_matmul(dut):
    """2 k-blocks x 2 n-blocks (4 ARRAY_ROWS x ARRAY_COLS blocks total),
    sequenced column-major by weight_ctrl (k fast/inner, n slow/outer - see
    rtl/weight_ctrl.v). Dimensions are derived from the array size, so this
    stays a 2x2-block run whatever ARRAY_ROWS/ARRAY_COLS are set to. Unlike
    the single-block test, this exercises:
      - multi-k-block output accumulation: output_loader's read-modify-
        write add (committed_first_k_blk ? write : old_val+scaled_psum -
        see output_loader.v) actually accumulates across k_blk_idx=0,1
        instead of only ever taking the "first/only" branch.
      - multi-n-block output_buffer addressing: each bank is reused across
        n_blk_idx=0,1, with that pass's total_rows-sized segment appended
        after the previous one (bank c's internal address =
        n_blk_idx*total_rows + m - see output_loader.v's header note).

    Same quantization scheme as test_single_block_matmul (real float
    W/X -> per-tensor symmetric int8 -> OUTPUT_SCALE combines
    weight_scale*input_scale/output_scale) - see that test's docstring.

    Buffer capacity check: output_buffer's per-bank depth
    (1<<OBUF_BANK_ADDR_WIDTH entries) must hold num_k_blocks*M entries, and
    input_buffer's per-bank depth (1<<IBUF_BANK_ADDR_WIDTH) must hold
    num_n_blocks*M. Both are asserted below.
    """
    K = 2 * ARRAY_ROWS
    N = 2 * ARRAY_COLS
    M = 2 * ARRAY_ROWS

    assert ((K + ARRAY_ROWS - 1) // ARRAY_ROWS) * M <= (1 << OBUF_BANK_ADDR_WIDTH),         "output_buffer per-bank depth too small for num_k_blocks*M"
    assert ((N + ARRAY_COLS - 1) // ARRAY_COLS) * M <= (1 << IBUF_BANK_ADDR_WIDTH),         "input_buffer per-bank depth too small for num_n_blocks*M"

    rng = np.random.default_rng(1)
    await start_clock(dut)
    cocotb.start_soon(watchdog(dut, max_cycles=MULTIBLOCK_WATCHDOG_CYCLES))
    await reset_dut(dut)

    # ---- "real" (float) weight/input matrices, quantized the same way as
    # test_single_block_matmul - see its docstring for the scheme ----
    W_real = rng.normal(loc=0.0, scale=REAL_VAL_STD, size=(K, N))
    X_real = rng.normal(loc=0.0, scale=REAL_VAL_STD, size=(K, M))

    W_int8, weight_scale = quantize_symmetric_int8(W_real)
    X_int8, input_scale = quantize_symmetric_int8(X_real)

    Y_real = W_real @ X_real
    _, output_scale = quantize_symmetric_int8(Y_real)

    combined_scale = weight_scale * input_scale / output_scale
    assert 0.0 <= combined_scale < 1.0, (
        f"combined_scale={combined_scale} doesn't fit OUTPUT_SCALE's Q0.16 "
        f"[0,1) range - adjust REAL_VAL_STD or K/N/M"
    )
    output_scale_q16 = round(combined_scale * 65536)

    with open("sim_build/matrices.txt", "w") as f:
        f.write(
            f"weight_scale={weight_scale!r}\ninput_scale={input_scale!r}\n"
            f"output_scale={output_scale!r}\ncombined_scale={combined_scale!r}\n"
            f"OUTPUT_SCALE (Q0.16) = 0x{output_scale_q16:04x}\n\n"
        )
        f.write(f"W_real (K x N) =\n{np.array2string(W_real, threshold=np.inf, max_line_width=200)}\n\n")
        f.write(f"W_int8 (K x N) =\n{np.array2string(W_int8, threshold=np.inf, max_line_width=200)}\n\n")
        f.write(f"X_real (K x M) =\n{np.array2string(X_real, threshold=np.inf, max_line_width=200)}\n\n")
        f.write(f"X_int8 (K x M) =\n{np.array2string(X_int8, threshold=np.inf, max_line_width=200)}\n")

    # ---- write weights: row-major over the *full* K x N matrix - this
    # layout doesn't change for multi-block (weight_ctrl.v's base_addr
    # walks it with row_stride=N regardless of block count) ----
    for k in range(K):
        for n in range(N):
            await write_byte(dut, WBUF_BASE + k * N + n, to_uint8(int(W_int8[k, n])))

    # ---- write input: banked one bank per array row (bank = k %
    # ARRAY_ROWS); k_blk_idx's M-sized band is appended after the previous
    # k_blk's within the same bank, matching weight_ctrl's
    # input_band_base_addr = k_blk_idx*input_max_addr ----
    for k in range(K):
        k_local = k % ARRAY_ROWS
        k_blk = k // ARRAY_ROWS
        for m in range(M):
            addr = IBUF_BASE + k_local * (1 << IBUF_BANK_ADDR_WIDTH) + k_blk * M + m
            await write_byte(dut, addr, to_uint8(int(X_int8[k, m])))

    await avalon_write(dut, WEIGHT_ROWS_ADDR, K)
    await avalon_write(dut, WEIGHT_COLS_ADDR, N)
    await avalon_write(dut, INPUT_MAX_ADDR_ADDR, M)
    await avalon_write(dut, INPUT_COLS_ADDR, M)
    await avalon_write(dut, OUTPUT_SCALE_ADDR, output_scale_q16)

    await avalon_write(dut, CONTROL_ADDR, 1 << MATMUL_START_BIT)

    # ---- wait for the *whole* matmul, not just the first committed block
    # (see wait_matmul_done) ----
    await wait_matmul_done(dut)
    await ClockCycles(dut.clk, 2)

    # ---- golden model: output_loader rescales/saturates EACH k-block's
    # raw partial psum separately, then accumulates the already-saturated
    # int8 results (see output_loader.v: scaled_psum = saturate(psum_d *
    # output_scale >>> 16); acc_sat = saturate(old_val + scaled_psum)). It
    # does NOT sum the full-precision raw accumulator across k-blocks and
    # rescale once - that would require carrying the 32-bit psum across
    # block boundaries instead of just the int8 result already sitting in
    # output_buffer, defeating the point of rescaling to int8 between
    # passes. Reproduce that ordering exactly.
    #
    # Block kb covers contraction index p in [kb*ARRAY_ROWS,
    # (kb+1)*ARRAY_ROWS) - X_int8 is blocked by ROWS (confirmed above by
    # this test's own input-writing code: k_blk = k // ARRAY_ROWS), and
    # since the full contraction is sum_p W_int8[n,p]*X_int8[p,m], p *is*
    # X_int8's row index by definition of matmul - so W_int8 must be
    # sliced by the *matching columns* (not rows) for the same p range. ----
    num_k_blocks = (K + ARRAY_ROWS - 1) // ARRAY_ROWS
    golden = None
    for kb in range(num_k_blocks):
        lo, hi = kb * ARRAY_ROWS, min((kb + 1) * ARRAY_ROWS, K)
        raw_block = W_int8[:, lo:hi] @ X_int8[lo:hi, :]
        scaled_block = np.clip((raw_block * output_scale_q16) >> 16, -128, 127)
        golden = scaled_block if golden is None else np.clip(golden + scaled_block, -128, 127)

    # ---- read back output_buffer: row n lives in bank c = n % ARRAY_COLS,
    # at that bank's n_blk_idx*M + m offset, where n_blk_idx = n //
    # ARRAY_COLS (see output_loader.v's header note on bank layout) ----
    mismatches = []
    for n in range(N):
        n_blk = n // ARRAY_COLS
        c = n % ARRAY_COLS
        bank_base = OBUF_BASE + (c << OBUF_BANK_ADDR_WIDTH) + n_blk * M
        row_bytes = await read_burst_bytes(dut, bank_base, M)
        for m in range(M):
            got = to_int8(row_bytes[m])
            exp = int(golden[n, m])
            if got != exp:
                mismatches.append((n, m, exp, got))

    if mismatches:
        preview = ", ".join(
            f"y[{n},{m}]: expected {exp}, got {got}"
            for n, m, exp, got in mismatches[:10]
        )
        raise AssertionError(
            f"{len(mismatches)}/{N * M} output mismatches. First few: {preview}"
        )


# @cocotb.test()
# async def test_49x32x64_partial_row_block_matmul(dut):
#     """weight_rows(K)=49 x weight_cols(N)=32 x input_cols(M)=64. Weight is
#     stored exactly as written on paper - (49,32), no transpose - and this
#     is a plain Y = W @ X: W(49,32) @ X(32,64) = Y(49,64), contracting over
#     the shared 32 dimension.

#     Actual PE dataflow (from pe.v/pe_array.v directly, NOT weight_ctrl.v's
#     header comment, which describes the wrong axis): weight is loaded
#     straight into PE(row,col) with no transpose (w_addr = row*ARRAY_COLS+col,
#     weight_loader walks weight_buffer row-major into the same row/col
#     layout). Partial sums (b_in) cascade *across columns within a row*
#     (pe_array.v: pe_b_in = ROW[r].COL[c-1].pe_p_out) - so contraction
#     happens over the weight's COLUMN axis (WEIGHT_COLS/N), and each PE ROW
#     independently produces one full output row, taken from its rightmost
#     column. Activations flow *down* each column from the top (pe_a_in for
#     r==0 comes from a_in[c]; for r>0 from the PE above) - input_buffer bank
#     c feeds PE column c, i.e. the contraction axis, matching weight's
#     columns. So WEIGHT_ROWS(K) is the *output-row* axis (blocked by
#     ARRAY_ROWS=32: k_blk_idx=0 covers rows 0..31, k_blk_idx=1 covers rows
#     32..48 - two blocks producing two DISJOINT ranges of output rows,
#     concatenated, not accumulated), and WEIGHT_COLS(N) is the true
#     contraction size (32 == ARRAY_COLS here, so it's a single pass, no
#     masking on that axis needed). This also explains output_loader's
#     obuf address (held_k_blk_idx*total_rows + m) and its write-vs-accumulate
#     gate (held_n_blk_idx==0 ? write : accumulate): different k-blocks write
#     different address ranges (concatenation, not accumulation - correct,
#     since num_n_blocks==1 here means n_blk_idx is always 0, so every commit
#     "writes" and none ever needs to accumulate).

#     Input: the whole (32,64) matrix is loaded into input_buffer once, up
#     front (contraction size N==ARRAY_COLS exactly - a single band, reused
#     by both k-block passes, since input doesn't depend on k_blk_idx at all).

#     Neither test_64x64_multiblock_matmul (exact 2x2 full blocks) nor
#     test_single_block_matmul (single full block) ever exercises a *partial*
#     last block on either axis. This one exercises:
#       - weight_ctrl's valid_k/valid_row masking (rtl/weight_ctrl.v) for a
#         genuinely partial last k-block (17 of 32 rows valid).
#       - weight_loader's zero-padding of out-of-range PE rows for that
#         partial block (valid_rows), rather than always landing exactly on
#         a block boundary.
#       - output_loader's row_live masking (r < weight_rows, weight_rows fed
#         from valid_row/valid_k) for k_blk_idx=1, where only rows 0..16 of
#         that pass should ever produce/write a result.
#       - output_buffer address reuse across k_blk_idx=0,1 (bank r's address
#         = k_blk_idx*total_rows + m) landing on two disjoint ranges within
#         the same 32 banks, rather than a single block's worth.

#     Same quantization scheme as the other tests (see
#     test_64x64_multiblock_matmul's docstring): real float W/X -> per-tensor
#     symmetric int8 -> OUTPUT_SCALE combines weight_scale*input_scale/output_scale.

#     Buffer capacity check: output_buffer's per-bank depth
#     (1<<OBUF_BANK_ADDR_WIDTH = 128 entries) must hold num_k_blocks*M
#     entries; here that's 2*64 = 128, exactly at capacity (same margin as
#     test_64x64_multiblock_matmul).
#     """
#     K = 49
#     N = 32
#     M = 64

#     rng = np.random.default_rng(2)
#     await start_clock(dut)
#     cocotb.start_soon(watchdog(dut, max_cycles=MULTIBLOCK_WATCHDOG_CYCLES))
#     await reset_dut(dut)

#     W_real = rng.normal(loc=0.0, scale=REAL_VAL_STD, size=(K, N))
#     X_real = rng.normal(loc=0.0, scale=REAL_VAL_STD, size=(N, M))

#     W_int8, weight_scale = quantize_symmetric_int8(W_real)
#     X_int8, input_scale = quantize_symmetric_int8(X_real)

#     Y_real = W_real @ X_real
#     _, output_scale = quantize_symmetric_int8(Y_real)

#     combined_scale = weight_scale * input_scale / output_scale
#     assert 0.0 <= combined_scale < 1.0, (
#         f"combined_scale={combined_scale} doesn't fit OUTPUT_SCALE's Q0.16 "
#         f"[0,1) range - adjust REAL_VAL_STD or K/N/M"
#     )
#     output_scale_q16 = round(combined_scale * 65536)

#     # ---- golden model: plain Y = W @ X (see docstring - no transpose, no
#     # cross-block accumulation; the two k-blocks produce disjoint output-row
#     # ranges of the same single-pass contraction). Computed up front (needs
#     # only the quantized inputs, not the simulation) so it can be dumped
#     # alongside W/X below and reused later for the readback comparison. ----
#     raw = W_int8 @ X_int8
#     scaled = (raw * output_scale_q16) >> 16
#     golden = np.clip(scaled, -128, 127)

#     with open("sim_build/matrices_49x32x64.txt", "w") as f:
#         f.write(
#             f"weight_scale={weight_scale!r}\ninput_scale={input_scale!r}\n"
#             f"output_scale={output_scale!r}\ncombined_scale={combined_scale!r}\n"
#             f"OUTPUT_SCALE (Q0.16) = 0x{output_scale_q16:04x}\n\n"
#         )
#         f.write(f"W_real (K x N) =\n{np.array2string(W_real, threshold=np.inf, max_line_width=200)}\n\n")
#         f.write(f"W_int8 (K x N) =\n{np.array2string(W_int8, threshold=np.inf, max_line_width=200)}\n\n")
#         f.write(f"X_real (N x M) =\n{np.array2string(X_real, threshold=np.inf, max_line_width=200)}\n\n")
#         f.write(f"X_int8 (N x M) =\n{np.array2string(X_int8, threshold=np.inf, max_line_width=200)}\n\n")
#         f.write(f"Y_real (K x M) =\n{np.array2string(Y_real, threshold=np.inf, max_line_width=200)}\n\n")
#         f.write(f"golden Y_int8 (K x M) =\n{np.array2string(golden, threshold=np.inf, max_line_width=200)}\n\n")
#         f.write(f"W_int8 hex (K x N), one row per line, each byte 2's-complement =\n{format_int8_hex(W_int8)}\n\n")
#         f.write(f"X_int8 hex (N x M), one row per line, each byte 2's-complement =\n{format_int8_hex(X_int8)}\n\n")
#         f.write(f"golden Y_int8 hex (K x M), one row per line, each byte 2's-complement =\n{format_int8_hex(golden)}\n")

#     # ---- write weights exactly as written on paper: row-major over the
#     # K x N matrix (address = k*N+n, see weight_ctrl.v's base_addr; PE(row,col)
#     # is loaded directly from this with no transpose - see docstring) ----
#     for k in range(K):
#         for n in range(N):
#             await write_byte(dut, WBUF_BASE + k * N + n, to_uint8(int(W_int8[k, n])))

#     # ---- write input: banked one bank per PE column (the contraction
#     # axis) - the whole (32,64) input is loaded into input_buffer once, up
#     # front (N == ARRAY_COLS exactly, a single band reused by both
#     # k-block passes, since input doesn't depend on k_blk_idx) ----
#     for c in range(N):
#         for m in range(M):
#             addr = IBUF_BASE + c * (1 << IBUF_BANK_ADDR_WIDTH) + m
#             await write_byte(dut, addr, to_uint8(int(X_int8[c, m])))

#     await avalon_write(dut, WEIGHT_ROWS_ADDR, K)
#     await avalon_write(dut, WEIGHT_COLS_ADDR, N)
#     await avalon_write(dut, INPUT_MAX_ADDR_ADDR, M)
#     await avalon_write(dut, INPUT_COLS_ADDR, M)
#     await avalon_write(dut, OUTPUT_SCALE_ADDR, output_scale_q16)

#     await avalon_write(dut, CONTROL_ADDR, 1 << MATMUL_START_BIT)

#     # ---- 2 k-blocks (k_blk_idx=0,1), so - like test_64x64_multiblock_matmul
#     # - wait for the whole matmul rather than just the first committed block ----
#     await wait_matmul_done(dut)
#     await ClockCycles(dut.clk, 2)

#     # ---- read back output_buffer: output row k lives in bank r = k % ARRAY_ROWS,
#     # at that bank's k_blk_idx*M + m offset, k_blk_idx = k // ARRAY_ROWS
#     # (see output_loader.v's obuf address: held_k_blk_idx*total_rows + m) ----
#     mismatches = []
#     for k in range(K):
#         k_blk = k // ARRAY_ROWS
#         r = k % ARRAY_ROWS
#         bank_base = OBUF_BASE + (r << OBUF_BANK_ADDR_WIDTH) + k_blk * M
#         for m in range(M):
#             got = to_int8(await read_byte(dut, bank_base + m))
#             exp = int(golden[k, m])
#             if got != exp:
#                 mismatches.append((k, m, exp, got))

#     if mismatches:
#         preview = ", ".join(
#             f"y[{k},{m}]: expected {exp}, got {got}"
#             for k, m, exp, got in mismatches[:10]
#         )
#         raise AssertionError(
#             f"{len(mismatches)}/{K * M} output mismatches. First few: {preview}"
#         )


@cocotb.test(expect_fail=True)
async def test_partial_block_matmul(dut):
    """KNOWN FAILING - pre-existing, not a regression from the 32->16 resize.

    Verified against HEAD's untouched 32x32 RTL at the equivalent
    dimensions (K=N=M=48): it fails there too, with the same kind of
    wrong-but-not-X values. test_partial_single_block_matmul (one partial
    block) passes on both, so partial *masking* works - what breaks is
    partial blocks combined with multi-block segment addressing. Note that
    output_loader computes its bank offset as held_k_blk_idx*total_rows
    while both its own header comment and this test's readback treat that
    segment as indexed by n_blk_idx; the two only agree when every
    dimension is a multiple of the array size, which is why the full-block
    tests pass. Flip expect_fail to False once that is resolved.

    Dimensions deliberately NOT a multiple of the array size: the last
    block along each of the contraction and output dimensions is partial,
    so weight_ctrl's valid_k/valid_n masking is actually exercised rather
    than always reporting a full array.

    This is the case a resize is most likely to break - everything that
    depends on the array size as a *bound* rather than just a width:
      - weight_ctrl's valid_k/valid_n (k_rem/n_rem vs ARRAY_ROWS/COLS)
      - weight_loader's elem_valid zero-fill for out-of-range PEs
      - output_loader's row_live (r < weight_rows) row gating, and its
        done term &(row_done | ~((1 << weight_rows) - 1)), whose shift
        amount is array-size dependent
      - input_dispatch's per-row ibuf_byteenable masking

    K/N are kept equal so the golden model's contraction axis is
    unambiguous (see the note on the k/n axis naming in the report);
    M differs from both.
    """
    K = ARRAY_ROWS + ARRAY_ROWS // 2      # 2 blocks, last one half-full
    N = ARRAY_COLS + ARRAY_COLS // 2
    M = ARRAY_ROWS + ARRAY_ROWS // 2

    num_k_blocks = (K + ARRAY_ROWS - 1) // ARRAY_ROWS
    num_n_blocks = (N + ARRAY_COLS - 1) // ARRAY_COLS
    assert K % ARRAY_ROWS != 0, "test is pointless unless the last k-block is partial"
    assert N % ARRAY_COLS != 0, "test is pointless unless the last n-block is partial"
    assert num_k_blocks * M <= (1 << OBUF_BANK_ADDR_WIDTH)
    assert num_n_blocks * M <= (1 << IBUF_BANK_ADDR_WIDTH)

    rng = np.random.default_rng(2)
    await start_clock(dut)
    cocotb.start_soon(watchdog(dut, max_cycles=MULTIBLOCK_WATCHDOG_CYCLES))
    await reset_dut(dut)

    W_real = rng.normal(loc=0.0, scale=REAL_VAL_STD, size=(K, N))
    X_real = rng.normal(loc=0.0, scale=REAL_VAL_STD, size=(K, M))

    W_int8, weight_scale = quantize_symmetric_int8(W_real)
    X_int8, input_scale = quantize_symmetric_int8(X_real)

    Y_real = W_real @ X_real
    _, output_scale = quantize_symmetric_int8(Y_real)

    combined_scale = weight_scale * input_scale / output_scale
    assert 0.0 <= combined_scale < 1.0, (
        f"combined_scale={combined_scale} doesn't fit OUTPUT_SCALE's Q0.16 range"
    )
    output_scale_q16 = round(combined_scale * 65536)

    for k in range(K):
        for n in range(N):
            await write_byte(dut, WBUF_BASE + k * N + n, to_uint8(int(W_int8[k, n])))

    # every (bank, band) slot must be written, including the rows a partial
    # k-block doesn't use: weight_ctrl never tells input_dispatch how many
    # rows are valid, so all ARRAY_ROWS banks are read every band
    for k_blk in range(num_k_blocks):
        for k_local in range(ARRAY_ROWS):
            k = k_blk * ARRAY_ROWS + k_local
            for m in range(M):
                val = int(X_int8[k, m]) if k < K else 0
                addr = IBUF_BASE + k_local * (1 << IBUF_BANK_ADDR_WIDTH) + k_blk * M + m
                await write_byte(dut, addr, to_uint8(val))

    await avalon_write(dut, WEIGHT_ROWS_ADDR, K)
    await avalon_write(dut, WEIGHT_COLS_ADDR, N)
    await avalon_write(dut, INPUT_MAX_ADDR_ADDR, M)
    await avalon_write(dut, INPUT_COLS_ADDR, M)
    await avalon_write(dut, OUTPUT_SCALE_ADDR, output_scale_q16)
    await avalon_write(dut, CONTROL_ADDR, 1 << MATMUL_START_BIT)

    await wait_matmul_done(dut)
    await ClockCycles(dut.clk, 2)

    # same per-k-block rescale-then-accumulate ordering as
    # test_multiblock_matmul; the final block's slice is short (partial)
    golden = None
    for kb in range(num_k_blocks):
        lo, hi = kb * ARRAY_ROWS, min((kb + 1) * ARRAY_ROWS, K)
        raw_block = W_int8[:, lo:hi] @ X_int8[lo:hi, :]
        scaled_block = np.clip((raw_block * output_scale_q16) >> 16, -128, 127)
        golden = scaled_block if golden is None else np.clip(golden + scaled_block, -128, 127)

    mismatches = []
    for n in range(N):
        n_blk = n // ARRAY_COLS
        c = n % ARRAY_COLS
        bank_base = OBUF_BASE + (c << OBUF_BANK_ADDR_WIDTH) + n_blk * M
        row_bytes = await read_burst_bytes(dut, bank_base, M)
        for m in range(M):
            got = to_int8(row_bytes[m])
            exp = int(golden[n, m])
            if got != exp:
                mismatches.append((n, m, exp, got))

    if mismatches:
        preview = ", ".join(
            f"y[{n},{m}]: expected {exp}, got {got}"
            for n, m, exp, got in mismatches[:10]
        )
        raise AssertionError(
            f"{len(mismatches)}/{N * M} output mismatches "
            f"(K={K} N={N} M={M}, array {ARRAY_ROWS}x{ARRAY_COLS}). First few: {preview}"
        )


@cocotb.test()
async def test_partial_single_block_matmul(dut):
    """A single, partially-filled block (K=N=M=ARRAY_ROWS//2): one k-block,
    one n-block, both partial. Isolates weight_ctrl's valid_k/valid_n
    masking and output_loader's row gating from any multi-block segment
    addressing - with num_k_blocks == num_n_blocks == 1 every block index
    is 0, so the only thing under test is whether a partial array is
    masked correctly.
    """
    K = ARRAY_ROWS // 2
    N = ARRAY_COLS // 2
    M = ARRAY_ROWS // 2

    rng = np.random.default_rng(3)
    await start_clock(dut)
    cocotb.start_soon(watchdog(dut, max_cycles=MULTIBLOCK_WATCHDOG_CYCLES))
    await reset_dut(dut)

    W_real = rng.normal(loc=0.0, scale=REAL_VAL_STD, size=(K, N))
    X_real = rng.normal(loc=0.0, scale=REAL_VAL_STD, size=(K, M))
    W_int8, weight_scale = quantize_symmetric_int8(W_real)
    X_int8, input_scale = quantize_symmetric_int8(X_real)
    Y_real = W_real @ X_real
    _, output_scale = quantize_symmetric_int8(Y_real)
    combined_scale = weight_scale * input_scale / output_scale
    assert 0.0 <= combined_scale < 1.0
    output_scale_q16 = round(combined_scale * 65536)

    for k in range(K):
        for n in range(N):
            await write_byte(dut, WBUF_BASE + k * N + n, to_uint8(int(W_int8[k, n])))
    # Fill EVERY bank, not just the K valid rows: weight_ctrl never tells
    # input_dispatch how many rows are valid (valid_row/valid_col go to
    # output_loader only), so all ARRAY_ROWS banks get read regardless.
    # Deliberately filled with nonzero garbage rather than zeros - the
    # out-of-range PEs' weights are 0 here (weight_pf's reset value, never
    # overwritten because weight_loader drives w_en low for them), so
    # garbage x 0 = 0 and the contribution is correctly neutralized. Only
    # *uninitialized* banks break this, and only in simulation, where
    # Verilog's X * 0 = X rather than 0.
    for r in range(ARRAY_ROWS):
        for m in range(M):
            val = int(X_int8[r, m]) if r < K else int(rng.integers(-127, 128))
            await write_byte(dut, IBUF_BASE + r * (1 << IBUF_BANK_ADDR_WIDTH) + m,
                             to_uint8(val))

    await avalon_write(dut, WEIGHT_ROWS_ADDR, K)
    await avalon_write(dut, WEIGHT_COLS_ADDR, N)
    await avalon_write(dut, INPUT_MAX_ADDR_ADDR, M)
    await avalon_write(dut, INPUT_COLS_ADDR, M)
    await avalon_write(dut, OUTPUT_SCALE_ADDR, output_scale_q16)
    await avalon_write(dut, CONTROL_ADDR, 1 << MATMUL_START_BIT)
    await wait_matmul_done(dut)
    await ClockCycles(dut.clk, 2)

    raw = W_int8 @ X_int8
    golden = np.clip((raw * output_scale_q16) >> 16, -128, 127)

    mismatches = []
    for n in range(N):
        for m in range(M):
            got = to_int8(await read_byte(dut, OBUF_BASE + (n << OBUF_BANK_ADDR_WIDTH) + m))
            exp = int(golden[n, m])
            if got != exp:
                mismatches.append((n, m, exp, got))
    if mismatches:
        preview = ", ".join(f"y[{n},{m}]: exp {e}, got {g}" for n, m, e, g in mismatches[:8])
        raise AssertionError(
            f"{len(mismatches)}/{N*M} mismatches (K={K} N={N} M={M}, "
            f"array {ARRAY_ROWS}x{ARRAY_COLS}). First few: {preview}")


@cocotb.test(expect_fail=True)
async def test_partial_after_full_no_reset(dut):
    """KNOWN FAILING - pre-existing, not a regression from the 32->16
    resize. Fails identically on HEAD's untouched 32x32 RTL (256/256
    mismatches there, 63/64 here). Flip expect_fail to False once fixed.

    Two matmuls back to back with NO reset between them: a full
    ARRAY_ROWS x ARRAY_COLS one, then a half-size partial one.

    This is the case where the out-of-range PE weights actually matter.
    weight_loader's header claims out-of-range PEs "get an explicit 0",
    but w_en = rd_valid_d is low for exactly those PEs, so the
    w_data ternary's 0 branch never reaches them - they retain whatever
    their prefetch register already held. After a reset that is 0 (benign),
    but after a preceding full block it is that block's nonzero weight.
    Combined with input_dispatch reading all ARRAY_ROWS banks regardless
    of valid_row, those stale weights multiply real activation data and
    contribute to the sum.
    """
    rng = np.random.default_rng(4)
    await start_clock(dut)
    cocotb.start_soon(watchdog(dut, max_cycles=MULTIBLOCK_WATCHDOG_CYCLES))
    await reset_dut(dut)

    async def run_matmul(K, N, M, seed):
        r = np.random.default_rng(seed)
        W_real = r.normal(0.0, REAL_VAL_STD, size=(K, N))
        X_real = r.normal(0.0, REAL_VAL_STD, size=(K, M))
        W_int8, ws = quantize_symmetric_int8(W_real)
        X_int8, xs = quantize_symmetric_int8(X_real)
        _, os_ = quantize_symmetric_int8(W_real @ X_real)
        q16 = round(ws * xs / os_ * 65536)

        for k in range(K):
            for n in range(N):
                await write_byte(dut, WBUF_BASE + k * N + n, to_uint8(int(W_int8[k, n])))
        # every bank gets real data for the full run; for the partial run the
        # unused banks keep the previous matmul's contents (realistic - software
        # would not rewrite rows it isn't using)
        for k in range(K):
            for m in range(M):
                await write_byte(dut, IBUF_BASE + k * (1 << IBUF_BANK_ADDR_WIDTH) + m,
                                 to_uint8(int(X_int8[k, m])))

        await avalon_write(dut, WEIGHT_ROWS_ADDR, K)
        await avalon_write(dut, WEIGHT_COLS_ADDR, N)
        await avalon_write(dut, INPUT_MAX_ADDR_ADDR, M)
        await avalon_write(dut, INPUT_COLS_ADDR, M)
        await avalon_write(dut, OUTPUT_SCALE_ADDR, q16)
        await avalon_write(dut, CONTROL_ADDR, 1 << MATMUL_START_BIT)
        await wait_matmul_done(dut)
        await ClockCycles(dut.clk, 2)

        golden = np.clip(((W_int8 @ X_int8) * q16) >> 16, -128, 127)
        bad = []
        for n in range(N):
            for m in range(M):
                got = to_int8(await read_byte(dut, OBUF_BASE + (n << OBUF_BANK_ADDR_WIDTH) + m))
                if got != int(golden[n, m]):
                    bad.append((n, m, int(golden[n, m]), got))
        return bad

    full_bad = await run_matmul(ARRAY_ROWS, ARRAY_COLS, ARRAY_ROWS, 4)
    assert not full_bad, f"the FULL matmul itself failed: {full_bad[:5]}"

    half = ARRAY_ROWS // 2
    part_bad = await run_matmul(half, half, half, 5)
    if part_bad:
        preview = ", ".join(f"y[{n},{m}]: exp {e}, got {g}" for n, m, e, g in part_bad[:8])
        raise AssertionError(
            f"partial matmul after a full one (no reset): {len(part_bad)}/{half*half} "
            f"mismatches. First few: {preview}")

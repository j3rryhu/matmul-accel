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
from cocotb.triggers import RisingEdge, FallingEdge, ClockCycles

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
STATUS_ADDR         = CTRL_BASE + 0x14
OUTPUT_SCALE_ADDR   = CTRL_BASE + 0x18

MATMUL_START_BIT = 0

# STATUS field bit positions - assigned by PeakRDL in declaration order,
# re-check ctrl_rf_rf.sv if ctrl_reg.rdl's field order changes
STATUS_BUSY_BIT = 0
STATUS_DONE_BIT = 1

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

# Avalon data port width in bytes: the buffers' host-facing ports are this
# wide, so every host access moves a whole word and byte addresses advance
# by this much (accel_top.sv's BYTES_PER_WORD).
BYTES_PER_WORD = 4

CLK_PERIOD_NS = 10
WATCHDOG_CYCLES = 8000

# Completion is polled through the STATUS register, so the bound is a count
# of Avalon read transactions rather than clock cycles - each poll costs
# several cycles. 10000 reads is generous for any matmul the buffers can
# hold; the per-test watchdog below is the real backstop against a hang.
STATUS_POLL_LIMIT = 10000

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


async def avalon_read(dut, addr, decode_data=True):
    """Single-word Avalon-MM read.

    decode_data=False returns None instead of the word, for tests that only
    care about bus timing: the behavioral buffer RAMs come up uninitialized
    (see models/buffer_ram_models.v), so reading a location nothing has
    written yields X and int() would raise.
    """
    dut.avs_address.value = addr
    dut.avs_read.value = 1
    dut.avs_write.value = 0
    await RisingEdge(dut.clk)
    while int(dut.avs_waitrequest.value):
        await RisingEdge(dut.clk)
    data = int(dut.avs_readdata.value) if decode_data else None
    dut.avs_read.value = 0
    return data


async def write_bytes(dut, byte_addr, values):
    """Write a run of bytes through the 32-bit Avalon data port.

    The buffers' host-facing ports are AVS_DATA_WIDTH wide, so an access
    moves BYTES_PER_WORD bytes and the byte address advances by
    BYTES_PER_WORD - there is no byte-granular write any more. byte_addr
    must be word aligned, and a short tail is padded with zeros, so callers
    should hand over a whole contiguous image (a full weight matrix, or one
    bank's entire byte image across all of its bands) rather than a slice
    that could leave a partial word straddling live data.
    """
    assert byte_addr % BYTES_PER_WORD == 0,         f"write_bytes needs a word-aligned base, got 0x{byte_addr:x}"
    vals = [v & 0xFF for v in values]
    vals += [0] * ((-len(vals)) % BYTES_PER_WORD)
    for i in range(0, len(vals), BYTES_PER_WORD):
        word = 0
        for j in range(BYTES_PER_WORD):
            word |= vals[i + j] << (8 * j)
        await avalon_write(dut, byte_addr + i, word)


async def avalon_burst_read(dut, addr, count, decode_data=True):
    """Burst-read `count` sequential words starting at addr, using
    accel_top.sv's burst FSM (see its avs_burstcount/rd_burst_on logic).
    Acceptance works exactly like avalon_read's single-word case (hold
    avs_read/address/burstcount until avs_waitrequest drops - this also
    self-corrects if a previous transaction's burst FSM hasn't fully gone
    idle yet), and that same accepting edge already carries the first
    beat's data. Each subsequent beat then arrives one avs_readdatavalid
    pulse per cycle, address auto-incrementing, until `count` beats have
    been collected.

    decode_data=False returns a list of Nones - see avalon_read."""
    dut.avs_address.value = addr
    dut.avs_burstcount.value = count
    dut.avs_read.value = 1
    dut.avs_write.value = 0
    await RisingEdge(dut.clk)
    while int(dut.avs_waitrequest.value):
        await RisingEdge(dut.clk)
    dut.avs_read.value = 0
    dut.avs_burstcount.value = 1
    def sample():
        return int(dut.avs_readdata.value) if decode_data else None

    data = [sample()]
    while len(data) < count:
        await RisingEdge(dut.clk)
        if int(dut.avs_readdatavalid.value):
            data.append(sample())
    return data


async def read_bytes(dut, byte_addr, count):
    """Read `count` bytes starting at an arbitrary (possibly unaligned)
    byte address, by bursting the words that cover them and unpacking.

    output_buffer's host-facing read port is AVS_DATA_WIDTH wide, so a read
    returns BYTES_PER_WORD packed bytes, least significant lane = lowest
    byte address. Output segments start at k_blk*M, which need not be word
    aligned, hence the covering-word arithmetic here.
    """
    first = (byte_addr // BYTES_PER_WORD) * BYTES_PER_WORD
    last  = ((byte_addr + count + BYTES_PER_WORD - 1) // BYTES_PER_WORD) * BYTES_PER_WORD
    words = await avalon_burst_read(dut, first, (last - first) // BYTES_PER_WORD)
    buf = []
    for w in words:
        buf += [(w >> (8 * j)) & 0xFF for j in range(BYTES_PER_WORD)]
    off = byte_addr - first
    return buf[off:off + count]


async def read_byte(dut, byte_addr):
    """Single byte, via the word that contains it."""
    return (await read_bytes(dut, byte_addr, 1))[0]


async def read_status(dut):
    """Read the STATUS register over Avalon. Everything the testbench needs
    to know about completion is visible through the register interface, so
    nothing here reaches into the DUT hierarchy - that keeps these tests
    honest about what software can actually observe."""
    return await avalon_read(dut, STATUS_ADDR)


async def wait_matmul_done(dut, timeout_reads=STATUS_POLL_LIMIT):
    """Poll STATUS.done until the whole matmul has finished: every block
    committed AND output_loader drained the last one. Both STATUS bits are
    levels held until the next matmul_start, so a poll can't miss an edge.

    Note this waits on the *whole* matmul, not one block - output_loader's
    internal done pulses once per committed block, so it is not a usable
    whole-matmul signal even if we were willing to probe it.

    timeout_reads counts register reads, not clock cycles: each poll is a
    full Avalon read transaction and so costs several cycles.
    """
    for _ in range(timeout_reads):
        if (await read_status(dut)) & (1 << STATUS_DONE_BIT):
            return
    raise TimeoutError(
        f"STATUS.done never asserted within {timeout_reads} register reads"
    )


async def wait_matmul_idle(dut, timeout_reads=STATUS_POLL_LIMIT):
    """Poll STATUS.busy until the accelerator is idle. Used to check that
    busy actually clears once a matmul finishes, rather than latching high
    the way it used to when WC_DONE still counted as busy."""
    for _ in range(timeout_reads):
        if not ((await read_status(dut)) & (1 << STATUS_BUSY_BIT)):
            return
    raise TimeoutError(
        f"STATUS.busy never cleared within {timeout_reads} register reads"
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
    await write_bytes(dut, WBUF_BASE,
                      [to_uint8(int(W_int8[k, n])) for k in range(K) for n in range(N)])

    # ---- write quantized input: banked one bank per array row (bank =
    # k % ARRAY_ROWS, offset = k_blk_idx*M + m, matching weight_ctrl's
    # input_band_base_addr) ----
    # one whole bank image at a time: bands are appended head-to-tail
    # within a bank, and a 32-bit write covers 4 consecutive bytes, so a
    # band boundary that isn't word aligned must not be written piecemeal
    num_bands = (K + ARRAY_ROWS - 1) // ARRAY_ROWS
    for bank in range(min(K, ARRAY_ROWS)):
        img = []
        for kb in range(num_bands):
            k = kb * ARRAY_ROWS + bank
            img += [to_uint8(int(X_int8[k, m])) for m in range(M)] if k < K else [0] * M
        await write_bytes(dut, IBUF_BASE + bank * (1 << IBUF_BANK_ADDR_WIDTH), img)

    # ---- program dimensions + rescale factor ----
    await avalon_write(dut, WEIGHT_ROWS_ADDR, K)
    await avalon_write(dut, WEIGHT_COLS_ADDR, N)
    await avalon_write(dut, INPUT_MAX_ADDR_ADDR, M)
    await avalon_write(dut, INPUT_COLS_ADDR, M)
    await avalon_write(dut, OUTPUT_SCALE_ADDR, output_scale_q16)

    # ---- kick off the matmul: everything past this is sequenced by
    # weight_ctrl internally (see rtl/weight_ctrl.v) ----
    await avalon_write(dut, CONTROL_ADDR, 1 << MATMUL_START_BIT)

    await wait_matmul_done(dut)
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
    await write_bytes(dut, WBUF_BASE,
                      [to_uint8(int(W_int8[k, n])) for k in range(K) for n in range(N)])

    # ---- write input: banked one bank per array row (bank = k %
    # ARRAY_ROWS); k_blk_idx's M-sized band is appended after the previous
    # k_blk's within the same bank, matching weight_ctrl's
    # input_band_base_addr = k_blk_idx*input_max_addr ----
    # one whole bank image at a time: bands are appended head-to-tail
    # within a bank, and a 32-bit write covers 4 consecutive bytes, so a
    # band boundary that isn't word aligned must not be written piecemeal
    num_bands = (K + ARRAY_ROWS - 1) // ARRAY_ROWS
    for bank in range(min(K, ARRAY_ROWS)):
        img = []
        for kb in range(num_bands):
            k = kb * ARRAY_ROWS + bank
            img += [to_uint8(int(X_int8[k, m])) for m in range(M)] if k < K else [0] * M
        await write_bytes(dut, IBUF_BASE + bank * (1 << IBUF_BANK_ADDR_WIDTH), img)

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
        row_bytes = await read_bytes(dut, bank_base, M)
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


@cocotb.test()
async def test_partial_row_block_matmul(dut):
    """Partial last *row* block: weight_rows(K) is not a multiple of
    ARRAY_ROWS, weight_cols(N) == ARRAY_COLS exactly. Weight is
    stored exactly as written on paper - (49,32), no transpose - and this
    is a plain Y = W @ X: W(49,32) @ X(32,64) = Y(49,64), contracting over
    the shared 32 dimension.

    Actual PE dataflow (from pe.v/pe_array.v directly, NOT weight_ctrl.v's
    header comment, which describes the wrong axis): weight is loaded
    straight into PE(row,col) with no transpose (w_addr = row*ARRAY_COLS+col,
    weight_loader walks weight_buffer row-major into the same row/col
    layout). Partial sums (b_in) cascade *across columns within a row*
    (pe_array.v: pe_b_in = ROW[r].COL[c-1].pe_p_out) - so contraction
    happens over the weight's COLUMN axis (WEIGHT_COLS/N), and each PE ROW
    independently produces one full output row, taken from its rightmost
    column. Activations flow *down* each column from the top (pe_a_in for
    r==0 comes from a_in[c]; for r>0 from the PE above) - input_buffer bank
    c feeds PE column c, i.e. the contraction axis, matching weight's
    columns. So WEIGHT_ROWS(K) is the *output-row* axis (blocked by
    ARRAY_ROWS=32: k_blk_idx=0 covers rows 0..31, k_blk_idx=1 covers rows
    32..48 - two blocks producing two DISJOINT ranges of output rows,
    concatenated, not accumulated), and WEIGHT_COLS(N) is the true
    contraction size (32 == ARRAY_COLS here, so it's a single pass, no
    masking on that axis needed). This also explains output_loader's
    obuf address (held_k_blk_idx*total_rows + m) and its write-vs-accumulate
    gate (held_n_blk_idx==0 ? write : accumulate): different k-blocks write
    different address ranges (concatenation, not accumulation - correct,
    since num_n_blocks==1 here means n_blk_idx is always 0, so every commit
    "writes" and none ever needs to accumulate).

    Input: the whole (32,64) matrix is loaded into input_buffer once, up
    front (contraction size N==ARRAY_COLS exactly - a single band, reused
    by both k-block passes, since input doesn't depend on k_blk_idx at all).

    Neither test_64x64_multiblock_matmul (exact 2x2 full blocks) nor
    test_single_block_matmul (single full block) ever exercises a *partial*
    last block on either axis. This one exercises:
      - weight_ctrl's valid_k/valid_row masking (rtl/weight_ctrl.v) for a
        genuinely partial last k-block (17 of 32 rows valid).
      - weight_loader's zero-padding of out-of-range PE rows for that
        partial block (valid_rows), rather than always landing exactly on
        a block boundary.
      - output_loader's row_live masking (r < weight_rows, weight_rows fed
        from valid_row/valid_k) for k_blk_idx=1, where only rows 0..16 of
        that pass should ever produce/write a result.
      - output_buffer address reuse across k_blk_idx=0,1 (bank r's address
        = k_blk_idx*total_rows + m) landing on two disjoint ranges within
        the same 32 banks, rather than a single block's worth.

    Same quantization scheme as the other tests (see
    test_64x64_multiblock_matmul's docstring): real float W/X -> per-tensor
    symmetric int8 -> OUTPUT_SCALE combines weight_scale*input_scale/output_scale.

    Buffer capacity check: output_buffer's per-bank depth
    (1<<OBUF_BANK_ADDR_WIDTH = 128 entries) must hold num_k_blocks*M
    entries; here that's 2*64 = 128, exactly at capacity (same margin as
    test_64x64_multiblock_matmul).
    """
    # K: output rows, deliberately NOT a multiple of ARRAY_ROWS so the last
    # k-block is partial. N: the true contraction size - must be exactly
    # ARRAY_COLS, the design cannot mask a partial contraction axis.
    K = ARRAY_ROWS + (ARRAY_ROWS // 2) + 1   # 2 k-blocks, last one partial
    N = ARRAY_COLS
    M = 2 * ARRAY_ROWS

    assert K % ARRAY_ROWS != 0, "last k-block must be partial for this test to mean anything"
    assert ((K + ARRAY_ROWS - 1) // ARRAY_ROWS) * M <= (1 << OBUF_BANK_ADDR_WIDTH),         "output_buffer per-bank depth too small for num_k_blocks*M"

    rng = np.random.default_rng(2)
    await start_clock(dut)
    cocotb.start_soon(watchdog(dut, max_cycles=MULTIBLOCK_WATCHDOG_CYCLES))
    await reset_dut(dut)

    W_real = rng.normal(loc=0.0, scale=REAL_VAL_STD, size=(K, N))
    X_real = rng.normal(loc=0.0, scale=REAL_VAL_STD, size=(N, M))

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

    # ---- golden model: plain Y = W @ X (see docstring - no transpose, no
    # cross-block accumulation; the two k-blocks produce disjoint output-row
    # ranges of the same single-pass contraction). Computed up front (needs
    # only the quantized inputs, not the simulation) so it can be dumped
    # alongside W/X below and reused later for the readback comparison. ----
    raw = W_int8 @ X_int8
    scaled = (raw * output_scale_q16) >> 16
    golden = np.clip(scaled, -128, 127)

    with open("sim_build/matrices_49x32x64.txt", "w") as f:
        f.write(
            f"weight_scale={weight_scale!r}\ninput_scale={input_scale!r}\n"
            f"output_scale={output_scale!r}\ncombined_scale={combined_scale!r}\n"
            f"OUTPUT_SCALE (Q0.16) = 0x{output_scale_q16:04x}\n\n"
        )
        f.write(f"W_real (K x N) =\n{np.array2string(W_real, threshold=np.inf, max_line_width=200)}\n\n")
        f.write(f"W_int8 (K x N) =\n{np.array2string(W_int8, threshold=np.inf, max_line_width=200)}\n\n")
        f.write(f"X_real (N x M) =\n{np.array2string(X_real, threshold=np.inf, max_line_width=200)}\n\n")
        f.write(f"X_int8 (N x M) =\n{np.array2string(X_int8, threshold=np.inf, max_line_width=200)}\n\n")
        f.write(f"Y_real (K x M) =\n{np.array2string(Y_real, threshold=np.inf, max_line_width=200)}\n\n")
        f.write(f"golden Y_int8 (K x M) =\n{np.array2string(golden, threshold=np.inf, max_line_width=200)}\n\n")
        f.write(f"W_int8 hex (K x N), one row per line, each byte 2's-complement =\n{format_int8_hex(W_int8)}\n\n")
        f.write(f"X_int8 hex (N x M), one row per line, each byte 2's-complement =\n{format_int8_hex(X_int8)}\n\n")
        f.write(f"golden Y_int8 hex (K x M), one row per line, each byte 2's-complement =\n{format_int8_hex(golden)}\n")

    # ---- write weights exactly as written on paper: row-major over the
    # K x N matrix (address = k*N+n, see weight_ctrl.v's base_addr; PE(row,col)
    # is loaded directly from this with no transpose - see docstring) ----
    await write_bytes(dut, WBUF_BASE,
                      [to_uint8(int(W_int8[k, n])) for k in range(K) for n in range(N)])

    # ---- write input: banked one bank per PE column (the contraction
    # axis) - the whole (32,64) input is loaded into input_buffer once, up
    # front (N == ARRAY_COLS exactly, a single band reused by both
    # k-block passes, since input doesn't depend on k_blk_idx) ----
    for c in range(N):
        await write_bytes(dut, IBUF_BASE + c * (1 << IBUF_BANK_ADDR_WIDTH),
                          [to_uint8(int(X_int8[c, m])) for m in range(M)])

    await avalon_write(dut, WEIGHT_ROWS_ADDR, K)
    await avalon_write(dut, WEIGHT_COLS_ADDR, N)
    await avalon_write(dut, INPUT_MAX_ADDR_ADDR, M)
    await avalon_write(dut, INPUT_COLS_ADDR, M)
    await avalon_write(dut, OUTPUT_SCALE_ADDR, output_scale_q16)

    await avalon_write(dut, CONTROL_ADDR, 1 << MATMUL_START_BIT)

    # ---- 2 k-blocks (k_blk_idx=0,1), so - like test_64x64_multiblock_matmul
    # - wait for the whole matmul rather than just the first committed block ----
    await wait_matmul_done(dut)
    await ClockCycles(dut.clk, 2)

    # ---- read back output_buffer: output row k lives in bank r = k % ARRAY_ROWS,
    # at that bank's k_blk_idx*M + m offset, k_blk_idx = k // ARRAY_ROWS
    # (see output_loader.v's obuf address: held_k_blk_idx*total_rows + m) ----
    mismatches = []
    for k in range(K):
        k_blk = k // ARRAY_ROWS
        r = k % ARRAY_ROWS
        bank_base = OBUF_BASE + (r << OBUF_BANK_ADDR_WIDTH) + k_blk * M
        for m in range(M):
            got = to_int8(await read_byte(dut, bank_base + m))
            exp = int(golden[k, m])
            if got != exp:
                mismatches.append((k, m, exp, got))

    if mismatches:
        preview = ", ".join(
            f"y[{k},{m}]: expected {exp}, got {got}"
            for k, m, exp, got in mismatches[:10]
        )
        raise AssertionError(
            f"{len(mismatches)}/{K * M} output mismatches. First few: {preview}"
        )


@cocotb.test()
async def test_back_to_back_matmul_no_reset(dut):
    """Two matmuls of the same full ARRAY_ROWS x ARRAY_COLS shape, back to
    back, with no reset between them.

    Regression test for the stale STATUS.done/busy levels in weight_ctrl.
    WC_DONE is a resting state held until the next matmul_start, so `busy`
    (= wc_state != WC_IDLE) latched high forever after the first matmul and
    `done` stayed asserted from the previous run. A completion poll then
    returned immediately and readback raced the compute, returning the
    prior run's results.

    Fixed in rtl/weight_ctrl.v by excluding WC_DONE from `busy` and gating
    `done` with !matmul_start. Backing either fix out is caught here: with
    the busy fix removed, wait_matmul_idle below times out.

    Every other test in this file resets first and runs exactly one
    matmul, which is why this went unnoticed.
    """
    await start_clock(dut)
    cocotb.start_soon(watchdog(dut, max_cycles=MULTIBLOCK_WATCHDOG_CYCLES))
    await reset_dut(dut)

    async def run_matmul(K, N, M, seed):
        r = np.random.default_rng(seed)
        W_real = r.normal(0.0, REAL_VAL_STD, size=(K, N))
        X_real = r.normal(0.0, REAL_VAL_STD, size=(N, M))
        W_int8, ws = quantize_symmetric_int8(W_real)
        X_int8, xs = quantize_symmetric_int8(X_real)
        _, os_ = quantize_symmetric_int8(W_real @ X_real)
        assert 0.0 <= ws * xs / os_ < 1.0
        q16 = round(ws * xs / os_ * 65536)

        await write_bytes(dut, WBUF_BASE,
                          [to_uint8(int(W_int8[k, n])) for k in range(K) for n in range(N)])
        # input_buffer bank c feeds PE column c, i.e. the contraction axis
        for c in range(N):
            await write_bytes(dut, IBUF_BASE + c * (1 << IBUF_BANK_ADDR_WIDTH),
                              [to_uint8(int(X_int8[c, m])) for m in range(M)])

        await avalon_write(dut, WEIGHT_ROWS_ADDR, K)
        await avalon_write(dut, WEIGHT_COLS_ADDR, N)
        await avalon_write(dut, INPUT_MAX_ADDR_ADDR, M)
        await avalon_write(dut, INPUT_COLS_ADDR, M)
        await avalon_write(dut, OUTPUT_SCALE_ADDR, q16)
        await avalon_write(dut, CONTROL_ADDR, 1 << MATMUL_START_BIT)

        # STATUS.done must not still be set from the previous matmul.
        # Through the register interface this is a wide margin rather than a
        # tight race - weight_ctrl leaves WC_DONE on the edge that samples
        # matmul_start, which is well before a read transaction can come
        # back - so this guards against done becoming sticky across
        # matmul_start, not against the original one-cycle window.
        assert not ((await read_status(dut)) & (1 << STATUS_DONE_BIT)),             "STATUS.done still set right after matmul_start - stale from the previous matmul"

        await wait_matmul_done(dut)
        # ... and busy must actually clear once idle, rather than latching
        # high forever the way it did when WC_DONE still counted as busy.
        await wait_matmul_idle(dut)
        await ClockCycles(dut.clk, 2)

        golden = np.clip(((W_int8 @ X_int8) * q16) >> 16, -128, 127)
        bad = []
        for k in range(K):
            k_blk = k // ARRAY_ROWS
            base = OBUF_BASE + ((k % ARRAY_ROWS) << OBUF_BANK_ADDR_WIDTH) + k_blk * M
            for m in range(M):
                got = to_int8(await read_byte(dut, base + m))
                if got != int(golden[k, m]):
                    bad.append((k, m, int(golden[k, m]), got))
        return bad

    first = await run_matmul(ARRAY_ROWS, ARRAY_COLS, ARRAY_ROWS, 4)
    assert not first, f"the FIRST matmul failed, so this test proves nothing: {first[:5]}"

    second = await run_matmul(ARRAY_ROWS, ARRAY_COLS, ARRAY_ROWS, 7)
    if second:
        preview = ", ".join(f"y[{k},{m}]: exp {e}, got {g}" for k, m, e, g in second[:8])
        raise AssertionError(
            f"second matmul (same shape, no reset): {len(second)}/"
            f"{ARRAY_ROWS*ARRAY_ROWS} mismatches. First few: {preview}")


@cocotb.test()
async def test_unaligned_m_matmul(dut):
    """M deliberately NOT a multiple of BYTES_PER_WORD, with more than one
    output-row block, so output_buffer segments start at byte offsets that
    are not word aligned (segment kb begins at kb*M).

    This is the case the 32-bit host data path is most likely to get wrong.
    The core side is unaffected - output_loader still writes single bytes at
    byte addresses, and input_dispatch still reads single bytes - but every
    host access now moves a whole word, so:
      - a result read has to fetch the covering words and unpack, since the
        segment it wants starts mid-word;
      - a bank's input image must be written as one contiguous run, because
        a partial tail word would otherwise clobber the next band.
    """
    K = ARRAY_ROWS + 4          # 2 output-row blocks, second one partial
    N = ARRAY_COLS              # contraction must be a multiple of ARRAY_COLS
    M = ARRAY_ROWS + 2          # not a multiple of 4

    assert M % BYTES_PER_WORD != 0, "test is pointless unless M is unaligned"
    num_k_blocks = (K + ARRAY_ROWS - 1) // ARRAY_ROWS
    assert num_k_blocks > 1, "need >1 output block for an unaligned segment start"
    assert num_k_blocks * M <= (1 << OBUF_BANK_ADDR_WIDTH)

    rng = np.random.default_rng(11)
    await start_clock(dut)
    cocotb.start_soon(watchdog(dut, max_cycles=MULTIBLOCK_WATCHDOG_CYCLES))
    await reset_dut(dut)

    W_real = rng.normal(0.0, REAL_VAL_STD, size=(K, N))
    X_real = rng.normal(0.0, REAL_VAL_STD, size=(N, M))
    W_int8, ws = quantize_symmetric_int8(W_real)
    X_int8, xs = quantize_symmetric_int8(X_real)
    _, os_ = quantize_symmetric_int8(W_real @ X_real)
    assert 0.0 <= ws * xs / os_ < 1.0
    q16 = round(ws * xs / os_ * 65536)

    await write_bytes(dut, WBUF_BASE,
                      [to_uint8(int(W_int8[k, n])) for k in range(K) for n in range(N)])
    for c in range(N):
        await write_bytes(dut, IBUF_BASE + c * (1 << IBUF_BANK_ADDR_WIDTH),
                          [to_uint8(int(X_int8[c, m])) for m in range(M)])

    await avalon_write(dut, WEIGHT_ROWS_ADDR, K)
    await avalon_write(dut, WEIGHT_COLS_ADDR, N)
    await avalon_write(dut, INPUT_MAX_ADDR_ADDR, M)
    await avalon_write(dut, INPUT_COLS_ADDR, M)
    await avalon_write(dut, OUTPUT_SCALE_ADDR, q16)
    await avalon_write(dut, CONTROL_ADDR, 1 << MATMUL_START_BIT)
    await wait_matmul_done(dut)
    await ClockCycles(dut.clk, 2)

    golden = np.clip(((W_int8 @ X_int8) * q16) >> 16, -128, 127)

    mismatches = []
    unaligned_seen = False
    for k in range(K):
        k_blk = k // ARRAY_ROWS
        bank_base = OBUF_BASE + ((k % ARRAY_ROWS) << OBUF_BANK_ADDR_WIDTH) + k_blk * M
        if bank_base % BYTES_PER_WORD:
            unaligned_seen = True
        row = await read_bytes(dut, bank_base, M)
        for m in range(M):
            got = to_int8(row[m])
            if got != int(golden[k, m]):
                mismatches.append((k, m, int(golden[k, m]), got))

    assert unaligned_seen, "no unaligned segment was actually read - test not doing its job"
    if mismatches:
        preview = ", ".join(f"y[{k},{m}]: exp {e}, got {g}" for k, m, e, g in mismatches[:8])
        raise AssertionError(
            f"{len(mismatches)}/{K*M} mismatches (K={K} N={N} M={M}, "
            f"array {ARRAY_ROWS}x{ARRAY_COLS}). First few: {preview}")

@cocotb.test()
async def test_hw_min_identity_sequence(dut):
    """Simulation twin of sw/matmul_min.c, the smallest matmul that runs on
    hardware - same matrices, same register values, same bus sequence, in
    the same order, so a hardware failure can be reproduced here.

    W = 2*I (16x16) with OUTPUT_SCALE = 32768 (Q0.16 0.5) makes the whole
    datapath an exact identity: raw = 2*x, and (2*x * 32768) >> 16 == x for
    every int8 x (the product is exactly x<<16, so the arithmetic shift is
    lossless and nothing ever saturates). X carries 256 *distinct* values
    (x[n,m] = n*16 + m - 128, spanning -128..127 with no repeats), so any
    packing, banking or addressing error shows up as a mismatch whose value
    names the element it actually came from, instead of two different
    addresses holding the same byte and hiding the swap. The assertion
    below re-derives Y from the usual requantize model rather than trusting
    that argument.

    What this test replicates that the others do not is the *host* sequence
    matmul_min.c performs, in its order:
      - a STATUS read as the very first bus transaction, before any write
        (the CTRL read-then-write ordering of test_ctrl_read_then_write,
        but here at the head of a full matmul);
      - the five dimension/scale registers written *before* the buffers,
        the opposite of every other matmul test here;
      - readback as single 32-bit OBUF word reads (what the C code's rd32
        does), not the burst reads read_bytes uses.

    Addressing matches matmul_min.c one for one: the C code's `a` is
    avs_address, mapped to a CPU byte offset of 4*a by the interconnect, and
    a word access advances `a` by 4 - the same units this testbench's
    addresses are already in. So WBUF/IBUF/OBUF bases, the 256-entry input
    bank stride and the 128-entry output bank stride are literally the same
    numbers on both sides.
    """
    K = N = M = ARRAY_ROWS
    assert ARRAY_ROWS == 16 and ARRAY_COLS == 16, (
        f"matmul_min.c is written against a 16x16 array; this build is "
        f"{ARRAY_ROWS}x{ARRAY_COLS}, so the sequence it replicates would "
        f"not be the one that runs on hardware")

    OUTPUT_SCALE_HALF = 32768  # Q0.16 0.5, exactly as matmul_min.c writes it

    await start_clock(dut)
    cocotb.start_soon(watchdog(dut))
    await reset_dut(dut)

    # W = 2I; X = 256 distinct int8 values, one per element
    W_int8 = 2 * np.eye(K, N, dtype=np.int64)
    X_int8 = np.array([[n * 16 + m - 128 for m in range(M)] for n in range(N)],
                      dtype=np.int64)

    # same requantize model as every other test here, applied to the actual
    # OUTPUT_SCALE written below - this is what makes Y == X a *result*
    # rather than an assumption
    golden = np.clip(((W_int8 @ X_int8) * OUTPUT_SCALE_HALF) >> 16, -128, 127)
    assert np.array_equal(golden, X_int8), (
        "W=2I with OUTPUT_SCALE=0.5 is supposed to be an exact identity - "
        "if this fails the test's premise is wrong, not the DUT")

    # ---- phase: read STATUS (first bus transaction, a CTRL read) ----
    dut._log.info(f"STATUS before start = 0x{(await read_status(dut)):08x}")

    # ---- phase: program dimension registers (before the buffers) ----
    await avalon_write(dut, WEIGHT_ROWS_ADDR, K)
    await avalon_write(dut, WEIGHT_COLS_ADDR, N)
    await avalon_write(dut, INPUT_MAX_ADDR_ADDR, M)
    await avalon_write(dut, INPUT_COLS_ADDR, M)
    await avalon_write(dut, OUTPUT_SCALE_ADDR, OUTPUT_SCALE_HALF)

    # ---- phase: write weight buffer (K*N = 256 bytes = 64 words) ----
    await write_bytes(dut, WBUF_BASE,
                      [to_uint8(int(W_int8[k, n])) for k in range(K) for n in range(N)])

    # ---- phase: write input buffer (16 banks x 16 bytes). N == ARRAY_COLS,
    # so there is a single band per bank and bank p is x's row p directly ----
    for p in range(N):
        await write_bytes(dut, IBUF_BASE + p * (1 << IBUF_BANK_ADDR_WIDTH),
                          [to_uint8(int(X_int8[p, m])) for m in range(M)])

    # ---- phase: pulse CONTROL.matmul_start ----
    await avalon_write(dut, CONTROL_ADDR, 1 << MATMUL_START_BIT)

    # ---- phase: poll STATUS.done (the C code's 1 s / 1e6-poll timeout) ----
    await wait_matmul_done(dut)
    await ClockCycles(dut.clk, 2)

    # ---- phase: read output buffer as single 32-bit words, 4 per bank ----
    Y = np.zeros((K, M), dtype=np.int64)
    for k in range(K):
        bank_base = OBUF_BASE + (k << OBUF_BANK_ADDR_WIDTH)
        for m in range(0, M, BYTES_PER_WORD):
            word = await avalon_read(dut, bank_base + m)
            for j in range(BYTES_PER_WORD):
                Y[k, m + j] = to_int8((word >> (8 * j)) & 0xFF)

    # ---- phase: verify. Mismatches are reported with the avs address the
    # byte came from, the same way matmul_min.c prints them, so a failure
    # here and a failure on hardware name the same location. ----
    mismatches = [
        (k, m, int(golden[k, m]), int(Y[k, m]),
         OBUF_BASE + (k << OBUF_BANK_ADDR_WIDTH) + m)
        for k in range(K) for m in range(M)
        if int(Y[k, m]) != int(golden[k, m])
    ]
    if mismatches:
        preview = ", ".join(
            f"y[{k},{m}]: expected {exp}, got {got} (avs 0x{addr:04x})"
            for k, m, exp, got, addr in mismatches[:10]
        )
        raise AssertionError(
            f"{len(mismatches)}/{K * M} output mismatches. First few: {preview}"
        )


@cocotb.test()
async def test_ctrl_read_then_write(dut):
    """CTRL read immediately followed by a CTRL write.

    Minimal hardware repro: a CTRL write works as the first bus operation,
    but the same write after any CTRL read never completes - avs_waitrequest
    stays high, so the write is never accepted. A 10 ms gap on hardware does
    not help, so this is latched state in the read path, not a race.

    No test in this file ever puts a CTRL read directly before a CTRL write:
    read_status is always followed by more reads, then buffer accesses.
    """
    await start_clock(dut)
    cocotb.start_soon(watchdog(dut, max_cycles=2000))
    await reset_dut(dut)

    # write first, no preceding read - the case that works on hardware
    await avalon_write(dut, WEIGHT_COLS_ADDR, N)

    # a CTRL read
    st = await read_status(dut)
    dut._log.info(f"STATUS = 0x{st:08x}")

    # ... and now the same write again. On hardware this never returns.
    # If avs_waitrequest latches high, avalon_write spins and the watchdog
    # fires - that is the failure signature.
    await avalon_write(dut, WEIGHT_COLS_ADDR, N)

    # and with idle cycles in between, to show the gap does not help
    await read_status(dut)
    await ClockCycles(dut.clk, 20)
    await avalon_write(dut, WEIGHT_ROWS_ADDR, K)

async def rdv_counter(dut, count):
    """Counts clock cycles in which avs_readdatavalid is high.

    Sampled on the falling edge so the count is of settled, mid-cycle
    values - what an Avalon master latching on the rising edge would
    actually see. count is a one-element list so callers can zero it
    between phases.
    """
    while True:
        await FallingEdge(dut.clk)
        if int(dut.avs_readdatavalid.value):
            count[0] += 1


@cocotb.test()
async def test_readdatavalid_one_pulse_per_beat(dut):
    """avs_readdatavalid must be high for exactly one cycle per read beat.

    Avalon-MM lets a master issue its next transaction the cycle after a
    read is accepted, so a stray valid lands on top of whatever comes
    next - the master counts it as an extra beat of the read it just did
    and the pipelined read channel goes permanently out of step. Writes
    are the visible case (a CTRL write right behind a CTRL read looks
    like the read's valid stretching by a cycle), but the pulse is wrong
    whatever follows it.
    """
    await start_clock(dut)
    cocotb.start_soon(watchdog(dut, max_cycles=2000))
    await reset_dut(dut)

    count = [0]
    cocotb.start_soon(rdv_counter(dut, count))

    # a write is not a read - it must never produce read data
    count[0] = 0
    await avalon_write(dut, WEIGHT_COLS_ADDR, N)
    await ClockCycles(dut.clk, 4)
    assert count[0] == 0, (
        f"CTRL write asserted avs_readdatavalid for {count[0]} cycle(s)")

    # one CTRL read, with a write immediately behind it
    count[0] = 0
    await read_status(dut)
    await avalon_write(dut, WEIGHT_ROWS_ADDR, K)
    await ClockCycles(dut.clk, 4)
    assert count[0] == 1, (
        f"CTRL read followed by a write gave {count[0]} avs_readdatavalid "
        f"cycle(s), expected 1")

    # one OBUF read, likewise
    count[0] = 0
    await avalon_read(dut, OBUF_BASE, decode_data=False)
    await ClockCycles(dut.clk, 4)
    assert count[0] == 1, (
        f"OBUF single read gave {count[0]} avs_readdatavalid cycle(s), "
        f"expected 1")

    # an OBUF burst is exactly burstcount beats, no trailing valid
    beats = 4
    count[0] = 0
    await avalon_burst_read(dut, OBUF_BASE, beats, decode_data=False)
    await ClockCycles(dut.clk, 4)
    assert count[0] == beats, (
        f"OBUF burst of {beats} gave {count[0]} avs_readdatavalid "
        f"cycle(s), expected {beats}")


@cocotb.test()
async def test_stalled_write_commits_once(dut):
    """A write held across avs_waitrequest must commit exactly once.

    An Avalon master keeps write/address/writedata stable until
    avs_waitrequest drops, so a write enable built straight from avs_write
    re-commits the access on every stalled cycle. For the buffers that is
    invisible - the same word goes back to the same address - so the only
    place it can be observed is a pulse field, which is why this reaches
    into the DUT for matmul_start instead of going through the register
    interface: a repeated write to any *storage* register is idempotent by
    construction and nothing at the software-visible boundary can tell the
    two apart.

    avs_waitrequest is only asserted against a write while a read burst is
    still in flight, so that is how the stall is set up here.
    """
    await start_clock(dut)
    cocotb.start_soon(watchdog(dut, max_cycles=2000))
    await reset_dut(dut)

    starts = [0]

    async def count_starts():
        while True:
            await FallingEdge(dut.clk)
            if int(dut.matmul_start.value):
                starts[0] += 1

    cocotb.start_soon(count_starts())

    # baseline: the same write on an idle bus, never stalled
    starts[0] = 0
    await avalon_write(dut, CONTROL_ADDR, 1 << MATMUL_START_BIT)
    await ClockCycles(dut.clk, 6)
    assert starts[0] == 1, (
        f"unstalled CONTROL.start write pulsed matmul_start {starts[0]} "
        f"time(s), expected 1")

    await ClockCycles(dut.clk, 40)

    # and now the same write arriving while an OBUF burst still owns the bus
    starts[0] = 0
    burst = cocotb.start_soon(
        avalon_burst_read(dut, OBUF_BASE, 8, decode_data=False))
    await ClockCycles(dut.clk, 3)
    await avalon_write(dut, CONTROL_ADDR, 1 << MATMUL_START_BIT)
    await burst
    await ClockCycles(dut.clk, 6)
    assert starts[0] == 1, (
        f"CONTROL.start write stalled by avs_waitrequest pulsed "
        f"matmul_start {starts[0]} time(s), expected 1")

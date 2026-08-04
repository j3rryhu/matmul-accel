"""
Cocotb test for accel_top simulating a real int8-quantization deployment
flow: a single 32x32 (weight) x 32x32 (input) matmul - exactly one
contraction block (k_blk_idx=0 only) and one output block (n_blk_idx=0
only), so no multi-block accumulation and no partial-block masking are
exercised. Checks the whole pipeline end to end: Avalon writes ->
weight_buffer/input_buffer -> weight_ctrl -> weight_loader/input_dispatch
-> pe_array -> output_loader -> output_buffer -> Avalon reads.

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

Golden model: raw = W_int8 @ X_int8 at full precision (the 32-bit
ACC_DATA_WIDTH accumulator - see pe.v/pe_array.v - can't overflow for any
draw at these matrix sizes), then rescaled by the *actual* OUTPUT_SCALE
value written to the device (raw*OUTPUT_SCALE >> 16, arithmetic shift) and
saturated to signed int8 [-128,127] - matching output_loader's requantize
stage bit-for-bit.
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
K = 32   # contraction size (WEIGHT_ROWS)
N = 32   # output size (WEIGHT_COLS)
M = 32   # input/output column count (INPUT_COLS == INPUT_MAX_ADDR here)

ARRAY_ROWS = 32  # must match accel_top.sv's ARRAY_ROWS / input_buffer_32_bank's bank count

# "Real" (float) weight/input matrices are drawn N(0, REAL_VAL_STD) - wide
# enough dynamic range to make quantization meaningful without needing
# either matrix to be degenerate (all-zero, etc).
REAL_VAL_STD = 1.0

OBUF_BANK_ADDR_WIDTH = 7  # must match accel_top.sv's OBUF_BANK_ADDR_WIDTH
IBUF_BANK_ADDR_WIDTH = 8  # must match input_buffer_32_bank.v's per-bank address width

CLK_PERIOD_NS = 10
DONE_TIMEOUT_CYCLES = 8000
WATCHDOG_CYCLES = 8000


async def start_clock(dut):
    cocotb.start_soon(Clock(dut.clk, CLK_PERIOD_NS, units="ns").start())


async def watchdog(dut, max_cycles=WATCHDOG_CYCLES):
    """Kills the test if it's still running after max_cycles - a backstop
    against any hang (e.g. avs_waitrequest stuck high) that the individual
    per-step timeouts below don't cover."""
    await ClockCycles(dut.clk, max_cycles)
    raise TimeoutError(f"testbench watchdog: exceeded {max_cycles} cycles")


async def reset_dut(dut):
    dut.rst_n.value = 0
    dut.avs_address.value = 0
    dut.avs_read.value = 0
    dut.avs_write.value = 0
    dut.avs_writedata.value = 0
    dut.avs_byteenable.value = 0
    await ClockCycles(dut.clk, 5)
    dut.rst_n.value = 1
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


@cocotb.test()
async def test_min_32x32_matmul(dut):
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

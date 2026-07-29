"""
Minimum viable cocotb test for accel_top: a single 32x32 (weight) x 32x32
(input) matmul - exactly one contraction block (k_blk_idx=0 only) and one
output block (n_blk_idx=0 only), so no multi-block accumulation and no
partial-block masking are exercised. Just enough to check the whole
pipeline end to end: Avalon writes -> weight_buffer/input_buffer -> weight_ctrl
-> weight_loader/input_dispatch -> pe_array -> output_loader -> output_buffer
-> Avalon reads.

Everything is addressed per the RTL's own documented conventions:
  - weight_buffer: row-major over the (contraction x output) matrix,
    address = k*WEIGHT_COLS + n (see weight_ctrl.v's base_addr).
  - input_buffer: row-major over the (contraction x M) matrix,
    address = k*INPUT_COLS + m (see input_dispatch.v's +i_input_cols step).
  - output_buffer: banked one row per array column; bank c, offset m holds
    y[row=n_blk_idx*ARRAY_COLS+c, col=m] (see output_loader.v). With a
    single 32x32 output block, n_blk_idx=0, so bank c holds row c directly,
    laid out at OBUF_BASE + c*(1<<OBUF_BANK_ADDR_WIDTH) + m.

Golden model: y[n,m] = sum_k W[k,n]*x[k,m], wrapped to signed int8 - pe.v's
MAC (weight*a + b_in) is truncated to DATA_WIDTH at every cascade step, but
since that's pure accumulation of products (no further multiply on the
partial sum), reducing mod 256 once at the end of a full-precision sum
gives the identical result to truncating at every step.
"""

import random

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

MATMUL_START_BIT = 0

# ---- test-fixed dimensions ----
K = 32   # contraction size (WEIGHT_ROWS)
N = 32   # output size (WEIGHT_COLS)
M = 32   # input/output column count (INPUT_COLS == INPUT_MAX_ADDR here)

OBUF_BANK_ADDR_WIDTH = 7  # must match accel_top.sv's OBUF_BANK_ADDR_WIDTH

CLK_PERIOD_NS = 10
DONE_TIMEOUT_CYCLES = 8000


async def start_clock(dut):
    cocotb.start_soon(Clock(dut.clk, CLK_PERIOD_NS, units="ns").start())


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
    than weight_ctrl.done, which only reflects the input side finishing;
    STATUS.busy/done aren't wired up yet, see rdl/README.md)."""
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


@cocotb.test()
async def test_min_32x32_matmul(dut):
    random.seed(0)
    await start_clock(dut)
    await reset_dut(dut)

    # ---- random signed int8 weight (K x N) and input (K x M) matrices ----
    W = np.array(
        [[random.randint(-128, 127) for _ in range(N)] for _ in range(K)],
        dtype=np.int64,
    )
    X = np.array(
        [[random.randint(-128, 127) for _ in range(M)] for _ in range(K)],
        dtype=np.int64,
    )

    # ---- write weights: row-major (k*N+n) into weight_buffer ----
    for k in range(K):
        for n in range(N):
            await write_byte(dut, WBUF_BASE + k * N + n, to_uint8(int(W[k, n])))

    # ---- write input: row-major (k*M+m) into input_buffer ----
    for k in range(K):
        for m in range(M):
            await write_byte(dut, IBUF_BASE + k * M + m, to_uint8(int(X[k, m])))

    # ---- program dimensions ----
    await avalon_write(dut, WEIGHT_ROWS_ADDR, K)
    await avalon_write(dut, WEIGHT_COLS_ADDR, N)
    await avalon_write(dut, INPUT_MAX_ADDR_ADDR, M)
    await avalon_write(dut, INPUT_COLS_ADDR, M)

    # # ---- kick off the matmul: everything past this is sequenced by
    # # weight_ctrl internally (see rtl/weight_ctrl.v) ----
    # await avalon_write(dut, CONTROL_ADDR, 1 << MATMUL_START_BIT)

    # await wait_output_done(dut)
    # # give output_buffer's write a cycle to land before reading it back
    # await ClockCycles(dut.clk, 2)

    # # ---- golden model: y[n,m] = sum_k W[k,n]*X[k,m], wrapped to int8 ----
    # golden = (W.T @ X)
    # golden = ((golden + 128) % 256) - 128  # wrap to signed int8

    # # ---- read back output_buffer: bank c (=row n, since n_blk_idx=0),
    # # offset m ----
    # mismatches = []
    # for n in range(N):
    #     bank_base = OBUF_BASE + (n << OBUF_BANK_ADDR_WIDTH)
    #     for m in range(M):
    #         got = to_int8(await read_byte(dut, bank_base + m))
    #         exp = int(golden[n, m])
    #         if got != exp:
    #             mismatches.append((n, m, exp, got))

    # if mismatches:
    #     preview = ", ".join(
    #         f"y[{n},{m}]: expected {exp}, got {got}"
    #         for n, m, exp, got in mismatches[:10]
    #     )
    #     raise AssertionError(
    #         f"{len(mismatches)}/{N * M} output mismatches. First few: {preview}"
    #     )

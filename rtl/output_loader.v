// output_loader : accumulates pe_array's partial sums into output_buffer,
// which is banked one bank per array row (ARRAY_COLS banks total - see
// naming note below) - mirrors input_buffer_32_bank's banking pattern on
// the input side, scaled to ARRAY_COLS so every row that's currently
// producing a real result can be read-modify-written the same cycle it
// arrives, with no elastic buffering or backpressure required.
//
// Naming note: each of pe_array's p_out lanes (and so each bank here)
// corresponds to one physical PE array *row* - p_in[r]/bank r is the same
// axis as input_dispatch's row-r a_en (see pe_array.v/input_dispatch.v),
// not a "column" despite the parameter still being named ARRAY_COLS (kept
// as-is so it matches pe_array's own p_out port width parameter). "row"
// elsewhere below means the *final written-out output matrix's* row (as
// you'd write y out on paper), not weight_ctrl's block indices or a PE
// array axis - watch for both senses in the same sentence, e.g. "bank r
// holds output row n". Bank r always holds output row
// (n_blk_idx*ARRAY_COLS+r) for whichever n_blk_idx is currently committed
// - i.e. each bank is reused across every n_blk_idx pass, and within a
// bank output rows are laid out as rows of the output canvas: canvas row R
// sits in bank R % ARRAY_ROWS at byte offset (R / ARRAY_ROWS)*row_pitch,
// row_pitch being the canvas width padded to a host word (accel_top). This
// matmul's result is one tile of that canvas, starting at canvas row
// tile_row_blk*ARRAY_ROWS and column tile_col_off, so output row
// k = blk*ARRAY_ROWS + r, column m lands at
//   (tile_row_blk + blk)*row_pitch + tile_col_off + m
// in bank r. The host reads the canvas as one row-major image.
//
// Since the contraction dimension may need more than one ARRAY_ROWS-block pass
// (k_blk_idx = 0..num_k_blocks-1) to fully sum, every block's psums must
// be added to whatever's already at that output position - except the
// very first k-block for a given n_blk_idx, whose target position is
// otherwise stale/uninitialized and gets written directly instead.
// weight_ctrl's committed_k_blk_idx/committed_first_k_blk (latched at
// commit, valid for that block's whole compute+drain window) tell us
// which case we're in; this module samples them once, at its own a_en_last
// start edge, and holds them for the rest of that block's drain - by
// construction weight_ctrl's *next* commit (which would advance those
// signals) can't happen until this block's input band has finished
// draining, which is always after this block's a_en_last has already risen,
// so the sampled copies are never overwritten out from under an
// in-progress drain.
//
// Rescale/requantize: p_in arrives at pe_array's full ACC_WIDTH (32-bit)
// accumulator width, but output_buffer stores DATA_WIDTH (int8) results -
// under the quantization scheme this array targets, ACC_WIDTH exists so a
// full contraction-dimension sum can't overflow before this rescale step,
// not so the raw accumulator gets stored. Each arriving psum is multiplied
// by output_scale (Q0.16 fixed point, unsigned, range [0,1), actual scale =
// output_scale/65536) and the product is arithmetic-shifted right by 16 to
// land back in ACC_WIDTH's integer domain, then saturated (not wrapped) to
// DATA_WIDTH's signed range. The k-block accumulate add (old_val +
// scaled psum) is saturated the same way before being written back, since
// that sum can exceed DATA_WIDTH on its own even when each term doesn't.
//
// Read-modify-write per row, as a 4-stage pipeline (the rescale multiply
// is too slow to do combinationally between the RAM read and the write
// back - and a single-cycle ACC_WIDTH x SCALE_WIDTH multiply was itself
// still the critical path, so it is split across two stages, see below):
//   stage 0 (cycle C)  : obuf_rdaddress/rden = cur_addr, and p_in[r] holds
//                        that same position's psum. Both are captured
//                        (addr_d/active_d/psum_d).
//   stage 1 (cycle C+1): output_buffer is a synchronous RAM, so obuf_q[r]
//                        is now the word for the address issued at C -
//                        i.e. obuf_q is only valid in the cycle where
//                        addr_d is its address, which is why the lane
//                        select uses addr_d and the result is registered
//                        (old_val_d) rather than used three cycles later.
//                        The two rescale *partial products* register here
//                        (prod_hi/prod_lo) - see the split note below.
//   stage 2 (cycle C+2): the partial products are shifted back into place
//                        and summed into the full product (scale_product).
//                        old_val_d/addr_2d/active_2d pipeline alongside.
//   stage 3 (cycle C+3): scale_product, old_val_2d and addr_3d all describe
//                        the same position - shift, saturate, accumulate,
//                        and write back.
// Each row's read address advances one position per cycle, so the byte
// being written at C+3 is never the byte being read at C+3 (different
// lanes of the word), and no read-during-write forwarding is needed.
//
// Rescale multiply split (stage 1 -> stage 2): a full ACC_WIDTH x
// SCALE_WIDTH product in one cycle does not make timing in soft logic
// (multstyle="logic", see the DSP note below), so psum is cut into a low
// LO_W-bit unsigned chunk and a high HI_W-bit signed chunk, with
//     psum        = $signed(hi)*2**LO_W + lo
// so that
//     psum*scale  = (($signed(hi)*scale) << LO_W) + (lo*scale)
// Stage 1 computes those two narrow products independently; stage 2 does
// only the shift-and-add. LO_W = SCALE_WIDTH, which for ACC_WIDTH=32 /
// SCALE_WIDTH=16 splits psum evenly and keeps both multiplies at 17x17.
//
// Timing model (from pe.v/pe_array.v):
//   - a_buf/b_buf only update when a_en/b_en pulse; p_out is recomputed
//     every cycle from whatever a_buf/b_buf currently hold. b_en must
//     therefore stay high for the *entire* compute+drain window once
//     started - gapping it would freeze that row's cascade mid-flight
//     and misalign every row still in transit. b_en is one signal
//     broadcast to all rows, not staggered per row.
//   - input_dispatch already staggers row r's a_en by r cycles, and the
//     a-side shift chain adds one more cycle of delay per row hop. Those
//     two skews cancel out along the diagonal, so row r's p_out starts
//     reflecting real (non-reset) data ARRAY_COLS-1+r cycles after row 0's
//     a_en first rises: row 0 goes live first, then one more row every
//     following cycle, all ARRAY_COLS rows live ARRAY_COLS-1 cycles after
//     that.
//   - once a row is live it produces exactly one new valid result per
//     cycle, forever, so each row's captures are independently addressed
//     by its own free-running counter (0..total_rows-1) - no need to
//     realign rows to a shared index.
//
// a_en_last is the *last* row's a_en bit (pe_array a_en[ARRAY_ROWS-1] /
// input_dispatch a_en[ARRAY_ROWS-1]) - input_dispatch staggers a_en by one
// cycle per row, so this bit rises ARRAY_ROWS-1 cycles after row 0's, which
// is exactly the anchor the elapsed>=r liveness test below expects. Only
// its first rising edge after reset/completion is used to anchor timing, so
// later gaps in a_en_last (e.g. input FIFO underrun) don't re-trigger the
// sequence.
//
// DSP: the rescale multiply below is instantiated once per bank
// (ARRAY_ROWS times) at ACC_WIDTH x SCALE_WIDTH, plus a per-bank address
// multiply - enough to exhaust a small device's DSP blocks on its own, on
// top of pe_array's. multstyle="logic" forces all of them into soft logic
// (see pe.v's header note).

// this helps control bias loading as well
//
// Two buffers + final-pass activation (supersedes the per-block int8
// rescale/accumulate described above):
//   - psum_buffer (internal, ACC_WIDTH slots, read+write by this module
//     only) holds each element's raw running sum. Every contraction pass
//     except an element's last reads it (psum_q) and writes old + psum
//     back in stage 1, no rescale. The first pass of an element
//     (n_blk_idx == 0, unless use_prev_out continues raw sums) ignores
//     what it held.
//   - output_buffer (int8 canvas, the one the host reads) gets the final
//     result. The last pass - committed_n_blk_idx == max_n_blk_idx, unless
//     out_raw says a later use_prev_out matmul continues the sum - adds the
//     row's int32 bias (bias_en, in accumulator units), rescales/saturates
//     to int8 through the stage 1-3 pipeline, optionally maps it through
//     the GELU LUT (gelu_en, one more stage), and writes the int8 as one
//     byte at its canvas position - no packing, no alignment rules.
// output_buffer is addressed at the tile's canvas position (pitch
// row_pitch8, bytes); psum_buffer relative to the tile itself - it holds
// one tile, every tile reusing the same slots (pitch row_pitch = INPUT_COLS).
//
// prev_int8: with use_prev_out, the final pass reads the previous int8
// result at this position from output_buffer (stage 0, byte selected in
// stage 1), carries it in old_val_d/old_val_2d to stage 3, and adds it to
// the rescaled value - it is already in output units - with one
// saturation, before GELU. The raw sums start fresh, so any number of
// contraction blocks works: the int8 sits in its own buffer until the
// final pass reads it, then gets overwritten with the new result.

`timescale 1ps/1ps

(* multstyle = "logic" *)
module output_loader #(
    parameter DATA_WIDTH     = 8,    // rescale target width (int8) - the output_buffer entry width
    parameter ACC_WIDTH      = 32,   // must match pe_array's ACC_WIDTH (p_in) - also the psum_buffer slot width
    parameter SCALE_WIDTH    = 16,   // output_scale width - Q0.16 fixed point (unsigned, range [0,1))
    parameter ARRAY_COLS     = 16,
    parameter ARRAY_ROWS     = 16,
    parameter ROW_ADDR_WIDTH = 16,   // output_buffer per-bank WRITE (byte) address width - spans the bank's share of the int8 canvas
    parameter PSUM_ADDR_WIDTH = 10,  // psum_buffer per-bank address width (32-bit slots)
    parameter RD_DATA_WIDTH  = 128,  // output_buffer read port width (wide, shared with Avalon)
    // The output_buffer read port is wide (one word per access) because the
    // Avalon host reads through it; this module only reads it for
    // prev_int8, wants one byte of that word, so it addresses the RAM in
    // words and selects the lane its byte actually lives in. The write port
    // stays DATA_WIDTH: one int8 per bank per cycle.
    parameter LANE_SEL_W     = $clog2(RD_DATA_WIDTH/DATA_WIDTH),
    parameter ROW_RADDR_WIDTH = ROW_ADDR_WIDTH - LANE_SEL_W,
    parameter DIM_WIDTH      = 16,   // width of committed_k_blk_idx (from weight_ctrl)
    parameter COL_SEL_W      = $clog2(ARRAY_COLS+1),
    parameter ROW_SEL_W      = $clog2(ARRAY_ROWS+1),
    parameter BIAS_ADDR_WIDTH = 4,   // bias_buffer per-bank address width (canvas row block)
    parameter GELU_LUT_FILE  = "activation/example/gelu_int8.hex"
)(
    input                                          clock,
    input                                           rst_n,       // active-low

    // ---- pe_array bottom edge ----
    input      [ARRAY_COLS*ACC_WIDTH-1:0]          p_in,        // pe_array p_out
    output     [ARRAY_COLS-1:0]                    b_en,        // pe_array b_en

    // ---- timing anchor + block identity (from weight_ctrl) ----
    input                                           a_en_last,             // pe_array a_en[ARRAY_ROWS-1]
    input      [ROW_ADDR_WIDTH-1:0]                 total_rows,            // M: expected valid results per row, per pass
    input      [DIM_WIDTH-1:0]                      committed_k_blk_idx,   // weight_ctrl: output block this pass belongs to
    input      [DIM_WIDTH-1:0]                      committed_n_blk_idx,
    input                                            committed_first_k_blk, // weight_ctrl: write instead of accumulate
    input      [DIM_WIDTH-1:0]                      max_n_blk_idx,         // last contraction block of this matmul (accel_top latch)
    input                                           out_raw,               // ACT_CFG.out_raw: never finalize, leave raw 32-bit sums
    input                                           prev_int8,             // ACT_CFG.prev_int8: use_prev_out adds onto the previous int8 result
    input                                           bias_en,               // ACT_CFG.bias_en
    input                                           gelu_en,               // ACT_CFG.gelu_en

    // ---- where this matmul's result goes in the canvas (accel_top's TILE_CFG latch) ----
    input      [DIM_WIDTH-1:0]                      tile_row_blk,   // first canvas row / ARRAY_ROWS
    input      [ROW_ADDR_WIDTH-1:0]                 tile_col_off,   // first canvas column (output_buffer bytes)
    input                                           use_prev_out,   // continue what the canvas holds (format: prev_int8)
    input      [PSUM_ADDR_WIDTH:0]                  row_pitch,      // psum row pitch in slots = INPUT_COLS (tile-relative)
    input      [ROW_ADDR_WIDTH:0]                   row_pitch8,     // int8 canvas row pitch in bytes

    // ---- rescale factor (from ctrl_rf, see rdl/ctrl_reg.rdl OUTPUT_SCALE) ----
    input      [SCALE_WIDTH-1:0]                    output_scale,

    // ---- current valid rows and cols for output ----
    input      [COL_SEL_W-1:0]                      weight_cols,
    input      [ROW_SEL_W-1:0]                      weight_rows,

    // ---- psum_buffer read+write ports (raw 32-bit sums), one bank per array row ----
    output     [ARRAY_COLS-1:0]                     psum_rden,
    output     [ARRAY_COLS*PSUM_ADDR_WIDTH-1:0]     psum_rdaddress,
    input      [ARRAY_COLS*ACC_WIDTH-1:0]           psum_q,
    output     [ARRAY_COLS-1:0]                     psum_wren,
    output     [ARRAY_COLS*PSUM_ADDR_WIDTH-1:0]     psum_waddr,
    output     [ARRAY_COLS*ACC_WIDTH-1:0]           psum_wdata,

    // ---- output_buffer read+write ports (int8 results), one bank per array row ----
    output     [ARRAY_COLS-1:0]                     obuf_rden,
    output     [ARRAY_COLS*ROW_RADDR_WIDTH-1:0]     obuf_rdaddress,
    input      [ARRAY_COLS*RD_DATA_WIDTH-1:0]       obuf_q,
    output     [ARRAY_COLS-1:0]                     obuf_wren,
    output     [ARRAY_COLS*ROW_ADDR_WIDTH-1:0]      obuf_waddr,
    output     [ARRAY_COLS*DATA_WIDTH-1:0]          obuf_wdata,

    // ---- bias buffer read, one bank per array row (see bias_buffer_32_bank.v) ----
    output     [ARRAY_COLS-1:0]                     bbuf_rden,
    output     [ARRAY_COLS*BIAS_ADDR_WIDTH-1:0]     bbuf_rdaddress,
    input      [ARRAY_COLS*ACC_WIDTH-1:0]           bbuf_q,        // int32 bias per lane, accumulator units

    output                                          busy,
    output                                          done
);

    localparam ELAPSED_MAX = ARRAY_COLS - 1;
    localparam ELAPSED_W   = $clog2(ELAPSED_MAX + 1);

    // psum split point for the two-stage rescale multiply (see header):
    // LO_W is the unsigned low chunk, HI_W the signed high chunk.
    localparam LO_W = SCALE_WIDTH;
    localparam HI_W = ACC_WIDTH - LO_W;

    // signed DATA_WIDTH saturation bounds (e.g. -128/127 for DATA_WIDTH=8)
    localparam signed [DATA_WIDTH-1:0] DATA_MAX = {1'b0, {(DATA_WIDTH-1){1'b1}}};
    localparam signed [DATA_WIDTH-1:0] DATA_MIN = {1'b1, {(DATA_WIDTH-1){1'b0}}};

    reg                     armed;      // ready to latch the next a_en_last rising edge
    reg                     a_en_last_d;
    reg                     running;
    reg                     running_d;
    reg                     running_2d;
    reg                     running_3d;
    reg                     running_4d;   // covers the GELU stage-4 write (see busy below)
    reg  [ELAPSED_W:0]      elapsed;

    reg  [ROW_ADDR_WIDTH-1:0] row_count [0:ARRAY_COLS-1];
    wire [ARRAY_COLS-1:0]    row_done;

    // ---- sampled once at start_edge, held for this block's whole drain ----
    reg [DIM_WIDTH-1:0] held_k_blk_idx;
    reg [DIM_WIDTH-1:0] held_n_blk_idx;
    reg                 held_first_k_blk;

    // where this pass's output-row block starts in every bank: canvas row
    // block (tile_row_blk + k_blk), column tile_col_off. Computed from the
    // committed index - stable long before the pass starts - and latched
    // with the other held_* values, so the multiply never sits on the
    // per-cycle RAM address path.
    // held_seg_base8 is that canvas position in output_buffer bytes (pitch
    // row_pitch8); held_seg_base is the block's position inside the tile in
    // psum_buffer slots (pitch row_pitch) - psum_buffer holds one tile, so
    // no canvas offset.
    (* multstyle = "logic" *)
    wire [PSUM_ADDR_WIDTH-1:0] committed_seg_base =
        committed_k_blk_idx * row_pitch;
    (* multstyle = "logic" *)
    wire [ROW_ADDR_WIDTH-1:0]  committed_seg_base8 =
        (tile_row_blk + committed_k_blk_idx) * row_pitch8 + tile_col_off;
    reg  [PSUM_ADDR_WIDTH-1:0] held_seg_base;
    reg  [ROW_ADDR_WIDTH-1:0]  held_seg_base8;

    wire start_edge = armed && a_en_last && !a_en_last_d;
    wire pass_start = !running && a_en_last;   // same condition that sets `running` below

    always @(posedge clock) begin
        if (~rst_n)
            a_en_last_d <= 1'b0;
        else
            a_en_last_d <= a_en_last;
    end

    always @(posedge clock) begin
        if (~rst_n) begin
            armed            <= 1'b1;
            running          <= 1'b0;
            elapsed          <= {ELAPSED_W{1'b0}};
            held_k_blk_idx   <= 0;
            held_n_blk_idx   <= 0;
            held_first_k_blk <= 1'b0;
            held_seg_base    <= 0;
            held_seg_base8   <= 0;
            running_d        <= 0;
            running_2d       <= 0;
            running_3d       <= 0;
            running_4d       <= 0;
        end
        else begin
            if (running) begin
                if (elapsed < ELAPSED_MAX)
                    elapsed <= elapsed + 1'b1;

                if (done) begin
                    running <= 1'b0;
                    armed   <= 1'b1;
                    elapsed <= 0;
                end
            end
            else begin
                if (a_en_last) begin
                    armed            <= 1'b0;
                    running          <= 1'b1;
                    elapsed          <= {ELAPSED_W{1'b0}};
                    held_k_blk_idx   <= committed_k_blk_idx;
                    held_n_blk_idx   <= committed_n_blk_idx;
                    held_first_k_blk <= committed_first_k_blk;
                    held_seg_base    <= committed_seg_base;
                    held_seg_base8   <= committed_seg_base8;
                end
            end
            running_d  <= running;
            running_2d <= running_d;
            running_3d <= running_2d;
            running_4d <= running_3d;
        end
    end

    // held high for the whole compute+drain window - never gated per row
    assign b_en = {ARRAY_COLS{running}};

    genvar r;
    generate
        for (r = 0; r < ARRAY_ROWS; r = r + 1) begin : ROW
            wire row_live   = running && elapsed >= r && r < weight_rows;
            wire row_active = row_live && (row_count[r] < total_rows);
            wire [ROW_ADDR_WIDTH-1:0] per_row_cnt = row_count[r];
            wire final_blk = (committed_n_blk_idx == max_n_blk_idx) && !out_raw;
            reg  final_blk_d;

            // cleared at the start of every pass, not only on &row_done: a
            // partial pass (weight_rows < ARRAY_ROWS) never counts its dead
            // rows, so &row_done stays low and the live rows would keep
            // row_count == total_rows into the next pass and skip it
            always @(posedge clock) begin
                if (~rst_n)
                    row_count[r] <= {ROW_ADDR_WIDTH{1'b0}};
                else if (pass_start)
                    row_count[r] <= {ROW_ADDR_WIDTH{1'b0}};
                else if (row_active)
                    row_count[r] <= row_count[r] + 1'b1;
                else if (&row_done)
                    row_count[r] <= 0;
            end

            assign row_done[r] = (row_count[r] == total_rows);

            // this bank's addresses for the position row_count[r] currently
            // points at: this pass's segment base plus the column, in each
            // canvas (see header)
            wire [PSUM_ADDR_WIDTH-1:0] cur_addr  = held_seg_base  + row_count[r];
            wire [ROW_ADDR_WIDTH-1:0]  cur_addr8 = held_seg_base8 + row_count[r];

            // what the canvas holds when use_prev_out is set (see header):
            // raw sums to continue (prev_int8 = 0), or an int8 result for the
            // final pass to add onto (prev_int8 = 1). Only the former stops
            // the first pass from starting the raw sum fresh.
            wire prev_raw   = use_prev_out && !prev_int8;
            wire prev_add   = final_blk && use_prev_out && prev_int8;

            assign psum_rden[r] = row_active;
            assign psum_rdaddress[r*PSUM_ADDR_WIDTH +: PSUM_ADDR_WIDTH] = cur_addr;

            assign obuf_rden[r] = row_active && prev_add;
            assign obuf_rdaddress[r*ROW_RADDR_WIDTH +: ROW_RADDR_WIDTH] = cur_addr8[ROW_ADDR_WIDTH-1:LANE_SEL_W];

            // bank r holds canvas rows r, r+ARRAY_ROWS, ...: this pass's rows
            // are canvas row block tile_row_blk + k_blk, constant for the pass
            assign bbuf_rden[r] = final_blk && bias_en ? row_active : 0;
            assign bbuf_rdaddress[r*BIAS_ADDR_WIDTH +: BIAS_ADDR_WIDTH] = tile_row_blk + held_k_blk_idx;

            // pipeline: the first stage lines psum_q[r] / obuf_q[r] (each
            // RAM's own one-cycle synchronous read latency) up with the psum
            // it should be added to, the rest carry that alongside the split
            // rescale multiply. On the final pass, stage 1b registers acc_last
            // between the accumulate/bias adds and the multiply (timing: RAM
            // out -> two adds -> soft multiply was one cycle), so stages 2..4
            // run one cycle later than their names. addr_d is the psum_buffer
            // slot (the stage-1 raw write-back); addr8_d, addr_1b and
            // addr_2d..addr_4d are the output_buffer byte (the final int8 write).
            reg                      active_d;
            reg                      active_1b;
            reg [ROW_ADDR_WIDTH-1:0] addr_1b;
            reg signed [ACC_WIDTH:0] acc_last_1b;   // acc_last, registered: the multiply's input
            reg [ACC_WIDTH-1:0]      old_val_1b;
            reg                      active_2d;
            reg                      active_3d;
            reg                      active_4d;
            reg [PSUM_ADDR_WIDTH-1:0] addr_d;
            reg [ROW_ADDR_WIDTH-1:0] addr8_d;
            reg [ROW_ADDR_WIDTH-1:0] addr_2d;
            reg [ROW_ADDR_WIDTH-1:0] addr_3d;
            reg [ROW_ADDR_WIDTH-1:0] addr_4d;
            reg [ACC_WIDTH-1:0]      psum_d;   // raw pre-rescale accumulator, captured this cycle
            reg [ACC_WIDTH-1:0]     old_val_d;
            wire [ACC_WIDTH-1:0]    old_val;  // prev_int8: this row's previous int8, sign-extended (lane select below)
            reg [ACC_WIDTH-1:0]     old_val_2d;
            wire [ACC_WIDTH-1:0]    old_psum = psum_q[r*ACC_WIDTH +: ACC_WIDTH];   // raw running sum read in stage 0
            wire [ACC_WIDTH-1:0]    bias;
            // stage-1 partial products: prod_hi is signed (psum's high chunk
            // carries the sign), prod_lo unsigned (a raw bit field).
            reg signed [HI_W+SCALE_WIDTH:0]      prod_hi;
            reg        [LO_W+SCALE_WIDTH-1:0]    prod_lo;
            reg signed [ACC_WIDTH+SCALE_WIDTH:0] scale_product;

            // stage 2's only arithmetic: line the two partial products back
            // up and add. prod_hi<<LO_W spans HI_W+SCALE_WIDTH+1+LO_W =
            // ACC_WIDTH+SCALE_WIDTH+1 bits, exactly scale_product's width,
            // so the shift can't drop the top of the high product.
            wire signed [ACC_WIDTH+SCALE_WIDTH:0] prod_hi_ext = $signed(prod_hi);
            wire signed [ACC_WIDTH+SCALE_WIDTH:0] prod_sum =
                (prod_hi_ext <<< LO_W) + $signed({1'b0, prod_lo});

            // stage-1 accumulate (declared here, ahead of the always block
            // that reads acc_last). psum_q/bbuf_q answer the stage-0 read, so
            // old_psum/bias line up with psum_d. The first pass of an element
            // takes psum alone - psum_buffer holds stale data there - unless
            // use_prev_out continues raw sums; a prev_int8 matmul starts
            // fresh, its old int8 joins after the rescale.
            assign bias = bbuf_q[r*ACC_WIDTH +: ACC_WIDTH];   // int32, accumulator units
            wire first_pass = (held_n_blk_idx == 0) && !prev_raw;
            wire signed [ACC_WIDTH:0] acc_sum = (running_d && !first_pass) ? $signed(old_psum) + $signed(psum_d)
                                                                            : $signed(psum_d);
            wire signed [ACC_WIDTH:0] acc_last = final_blk ? acc_sum + (bias_en ? $signed(bias) : 0) : 0;

            

            always @(posedge clock) begin
                if (~rst_n) begin
                    active_d  <= 1'b0;
                    addr_d    <= {PSUM_ADDR_WIDTH{1'b0}};
                    addr8_d   <= {ROW_ADDR_WIDTH{1'b0}};
                    psum_d    <= {ACC_WIDTH{1'b0}};
                    active_2d <= 1'b0;
                    addr_2d   <= {ROW_ADDR_WIDTH{1'b0}};
                    prod_hi   <= 0;
                    prod_lo   <= 0;
                    old_val_d <= 0;
                    active_3d <= 1'b0;
                    addr_3d   <= {ROW_ADDR_WIDTH{1'b0}};
                    scale_product <= 0;
                    old_val_2d    <= 0;
                    final_blk_d   <= 0;
                    active_4d     <= 1'b0;
                    addr_4d       <= {ROW_ADDR_WIDTH{1'b0}};
                    active_1b     <= 1'b0;
                    addr_1b       <= {ROW_ADDR_WIDTH{1'b0}};
                    acc_last_1b   <= 0;
                    old_val_1b    <= 0;
                end
                else begin
                    // stage 0 -> 1
                    active_d <= row_active;
                    addr_d   <= cur_addr;
                    addr8_d  <= cur_addr8;
                    psum_d   <= p_in[r*ACC_WIDTH +: ACC_WIDTH];
                    final_blk_d <= final_blk;

                    // only rescale at final output
                    if(final_blk_d)begin
                        // stage 1 -> 1b : register the accumulated, biased sum
                        acc_last_1b <= acc_last;
                        addr_1b     <= addr8_d;
                        active_1b   <= active_d;
                        old_val_1b  <= old_val;

                        // stage 1b -> 2 : the two narrow partial products
                        // (* multstyle = "logic" *)
                        prod_hi   <= $signed(acc_last_1b[ACC_WIDTH-1 -: HI_W]) *
                                    $signed({1'b0, output_scale});
                        // (* multstyle = "logic" *)
                        prod_lo   <= acc_last_1b[LO_W-1:0] * output_scale;
                        addr_2d   <= addr_1b;
                        active_2d <= active_1b;
                        old_val_d <= old_val_1b;

                        // stage 2 -> 3 : shift-and-add into the full product
                        scale_product <= prod_sum;
                        addr_3d       <= addr_2d;
                        active_3d     <= active_2d;
                        old_val_2d    <= old_val_d;

                        if(gelu_en)begin
                            addr_4d <= addr_3d;
                            active_4d <= active_3d;
                        end
                    end
                end
            end

            
            wire signed [ACC_WIDTH+SCALE_WIDTH:0] scale_shifted = scale_product >>> SCALE_WIDTH;

            // prev_int8: the previous result is already int8 (output units),
            // so it joins after the rescale - one saturation over the sum
            wire signed [ACC_WIDTH+SCALE_WIDTH+1:0] scale_shifted_prev =
                scale_shifted + (prev_add ? $signed(old_val_2d) : 0);

            wire [DATA_WIDTH-1:0] scaled_acc =
                (scale_shifted_prev > $signed(DATA_MAX)) ? DATA_MAX :
                (scale_shifted_prev < $signed(DATA_MIN)) ? DATA_MIN :
                                                         scale_shifted_prev[DATA_WIDTH-1:0];
            wire [DATA_WIDTH-1:0] gelu_out;
            wire                  gelu_out_valid;

            // prev_int8: obuf_q holds the word for the address issued last
            // cycle, so the byte is selected with the *delayed* address
            // (addr8_d), matching the same one-cycle RAM latency the rest of
            // this stage assumes. Bank r's word is at r*RD_DATA_WIDTH; the
            // byte's lane in it is the low LANE_SEL_W bits of its address.
            wire [DATA_WIDTH-1:0] old_byte =
                obuf_q[r*RD_DATA_WIDTH + addr8_d[LANE_SEL_W-1:0]*DATA_WIDTH +: DATA_WIDTH];
            assign old_val = {{(ACC_WIDTH-DATA_WIDTH){old_byte[DATA_WIDTH-1]}}, old_byte};

            // (first_pass / acc_sum / acc_last are declared above the always
            // block - acc_last is registered in stage 1b, ahead of the multiply)
            
            
            // registered LUT: out_data is valid one cycle after in_valid, i.e.
            // stage 4, alongside active_4d/addr_4d. Nothing downstream ever
            // stalls, so out_ready is tied high and in_ready is unused.
            gelu_int8 #(
                .LUT_FILE   (GELU_LUT_FILE)
            ) bias_gelu_inst (
                .clk        (clock),
                .reset_n    (rst_n),
                .in_data    (scaled_acc),
                .in_valid   (active_3d && gelu_en),
                .in_ready   (),
                .out_data   (gelu_out),
                .out_valid  (gelu_out_valid),
                .out_ready  (1'b1)
            );

            // every pass that doesn't finish the element writes the raw
            // running sum back to psum_buffer in stage 1; the final pass
            // writes the int8 result to output_buffer instead - one byte at
            // its canvas position, from stage 3, or stage 4 when it goes
            // through GELU. gelu_en only ever applies to the final pass.
            wire use_gelu = final_blk && gelu_en;

            assign psum_wren[r] = !final_blk && active_d;
            assign psum_waddr[r*PSUM_ADDR_WIDTH +: PSUM_ADDR_WIDTH] = addr_d;
            assign psum_wdata[r*ACC_WIDTH +: ACC_WIDTH] = acc_sum[ACC_WIDTH-1:0];

            wire [DATA_WIDTH-1:0] new_val = use_gelu ? gelu_out : scaled_acc;
            wire                  new_wren = use_gelu ? active_4d : active_3d;
            wire [ROW_ADDR_WIDTH-1:0] new_waddr = use_gelu ? addr_4d : addr_3d;

            assign obuf_wren[r] = final_blk && new_wren;
            assign obuf_waddr[r*ROW_ADDR_WIDTH +: ROW_ADDR_WIDTH] = new_waddr;
            assign obuf_wdata[r*DATA_WIDTH +: DATA_WIDTH] = new_val;
        end
    endgenerate

    // busy has to span the *union* of the read window and the write drain,
    // because it does two jobs: output_buffer_32_bank muxes each bank's read
    // port with it (loader wins while busy, Avalon once idle - see that
    // file's header), so it must already be high on the pass's very first
    // read cycle (a prev_int8 read), i.e. rise with `running`; and accel_top gates
    // matmul_complete on !busy, so it must not drop until the last write has
    // left the pipeline, i.e. fall with `running_4d` - the GELU stage-4
    // write, which stage 1b puts five cycles after the pass's last read
    // (also keeps weight_ctrl from committing the
    // next block while that write is still pending). Using the
    // delayed copies alone throws away the first read of every pass (obuf_q
    // then still holds the previous pass's word, and m=0 accumulates onto
    // garbage).
    assign busy = running || running_d || running_2d || running_3d || running_4d;
    assign done = running && &(row_done | ~((1 << weight_rows) - 1));

endmodule

// output_loader : accumulates pe_array's partial sums into output_buffer,
// which is banked one bank per array column (ARRAY_COLS banks total) -
// mirrors input_buffer_32_bank's banking pattern on the input side, scaled
// to ARRAY_COLS so every column that's currently producing a real result
// can be read-modify-written the same cycle it arrives, with no elastic
// buffering or backpressure required.
//
// "row"/"column" below mean the *final written-out output matrix's* row
// and column (as you'd write y out on paper), not weight_ctrl's block
// indices. Bank c always holds output row (n_blk_idx*ARRAY_COLS+c) for
// whichever n_blk_idx is currently committed - i.e. each bank is reused
// across every n_blk_idx pass, and within a bank the per-n_blk_idx
// segments (each total_rows elements, one per output column m) are simply
// appended head-to-tail: bank c's internal address = n_blk_idx*total_rows
// + m. That makes the layout row-major *within* a bank, but NOT globally
// row-major across the whole output_buffer (row n's data isn't one
// contiguous run - it's split one-32nd-per-bank).
//
// Since the contraction dimension may need more than one 32-block pass
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
// Read-modify-write per column: output_buffer is a synchronous on-chip
// RAM (q valid one cycle after rdaddress/rden), so the read for a
// position is issued one cycle *before* p_in[c] holds the psum for that
// same position - by the cycle the psum actually arrives, the old value
// is already sitting in obuf_q[c], ready to add and write back that same
// cycle.
//
// Timing model (from pe.v/pe_array.v):
//   - a_buf/b_buf only update when a_en/b_en pulse; p_out is recomputed
//     every cycle from whatever a_buf/b_buf currently hold. b_en must
//     therefore stay high for the *entire* compute+drain window once
//     started - gapping it would freeze that column's cascade mid-flight
//     and misalign every row still in transit. b_en is one signal
//     broadcast to all columns, not staggered per column.
//   - input_dispatch already staggers row r's a_en by r cycles, and the
//     a-side shift chain adds one more cycle of delay per column hop.
//     Those two skews cancel out along the diagonal, so column c's p_out
//     starts reflecting real (non-reset) data ARRAY_COLS-1+c cycles after
//     row 0's a_en first rises: column 0 first, then one more column every
//     following cycle, all ARRAY_COLS columns live ARRAY_COLS-1 cycles
//     after that.
//   - once a column is live it produces exactly one new valid result per
//     cycle, forever, so each column's captures are independently
//     addressed by its own free-running counter (0..total_rows-1) - no
//     need to realign columns to a shared row index.
//
// a_en_last is row 0's a_en bit (pe_array a_en[0] / input_dispatch a_en[0]);
// only its first rising edge after reset/completion is used to anchor
// timing, so later gaps in a_en_last (e.g. input FIFO underrun) don't
// re-trigger the sequence.
`timescale 1ps/1ps

module output_loader #(
    parameter DATA_WIDTH     = 8,    // output_buffer entry width (int8, post-rescale)
    parameter ACC_WIDTH      = 32,   // must match pe_array's ACC_WIDTH (p_in - raw pre-rescale accumulator)
    parameter SCALE_WIDTH    = 16,   // output_scale width - Q0.16 fixed point (unsigned, range [0,1))
    parameter ARRAY_COLS     = 32,
    parameter ROW_ADDR_WIDTH = 16,   // per-bank address width - must span num_n_blocks*total_rows
    parameter DIM_WIDTH      = 16    // width of committed_k_blk_idx (from weight_ctrl)
)(
    input                                          clock,
    input                                           rst_n,       // active-low

    // ---- pe_array bottom edge ----
    input      [ARRAY_COLS*ACC_WIDTH-1:0]          p_in,        // pe_array p_out
    output     [ARRAY_COLS-1:0]                    b_en,        // pe_array b_en

    // ---- timing anchor + block identity (from weight_ctrl) ----
    input                                           a_en_last,                 // pe_array a_en[0]
    input      [ROW_ADDR_WIDTH-1:0]                 total_rows,            // M: expected valid results per column, per pass
    input      [DIM_WIDTH-1:0]                      committed_k_blk_idx,   // weight_ctrl: output block this pass belongs to
    input      [DIM_WIDTH-1:0]                      committed_n_blk_idx,
    input                                            committed_first_k_blk, // weight_ctrl: write instead of accumulate

    // ---- rescale factor (from ctrl_rf, see rdl/ctrl_reg.rdl OUTPUT_SCALE) ----
    input      [SCALE_WIDTH-1:0]                    output_scale,

    // ---- output_buffer read+write ports, one bank per array column ----
    output     [ARRAY_COLS-1:0]                     obuf_rden,
    output     [ARRAY_COLS*ROW_ADDR_WIDTH-1:0]      obuf_rdaddress,
    input      [ARRAY_COLS*DATA_WIDTH-1:0]          obuf_q,
    output     [ARRAY_COLS-1:0]                     obuf_wren,
    output     [ARRAY_COLS*ROW_ADDR_WIDTH-1:0]      obuf_waddr,
    output     [ARRAY_COLS*DATA_WIDTH-1:0]          obuf_wdata,

    output                                          busy,
    output                                          done
);

    localparam ELAPSED_MAX = ARRAY_COLS - 1;
    localparam ELAPSED_W   = $clog2(ELAPSED_MAX + 1);

    // signed DATA_WIDTH saturation bounds (e.g. -128/127 for DATA_WIDTH=8)
    localparam signed [DATA_WIDTH-1:0] DATA_MAX = {1'b0, {(DATA_WIDTH-1){1'b1}}};
    localparam signed [DATA_WIDTH-1:0] DATA_MIN = {1'b1, {(DATA_WIDTH-1){1'b0}}};

    reg                     armed;      // ready to latch the next a_en_last rising edge
    reg                     a_en_last_d;
    reg                     running;
    reg  [ELAPSED_W:0]      elapsed;

    reg  [ROW_ADDR_WIDTH-1:0] col_count [0:ARRAY_COLS-1];
    wire [ARRAY_COLS-1:0]    col_done;

    // ---- sampled once at start_edge, held for this block's whole drain ----
    reg [DIM_WIDTH-1:0] held_k_blk_idx;
    reg [DIM_WIDTH-1:0] held_n_blk_idx;
    reg                 held_first_k_blk;

    wire start_edge = armed && a_en_last && !a_en_last_d;

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
            held_k_blk_idx   <= '0;
            held_n_blk_idx   <= '0;
            held_first_k_blk <= 1'b0;
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
                end
            end
            
        end
    end

    // held high for the whole compute+drain window - never gated per column
    assign b_en = {ARRAY_COLS{running}};

    genvar c;
    generate
        for (c = 0; c < ARRAY_COLS; c = c + 1) begin : COL
            wire col_live   = running && elapsed >= c;
            wire col_active = col_live && (col_count[c] < total_rows);
            wire [ROW_ADDR_WIDTH-1:0] per_col_cnt = col_count[c];

            always @(posedge clock) begin
                if (~rst_n)
                    col_count[c] <= {ROW_ADDR_WIDTH{1'b0}};
                else if (col_active)
                    col_count[c] <= col_count[c] + 1'b1;
                else if (&col_done)
                    col_count[c] <= 0;
            end

            assign col_done[c] = (col_count[c] == total_rows);

            // this bank's address for the position col_count[c] currently
            // points at: held_k_blk_idx's segment, appended after every
            // earlier n_blk_idx's total_rows-sized segment in this bank
            wire [ROW_ADDR_WIDTH-1:0] cur_addr = held_k_blk_idx*total_rows + col_count[c];

            assign obuf_rden[c] = col_active;
            assign obuf_rdaddress[c*ROW_ADDR_WIDTH +: ROW_ADDR_WIDTH] = cur_addr;

            // one-cycle pipeline so obuf_q[c] (this RAM's own one-cycle
            // synchronous read latency) lines up with the psum it should
            // be added to
            reg                      active_d;
            reg                      active_2d;
            reg [ROW_ADDR_WIDTH-1:0] addr_d;
            reg [ROW_ADDR_WIDTH-1:0] addr_2d;
            reg [ACC_WIDTH-1:0]      psum_d;   // raw pre-rescale accumulator, captured this cycle

            always @(posedge clock) begin
                if (~rst_n) begin
                    active_d <= 1'b0;
                    addr_d   <= {ROW_ADDR_WIDTH{1'b0}};
                    psum_d   <= {ACC_WIDTH{1'b0}};
                end
                else begin
                    active_d <= col_active;
                    addr_d   <= cur_addr;
                    psum_d   <= p_in[c*ACC_WIDTH +: ACC_WIDTH];

                    addr_2d  <= addr_d;
                    active_2d <= active_d;
                end
            end

            // rescale: psum_d (ACC_WIDTH) * output_scale (Q0.16) >> 16,
            // saturated into DATA_WIDTH's signed range. {1'b0,output_scale}
            // keeps the (always non-negative) scale factor signed-safe for
            // the multiply.
            wire signed [ACC_WIDTH+SCALE_WIDTH:0] scale_product = $signed(psum_d) * $signed({1'b0, output_scale});
            wire signed [ACC_WIDTH+SCALE_WIDTH:0] scale_shifted = scale_product >>> SCALE_WIDTH;

            wire [DATA_WIDTH-1:0] scaled_psum =
                (scale_shifted > $signed(DATA_MAX)) ? DATA_MAX :
                (scale_shifted < $signed(DATA_MIN)) ? DATA_MIN :
                                                       scale_shifted[DATA_WIDTH-1:0];

            wire [DATA_WIDTH-1:0] old_val = obuf_q[c*DATA_WIDTH +: DATA_WIDTH];

            // k-block accumulate: also saturated, since old_val+scaled_psum
            // can exceed DATA_WIDTH even when each term is already in range
            wire signed [DATA_WIDTH:0] acc_sum = running ? $signed(old_val) + $signed(scaled_psum) : 0;
            wire [DATA_WIDTH-1:0] acc_sat =
                (acc_sum > $signed(DATA_MAX)) ? DATA_MAX :
                (acc_sum < $signed(DATA_MIN)) ? DATA_MIN :
                                                 acc_sum[DATA_WIDTH-1:0];

            wire [DATA_WIDTH-1:0] new_val = (held_n_blk_idx == 0) ? scaled_psum : acc_sat;

            assign obuf_wren[c] = active_d;
            assign obuf_waddr[c*ROW_ADDR_WIDTH +: ROW_ADDR_WIDTH]  = addr_d;
            assign obuf_wdata[c*DATA_WIDTH +: DATA_WIDTH]         = new_val;
        end
    endgenerate

    assign busy = running;
    assign done = running && &col_done;

endmodule

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
// bank the per-n_blk_idx segments (each total_rows elements, one per
// output column m) are simply appended head-to-tail: bank r's internal
// address = n_blk_idx*total_rows + m. That makes the layout row-major
// *within* a bank, but NOT globally row-major across the whole
// output_buffer (output row n's data isn't one contiguous run - it's
// split one-ARRAY_COLS-th-per-bank).
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
// Read-modify-write per row: output_buffer is a synchronous on-chip
// RAM (q valid one cycle after rdaddress/rden), so the read for a
// position is issued one cycle *before* p_in[r] holds the psum for that
// same position - by the cycle the psum actually arrives, the old value
// is already sitting in obuf_q[r], ready to add and write back that same
// cycle.
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
`timescale 1ps/1ps

(* multstyle = "logic" *)
module output_loader #(
    parameter DATA_WIDTH     = 8,    // output_buffer entry width (int8, post-rescale)
    parameter ACC_WIDTH      = 32,   // must match pe_array's ACC_WIDTH (p_in - raw pre-rescale accumulator)
    parameter SCALE_WIDTH    = 16,   // output_scale width - Q0.16 fixed point (unsigned, range [0,1))
    parameter ARRAY_COLS     = 16,
    parameter ARRAY_ROWS     = 16,
    parameter ROW_ADDR_WIDTH = 16,   // per-bank WRITE (byte) address width - must span num_n_blocks*total_rows
    parameter RD_DATA_WIDTH  = 32,   // output_buffer read port width (wide, shared with Avalon)
    // The output_buffer read port is wide (one word per access) because the
    // Avalon host reads through it; this module only ever wants one byte of
    // that word, so it addresses the RAM in words and selects the lane its
    // byte actually lives in. The write port stays DATA_WIDTH, so the
    // read-modify-write still writes back exactly one byte.
    parameter LANE_SEL_W     = $clog2(RD_DATA_WIDTH/DATA_WIDTH),
    parameter ROW_RADDR_WIDTH = ROW_ADDR_WIDTH - LANE_SEL_W,
    parameter DIM_WIDTH      = 16,   // width of committed_k_blk_idx (from weight_ctrl)
    parameter COL_SEL_W      = $clog2(ARRAY_COLS+1),
    parameter ROW_SEL_W      = $clog2(ARRAY_ROWS+1)
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

    // ---- rescale factor (from ctrl_rf, see rdl/ctrl_reg.rdl OUTPUT_SCALE) ----
    input      [SCALE_WIDTH-1:0]                    output_scale,

    // ---- current valid rows and cols for output ----
    input      [COL_SEL_W-1:0]                      weight_cols,
    input      [ROW_SEL_W-1:0]                      weight_rows,

    // ---- output_buffer read+write ports, one bank per array row ----
    output     [ARRAY_COLS-1:0]                     obuf_rden,
    output     [ARRAY_COLS*ROW_RADDR_WIDTH-1:0]     obuf_rdaddress,
    input      [ARRAY_COLS*RD_DATA_WIDTH-1:0]       obuf_q,
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

    reg  [ROW_ADDR_WIDTH-1:0] row_count [0:ARRAY_COLS-1];
    wire [ARRAY_COLS-1:0]    row_done;

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
            held_k_blk_idx   <= 0;
            held_n_blk_idx   <= 0;
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

    // held high for the whole compute+drain window - never gated per row
    assign b_en = {ARRAY_COLS{running}};

    genvar r;
    generate
        for (r = 0; r < ARRAY_ROWS; r = r + 1) begin : ROW
            wire row_live   = running && elapsed >= r && r < weight_rows;
            wire row_active = row_live && (row_count[r] < total_rows);
            wire [ROW_ADDR_WIDTH-1:0] per_row_cnt = row_count[r];

            always @(posedge clock) begin
                if (~rst_n)
                    row_count[r] <= {ROW_ADDR_WIDTH{1'b0}};
                else if (row_active)
                    row_count[r] <= row_count[r] + 1'b1;
                else if (&row_done)
                    row_count[r] <= 0;
            end

            assign row_done[r] = (row_count[r] == total_rows);

            // this bank's address for the position row_count[r] currently
            // points at: held_k_blk_idx's segment, appended after every
            // earlier n_blk_idx's total_rows-sized segment in this bank
            (* multstyle = "logic" *)
            wire [ROW_ADDR_WIDTH-1:0] cur_addr = held_k_blk_idx*total_rows + row_count[r];

            assign obuf_rden[r] = row_active;
            assign obuf_rdaddress[r*ROW_RADDR_WIDTH +: ROW_RADDR_WIDTH] = cur_addr[ROW_ADDR_WIDTH-1:LANE_SEL_W];

            // one-cycle pipeline so obuf_q[r] (this RAM's own one-cycle
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
                    active_d <= row_active;
                    addr_d   <= cur_addr;
                    psum_d   <= p_in[r*ACC_WIDTH +: ACC_WIDTH];

                    addr_2d  <= addr_d;
                    active_2d <= active_d;
                end
            end

            // rescale: psum_d (ACC_WIDTH) * output_scale (Q0.16) >> 16,
            // saturated into DATA_WIDTH's signed range. {1'b0,output_scale}
            // keeps the (always non-negative) scale factor signed-safe for
            // the multiply.
            (* multstyle = "logic" *)
            wire signed [ACC_WIDTH+SCALE_WIDTH:0] scale_product = $signed(psum_d) * $signed({1'b0, output_scale});
            wire signed [ACC_WIDTH+SCALE_WIDTH:0] scale_shifted = scale_product >>> SCALE_WIDTH;

            wire [DATA_WIDTH-1:0] scaled_psum =
                (scale_shifted > $signed(DATA_MAX)) ? DATA_MAX :
                (scale_shifted < $signed(DATA_MIN)) ? DATA_MIN :
                                                    scale_shifted[DATA_WIDTH-1:0];

            // obuf_q holds the word for the address issued last cycle, so the
            // lane is selected with the *delayed* address (addr_d), matching
            // the same one-cycle RAM latency the rest of this stage assumes.
            wire [DATA_WIDTH-1:0] old_val =
                obuf_q[r*RD_DATA_WIDTH + addr_d[LANE_SEL_W-1:0]*DATA_WIDTH +: DATA_WIDTH];

            // k-block accumulate: also saturated, since old_val+scaled_psum
            // can exceed DATA_WIDTH even when each term is already in range
            wire signed [DATA_WIDTH:0] acc_sum = running ? $signed(old_val) + $signed(scaled_psum) : 0;
            wire [DATA_WIDTH-1:0] acc_sat =
                (acc_sum > $signed(DATA_MAX)) ? DATA_MAX :
                (acc_sum < $signed(DATA_MIN)) ? DATA_MIN :
                                                acc_sum[DATA_WIDTH-1:0];

            wire [DATA_WIDTH-1:0] new_val = (held_n_blk_idx == 0) ? scaled_psum : acc_sat;

            assign obuf_wren[r] = active_d;
            assign obuf_waddr[r*ROW_ADDR_WIDTH +: ROW_ADDR_WIDTH]  = addr_d;
            assign obuf_wdata[r*DATA_WIDTH +: DATA_WIDTH]         = new_val;
        end
    endgenerate

    assign busy = running;
    assign done = running && &(row_done | ~((1 << weight_rows) - 1));

endmodule

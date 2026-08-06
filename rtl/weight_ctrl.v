// weight_ctrl : block/tile sequencer for weight-stationary loading, driving
// both weight_loader (weight_buffer -> pe_array) and input_dispatch's
// per-block preload/compute-start handshake. matmul_start is the only
// external trigger - everything past that (which block is current, when to
// prefetch/commit weights, when to (re)preload input bands, when the whole
// matmul is done) is sequenced internally, since the access pattern is
// fixed and doesn't benefit from being driven step-by-step by software.
//
// Naming note, since this tripped us up before: k_blk_idx/n_blk_idx name
// which *PE-array axis* a block index feeds, not "row/column of the matrix
// as you'd write it on paper":
//   - k_blk_idx selects a 32-wide slice of WEIGHT_ROWS and feeds pe_array's
//     ARRAY_ROWS axis - the one pe.v's b_in cascades down and sums. That
//     makes k_blk_idx the *contraction* dimension's block index, whatever
//     you happen to call that dimension in your own matrix convention.
//   - n_blk_idx selects a 32-wide slice of WEIGHT_COLS and feeds
//     ARRAY_COLS - values here stay separate (one accumulated result per
//     array column), so this is the *output* dimension's block index.
//   - Concretely: if you write your weight matrix as (out_features x
//     in_features), WEIGHT_ROWS must be programmed with in_features (the
//     contraction size) and WEIGHT_COLS with out_features, regardless of
//     which one is literally "rows" on paper.
//
// Access pattern (y = W x, W and x both tiled into 32x32 blocks):
//   - Traverses W's block grid *column-major*: k_blk_idx is the fast/inner
//     index, n_blk_idx the slow/outer one. For a fixed n_blk_idx, every
//     k_blk_idx is processed (and summed into that output block, per the
//     b_in cascade) before moving to the next n_blk_idx. This keeps the
//     amount of in-flight state small: only one output block's worth of
//     accumulation needs to be resident at a time, unlike row-major order
//     which would need the whole output width kept live at once (see
//     rdl/README.md - multi-k_blk output accumulation itself is still a
//     separate TODO, not handled here).
//   - x is row-major with its rows indexed by the same contraction
//     dimension as W's k_blk_idx (x[k,:] pairs with W[k,:] for every
//     output block), so k_blk_idx also selects which 32-row band of
//     input_buffer the *current* block needs; input_dispatch is
//     retriggered with a new base address every time k_blk_idx moves,
//     whether that's within the same n_blk_idx or wrapping into the next.
//   - matmul_start latches WEIGHT_ROWS/WEIGHT_COLS and resets the block
//     position to (0,0) - the only point they're re-sampled.
`timescale 1ps/1ps

module weight_ctrl #(
    parameter DATA_WIDTH      = 8,
    parameter ARRAY_ROWS      = 32,
    parameter ARRAY_COLS      = 32,
    parameter WBUF_ADDR_WIDTH = 14,
    parameter IBUF_ADDR_WIDTH = 8,    // input_buffer per-bank address width (input_dispatch)
    parameter DIM_WIDTH       = 16    // WEIGHT_ROWS/WEIGHT_COLS field width
)(
    input                        clk,
    input                        rst_n,

    // ---- ctrl_rf handshake ----
    input                        matmul_start,       // pulse: latch dims, reset block position to (0,0)
    input      [DIM_WIDTH-1:0]   weight_rows,        // contraction size (see header note)
    input      [DIM_WIDTH-1:0]   weight_cols,        // output size (see header note)

    output                       busy,               // whole matmul in progress (from matmul_start to done)
    output                       done,               // last block committed and fully streamed/drained

    // ---- weight_buffer read port ----
    output                       wbuf_rden,
    output [WBUF_ADDR_WIDTH-1:0] wbuf_rdaddress,
    input      [DATA_WIDTH-1:0]  wbuf_q,

    // ---- pe_array weight-load port ----
    output [$clog2(ARRAY_ROWS*ARRAY_COLS)-1:0] w_addr,
    output [DATA_WIDTH-1:0]                     w_data,
    output                                      w_en,
    output                                      w_load,

    // ---- input_dispatch handshake ----
    input      [IBUF_ADDR_WIDTH-1:0] input_max_addr,       // per-band input_buffer bound (input_dispatch i_max_addr)
    output reg                       input_band_start,      // pulse: begin preloading input_band_base_addr's band
    output     [IBUF_ADDR_WIDTH-1:0] input_band_base_addr,  // this block's 32-row band base address in input_buffer
    output reg                       input_compute_start,   // pulse: begin streaming the preloaded band into pe_array
    input                             input_fifos_primed,    // level: all row FIFOs hold >=1 element
    input                             input_band_done,       // pulse: current band fully drained, next preload may start

    // ---- output_loader handshake ----
    // Latched at commit time (not live k_blk_idx/n_blk_idx!), since those
    // advance to the *next* block's position as soon as this one commits,
    // while output_loader is still draining psums for the block that just
    // committed. output_loader samples these once, at its own a_en0 start
    // edge (which always happens well before the next commit could
    // possibly land), so they stay valid for its whole drain window.
    output reg [DIM_WIDTH-1:0] committed_n_blk_idx,   // which output block this draining pass belongs to
    output reg                 committed_first_k_blk,  // this pass is k_blk_idx==0 for that output block - write, don't accumulate
    output reg [DIM_WIDTH-1:0] committed_k_blk_idx
);

    localparam K_CNT_WIDTH = $clog2(ARRAY_ROWS+1);
    localparam N_CNT_WIDTH = $clog2(ARRAY_COLS+1);

    // ARRAY_ROWS/ARRAY_COLS are powers of 2 - divide by shifting instead
    localparam K_SHIFT = $clog2(ARRAY_ROWS);
    localparam N_SHIFT = $clog2(ARRAY_COLS);

    localparam WC_IDLE = 2'd0,   // waiting for matmul_start
               WC_WAIT = 2'd1,   // weight prefetch + input band preload in flight
               WC_RUN  = 2'd2,   // committed; streaming; waiting for input_band_done
               WC_DONE = 2'd3;   // last block finished

    reg [1:0] wc_state;
    reg       last_block_pending;  // latched at commit time: was that the last block?

    // ---- latched matmul dimensions + current block position ----
    reg [DIM_WIDTH-1:0] k_latched, n_latched;              // contraction size, output size
    reg [DIM_WIDTH-1:0] k_blk_idx, n_blk_idx;               // current block position (see header note)

    // number of 32-wide blocks spanning the latched contraction/output dims
    wire [DIM_WIDTH-1:0] num_k_blocks = (k_latched + ARRAY_ROWS - 1) >> K_SHIFT;
    wire [DIM_WIDTH-1:0] num_n_blocks = (n_latched + ARRAY_COLS - 1) >> N_SHIFT;

    wire is_last_k_blk  = (k_blk_idx == num_k_blocks-1);
    wire is_last_n_blk  = (n_blk_idx == num_n_blocks-1);
    wire is_last_block  = is_last_n_blk && is_last_k_blk;

    // ---- current block's valid extent - masks a partial last block along
    // either dimension when the contraction/output size isn't a multiple
    // of 32 ----
    wire [DIM_WIDTH-1:0] k_rem = k_latched - k_blk_idx*ARRAY_ROWS;
    wire [DIM_WIDTH-1:0] n_rem = n_latched - n_blk_idx*ARRAY_COLS;
    wire [K_CNT_WIDTH-1:0] valid_k = (k_rem < ARRAY_ROWS) ? k_rem[K_CNT_WIDTH-1:0] : ARRAY_ROWS[K_CNT_WIDTH-1:0];
    wire [N_CNT_WIDTH-1:0] valid_n = (n_rem < ARRAY_COLS) ? n_rem[N_CNT_WIDTH-1:0] : ARRAY_COLS[N_CNT_WIDTH-1:0];

    // ---- current block's top-left address in weight_buffer (row-major
    // over the full contraction x output matrix) ----
    wire [WBUF_ADDR_WIDTH-1:0] base_addr = (k_blk_idx*ARRAY_ROWS)*n_latched + n_blk_idx*ARRAY_COLS;

    // ---- current block's 32-row band base address in input_buffer (also
    // assumed row-major: each band occupies one input_max_addr-sized span) ----
    assign input_band_base_addr = n_blk_idx[IBUF_ADDR_WIDTH-1:0] * input_max_addr;

    reg loader_start;
    reg commit_pulse;
    wire loader_busy;
    wire loader_ready;  // block fully prefetched, sitting in pe_array's prefetch regs

    assign busy = (wc_state != WC_IDLE);
    assign done = (wc_state == WC_DONE);

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            wc_state             <= WC_IDLE;
            last_block_pending   <= 1'b0;
            k_latched            <= '0;
            n_latched            <= '0;
            k_blk_idx            <= '0;
            n_blk_idx            <= '0;
            loader_start         <= 1'b0;
            commit_pulse         <= 1'b0;
            input_band_start     <= 1'b0;
            input_compute_start  <= 1'b0;
            committed_n_blk_idx  <= '0;
            committed_first_k_blk <= 1'b0;
            committed_k_blk_idx  <= '0;
        end
        else begin
            loader_start        <= 1'b0;
            commit_pulse        <= 1'b0;
            input_band_start    <= 1'b0;
            input_compute_start <= 1'b0;

            if (matmul_start) begin
                // weight prefetch takes >=1024 cycles regardless, so kicking
                // it and the input band-0 preload the same cycle (rather
                // than strictly weight-then-input, one cycle apart) makes
                // no difference to when both are actually ready
                k_latched         <= weight_rows;
                n_latched         <= weight_cols;
                k_blk_idx         <= '0;
                n_blk_idx         <= '0;
                loader_start      <= 1'b1;  // kick block (0,0)'s weight prefetch
                input_band_start  <= 1'b1;  // kick input band 0's preload
                wc_state          <= WC_WAIT;
            end
            else begin

                case (wc_state)
                    WC_WAIT: begin
                        if (loader_ready && input_fifos_primed) begin
                            commit_pulse         <= 1'b1;  // pe_array.w_load
                            input_compute_start  <= 1'b1;  // input_dispatch PRELOAD -> OFFLOAD
                            last_block_pending   <= is_last_block;
                            committed_n_blk_idx   <= n_blk_idx;
                            committed_k_blk_idx   <= k_blk_idx;
                            committed_first_k_blk <= (k_blk_idx == '0);

                            if (!is_last_block) begin
                                if (is_last_k_blk) begin
                                    k_blk_idx <= '0;
                                    n_blk_idx <= n_blk_idx + 1'b1;
                                end
                                else begin
                                    k_blk_idx <= k_blk_idx + 1'b1;
                                end
                                loader_start <= 1'b1;  // prefetch the next block, overlapping this one's compute
                            end
                            wc_state <= WC_RUN;
                        end
                    end

                    WC_RUN: begin
                        if (input_band_done) begin
                            if (last_block_pending) begin
                                wc_state <= WC_DONE;
                            end
                            else begin
                                input_band_start <= 1'b1;  // preload the (already-advanced) next band
                                wc_state         <= WC_WAIT;
                            end
                        end
                    end

                    WC_DONE: begin
                        // sit here until the next matmul_start
                    end

                    default: wc_state <= WC_IDLE;
                endcase
            end
        end
    end

    weight_loader #(
        .DATA_WIDTH      (DATA_WIDTH),
        .ARRAY_ROWS      (ARRAY_ROWS),
        .ARRAY_COLS      (ARRAY_COLS),
        .WBUF_ADDR_WIDTH (WBUF_ADDR_WIDTH),
        .DIM_WIDTH       (DIM_WIDTH)
    ) u_weight_loader (
        .clk            (clk),
        .rst_n          (rst_n),

        .wbuf_rdy       (loader_start),
        .base_addr      (base_addr),
        .row_stride     (n_latched),
        .valid_rows     (valid_k),
        .valid_cols     (valid_n),
        .commit         (commit_pulse),
        .busy           (loader_busy),
        .ready          (loader_ready),
        .done           (),

        .wbuf_rden      (wbuf_rden),
        .wbuf_rdaddress (wbuf_rdaddress),
        .wbuf_q         (wbuf_q),

        .w_addr         (w_addr),
        .w_data         (w_data),
        .w_en           (w_en),
        .w_load         (w_load)
    );

endmodule

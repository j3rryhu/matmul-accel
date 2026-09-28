// weight_loader : streams one ARRAY_ROWS x ARRAY_COLS block's worth of
// weights out of weight_buffer_32_bank and prefetches them into pe_array,
// one column per cycle across every row in parallel, then commits them all
// at once.
//
// weight_buffer_32_bank has one bank per PE row (layout in that file), and
// bank r's read port feeds pe_array's row-r weight port and nothing else.
// A block's columns sit at the same offsets in every bank, so one shared
// read address walks the whole block: a column per cycle, the full block in
// ARRAY_COLS cycles instead of ARRAY_ROWS*ARRAY_COLS.
//
// Each bank is a byte-wide, synchronous on-chip RAM: q is valid exactly one
// cycle after rdaddress/rden are driven. pe_array takes one regfile-style
// weight write port per row:
//   - w_addr[r] : flat PE index, r*ARRAY_COLS+col
//   - w_data[r] : weight value for that PE
//   - w_en      : shared by every row - latches each row's w_data into the
//                 addressed PE's prefetch reg this cycle
//   - w_load    : single global pulse, commits every prefetch reg -> active weight
//
// This module only knows how to stream and commit a single block; it has no
// notion of where that block sits within a larger K x N matrix or which
// block is next - that bookkeeping (block position, base address, partial-
// block extent, when to commit) lives in weight_ctrl, which drives
// base_addr/valid_rows/valid_cols/wbuf_rdy/commit here and re-triggers this
// module once per block.
//
// valid_rows/valid_cols mask a partial block (K or N not a multiple of the
// array size) down to the region that actually holds real data. Every PE of
// the block is written on every load: out-of-range rows' banks and
// out-of-range columns are not read at all, and those PEs get an explicit 0,
// so a partial block always zero-pads rather than leaving the previous
// block's weights behind.
`timescale 1ps/1ps

module weight_loader #(
    parameter DATA_WIDTH      = 8,
    parameter ARRAY_ROWS      = 16,
    parameter ARRAY_COLS      = 16,
    parameter BANK_ADDR_WIDTH = 10    // weight_buffer_32_bank per-bank read address width
)(
    input                                       clk,
    input                                       rst_n,       // active-low

    // ---- control ----
    input                                       wbuf_rdy,     // pulse: start streaming a new block
    input      [BANK_ADDR_WIDTH-1:0]            base_addr,    // block's first column, same offset in every bank
    input      [$clog2(ARRAY_ROWS+1)-1:0]       valid_rows,   // rows of this block holding real data
    input      [$clog2(ARRAY_COLS+1)-1:0]       valid_cols,   // cols of this block holding real data
    input                                       commit,       // pulse: commit the prefetched block (-> pe_array w_load)
    output                                      busy,         // high from wbuf_rdy until commit
    output                                      ready,        // high once the block is fully prefetched, waiting for commit
    output                                      done,         // high once committed, until the next wbuf_rdy

    // ---- weight_buffer_32_bank read port, one bank per PE row ----
    output reg                                  wbuf_rden,        // shared by every bank
    output     [ARRAY_ROWS*BANK_ADDR_WIDTH-1:0] wbuf_rdaddress,   // one offset, replicated to every bank
    output reg [ARRAY_ROWS-1:0]                 wbuf_byteenable,  // per bank (= per row) read enable
    input      [ARRAY_ROWS*DATA_WIDTH-1:0]      wbuf_q,

    // ---- pe_array weight-load ports, one per PE row ----
    output     [ARRAY_ROWS*$clog2(ARRAY_ROWS*ARRAY_COLS)-1:0] w_addr,
    output     [ARRAY_ROWS*DATA_WIDTH-1:0]      w_data,
    output                                      w_en,
    output                                      w_load
);

    localparam PE_ADDR_WIDTH  = $clog2(ARRAY_ROWS*ARRAY_COLS);
    localparam COL_ADDR_WIDTH = $clog2(ARRAY_COLS);
    localparam ROW_CNT_WIDTH  = $clog2(ARRAY_ROWS+1);
    localparam COL_CNT_WIDTH  = $clog2(ARRAY_COLS+1);

    localparam STATE_IDLE    = 2'd0,
               STATE_PRELOAD = 2'd1,   // walking the block's columns, issuing bank reads
               STATE_DRAIN   = 2'd2,   // last column's read is in flight
               STATE_READY   = 2'd3;   // block fully prefetched, waiting for commit

    reg [1:0] wload_state;
    reg       done_r;

    // ---- walk position within the current block, plus the block's extent
    // (latched with base_addr at wbuf_rdy so nothing shifts mid-walk) ----
    reg [COL_CNT_WIDTH-1:0]   cur_col;
    reg [BANK_ADDR_WIDTH-1:0] rd_addr;
    reg [ROW_CNT_WIDTH-1:0]   blk_valid_rows;
    reg [COL_CNT_WIDTH-1:0]   blk_valid_cols;

    wire col_valid = (cur_col < blk_valid_cols);
    wire last_col  = (cur_col == ARRAY_COLS-1);

    // banks of rows past valid_rows are never read
    wire [ARRAY_ROWS-1:0] row_valid;
    genvar gr;
    generate
        for (gr = 0; gr < ARRAY_ROWS; gr = gr + 1) begin : ROW_VALID
            assign row_valid[gr] = (gr < blk_valid_rows);
        end
    endgenerate

    assign busy  = (wload_state != STATE_IDLE);
    assign ready = (wload_state == STATE_READY);
    assign done  = done_r;

    // ---- read-issue stage: registered straight into every bank's RAM
    // inputs. rd_walk/rd_col tag the block column those reads belong to. ----
    reg [BANK_ADDR_WIDTH-1:0] wbuf_rd_addr;
    reg                       rd_walk;
    reg [COL_ADDR_WIDTH-1:0]  rd_col;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            wload_state     <= STATE_IDLE;
            done_r          <= 1'b0;
            cur_col         <= 0;
            rd_addr         <= 0;
            blk_valid_rows  <= 0;
            blk_valid_cols  <= 0;
            wbuf_rden       <= 1'b0;
            wbuf_rd_addr    <= 0;
            wbuf_byteenable <= 0;
            rd_walk         <= 1'b0;
            rd_col          <= 0;
        end
        else begin
            case (wload_state)
                STATE_IDLE: begin
                    if (wbuf_rdy) begin
                        wload_state    <= STATE_PRELOAD;
                        done_r         <= 1'b0;
                        cur_col        <= 0;
                        rd_addr        <= base_addr;
                        blk_valid_rows <= valid_rows;
                        blk_valid_cols <= valid_cols;
                    end
                end

                STATE_PRELOAD: begin
                    // one block column per cycle, every row's bank at once
                    done_r          <= 1'b0;
                    wbuf_rden       <= col_valid;
                    wbuf_rd_addr    <= rd_addr;
                    wbuf_byteenable <= row_valid;
                    rd_walk         <= 1'b1;
                    rd_col          <= cur_col[COL_ADDR_WIDTH-1:0];
                    cur_col         <= cur_col + 1'b1;
                    rd_addr         <= rd_addr + 1'b1;

                    if (last_col)
                        wload_state <= STATE_DRAIN;
                end

                STATE_DRAIN: begin
                    // the last column's read is in the RAMs this cycle; its
                    // wbuf_q (1-cycle synchronous read) is written into
                    // pe_array on the first STATE_READY cycle. weight_ctrl's
                    // commit is registered off `ready`, so w_load always
                    // lands at least one cycle after that last write
                    wbuf_rden   <= 1'b0;
                    rd_walk     <= 1'b0;
                    wload_state <= STATE_READY;
                end

                STATE_READY: begin
                    // prefetch registers hold the full block; sit here until
                    // weight_ctrl pulses commit
                    if (commit) begin
                        wload_state <= STATE_IDLE;
                        done_r      <= 1'b1;
                    end

                    if (wbuf_rdy) begin
                        wload_state    <= STATE_PRELOAD;
                        done_r         <= 1'b1;
                        cur_col        <= 0;
                        rd_addr        <= base_addr;
                        blk_valid_rows <= valid_rows;
                        blk_valid_cols <= valid_cols;
                    end
                end
            endcase
        end
    end

    // ---- data stage: wbuf_q now holds the column rd_walk/rd_col tagged
    // last cycle. q_lane_valid[r] = bank r was actually read for it (column
    // AND row in range); every other lane is written as 0. ----
    reg                      q_walk;
    reg [COL_ADDR_WIDTH-1:0] q_col;
    reg [ARRAY_ROWS-1:0]     q_lane_valid;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            q_walk       <= 1'b0;
            q_col        <= 0;
            q_lane_valid <= 0;
        end
        else begin
            q_walk       <= rd_walk;
            q_col        <= rd_col;
            q_lane_valid <= wbuf_byteenable & {ARRAY_ROWS{wbuf_rden}};
        end
    end

    assign wbuf_rdaddress = {ARRAY_ROWS{wbuf_rd_addr}};

    // every row writes the same column each cycle, each into its own PE
    genvar gw;
    generate
        for (gw = 0; gw < ARRAY_ROWS; gw = gw + 1) begin : W_PORT
            assign w_addr[gw*PE_ADDR_WIDTH +: PE_ADDR_WIDTH] = gw*ARRAY_COLS + q_col;
            assign w_data[gw*DATA_WIDTH +: DATA_WIDTH] =
                q_lane_valid[gw] ? wbuf_q[gw*DATA_WIDTH +: DATA_WIDTH] : {DATA_WIDTH{1'b0}};
        end
    endgenerate

    assign w_en   = q_walk;
    assign w_load = commit && (wload_state == STATE_READY);

endmodule

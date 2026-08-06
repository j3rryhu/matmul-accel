// weight_loader : streams one 32x32 block's worth of weights out of the
// weight_buffer RAM and prefetches them into pe_array, one PE per cycle,
// then commits them all at once.
//
// weight_buffer is a byte-wide, synchronous on-chip RAM: q is valid exactly
// one cycle after rdaddress/rden are driven.  pe_array takes a regfile-style
// weight write port:
//   - w_addr : flat PE index, row*ARRAY_COLS+col
//   - w_data : weight value for that PE
//   - w_en   : latches w_data into the addressed PE's prefetch reg this cycle
//   - w_load : single global pulse, commits every prefetch reg -> active weight
//
// This module only knows how to stream and commit a single block; it has no
// notion of where that block sits within a larger K x N matrix or which
// block is next - that bookkeeping (matmul_start/weight_load_start/
// compute_start sequencing, block position, row-major base address) lives
// in weight_ctrl, which drives base_addr/row_stride/valid_rows/valid_cols/
// wbuf_rdy/commit here and re-triggers this module once per block.
//
// weight_buffer layout is assumed row-major over the *full* K x N matrix,
// so a block's rows are not contiguous: row_stride (= N) is how far to step
// in the RAM to move from the end of one block row to the start of the
// next. valid_rows/valid_cols mask a partial block (K or N not a multiple
// of 32) down to the region that actually holds real data - PEs outside
// that region are explicitly written 0 instead of read from the RAM, so a
// partial block always zero-pads rather than leaving stale weights behind.
`timescale 1ps/1ps

module weight_loader #(
    parameter DATA_WIDTH      = 8,
    parameter ARRAY_ROWS      = 32,
    parameter ARRAY_COLS      = 32,
    parameter WBUF_ADDR_WIDTH = 14,   // weight_buffer address width
    parameter DIM_WIDTH       = 16    // width of row_stride (full-matrix N)
)(
    input                                       clk,
    input                                       rst_n,       // active-low

    // ---- control ----
    input                                       wbuf_rdy,     // pulse: start streaming a new block
    input      [WBUF_ADDR_WIDTH-1:0]            base_addr,    // block's top-left address in weight_buffer
    input      [DIM_WIDTH-1:0]                  row_stride,   // full matrix width N (row-major stride)
    input      [$clog2(ARRAY_ROWS+1)-1:0]       valid_rows,   // rows of this block holding real data
    input      [$clog2(ARRAY_COLS+1)-1:0]       valid_cols,   // cols of this block holding real data
    input                                       commit,       // pulse: commit the prefetched block (-> pe_array w_load)
    output                                      busy,         // high from wbuf_rdy until commit
    output                                      ready,        // high once the block is fully prefetched, waiting for commit
    output                                      done,         // high once committed, until the next wbuf_rdy

    // ---- weight_buffer read port ----
    output reg                                  wbuf_rden,
    output reg [WBUF_ADDR_WIDTH-1:0]            wbuf_rdaddress,
    input      [DATA_WIDTH-1:0]                 wbuf_q,

    // ---- pe_array weight-load port ----
    output     [$clog2(ARRAY_ROWS*ARRAY_COLS)-1:0] w_addr,
    output     [DATA_WIDTH-1:0]                 w_data,
    output                                      w_en,
    output                                      w_load
);

    localparam PE_ADDR_WIDTH = $clog2(ARRAY_ROWS*ARRAY_COLS);
    localparam ROW_CNT_WIDTH = $clog2(ARRAY_ROWS+1);
    localparam COL_CNT_WIDTH = $clog2(ARRAY_COLS+1);

    localparam STATE_IDLE    = 2'd0,
               STATE_PRELOAD = 2'd1,   // walking the block, issuing RAM reads
               STATE_DRAIN   = 2'd2,   // last read's data is landing this cycle
               STATE_READY   = 2'd3;   // block fully prefetched, waiting for commit

    reg [1:0] wload_state;
    reg       done_r;

    // ---- position within the current block, row-major over the full
    // K x N matrix (row_stride = N, so a block row is not contiguous) ----
    reg [ROW_CNT_WIDTH-1:0]   cur_row;
    reg [COL_CNT_WIDTH-1:0]   cur_col;
    reg [WBUF_ADDR_WIDTH-1:0] rd_addr;
    reg [PE_ADDR_WIDTH-1:0]   pe_ld_addr;

    wire elem_valid = (cur_row < valid_rows) && (cur_col < valid_cols);
    wire last_elem  = (cur_row == ARRAY_ROWS-1) && (cur_col == ARRAY_COLS-1);

    assign busy  = (wload_state != STATE_IDLE);
    assign ready = (wload_state == STATE_READY);
    assign done  = done_r;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            wload_state    <= STATE_IDLE;
            done_r         <= 1'b0;
            cur_row        <= '0;
            cur_col        <= '0;
            rd_addr        <= '0;
            wbuf_rden      <= 1'b0;
            wbuf_rdaddress <= '0;
            pe_ld_addr     <= '0;
        end
        else begin
            case (wload_state)
                STATE_IDLE: begin
                    if (wbuf_rdy) begin
                        wload_state <= STATE_PRELOAD;
                        done_r      <= 1'b0;
                        cur_row     <= '0;
                        cur_col     <= '0;
                        rd_addr     <= base_addr;
                    end
                end

                STATE_PRELOAD: begin
                    done_r         <= '0;
                    wbuf_rden      <= elem_valid;
                    wbuf_rdaddress <= rd_addr;
                    pe_ld_addr     <= pe_ld_addr + 1;

                    if (cur_col == ARRAY_COLS-1) begin
                        cur_col <= '0;
                        cur_row <= cur_row + 1'b1;
                        rd_addr <= rd_addr + (row_stride - (ARRAY_COLS-1));
                    end
                    else begin
                        cur_col <= cur_col + 1'b1;
                        rd_addr <= rd_addr + 1'b1;
                    end

                    if (last_elem)
                        wload_state <= STATE_DRAIN;
                end

                STATE_DRAIN: begin
                    // last read was issued last cycle; wbuf_q for it is
                    // valid this cycle (weight_buffer's 1-cycle synchronous
                    // read), so one cycle here is always enough
                    wbuf_rden   <= 1'b0;
                    wload_state <= STATE_READY;
                    pe_ld_addr  <= 0;
                end

                STATE_READY: begin
                    // prefetch registers hold the full block; sit here until
                    // weight_ctrl pulses commit (after compute_start fires)
                    if (commit) begin
                        wload_state <= STATE_IDLE;
                        done_r      <= 1'b1;
                    end

                    if(wbuf_rdy)begin
                        wload_state <= STATE_PRELOAD;
                        done_r      <= 1'b1;
                        cur_row     <= '0;
                        cur_col     <= '0;
                        rd_addr     <= base_addr;
                    end
                end
            endcase
        end
    end

    // ---- pair wbuf_q with the (issued, valid, PE index) of the read that
    // produced it. weight_buffer is a synchronous on-chip RAM (q valid one
    // cycle after rdaddress/rden), so a single register stage is enough.
    // Every PE in the block is written every cycle it's walked: in-range
    // PEs get wbuf_q, out-of-range PEs (past valid_rows/valid_cols) get an
    // explicit 0 instead of a stale wbuf_q from the last real read. ----
    reg                     rd_valid_d;
    reg [PE_ADDR_WIDTH-1:0] rd_addr_d;
    reg [PE_ADDR_WIDTH-1:0] rd_addr_2d;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            rd_valid_d  <= 1'b0;
            rd_addr_d   <= '0;
            rd_addr_2d  <= '0;
        end
        else begin
            rd_valid_d  <= wbuf_rden;
            rd_addr_d   <= pe_ld_addr;
            rd_addr_2d  <= rd_addr_d;
        end
    end

    assign w_addr = rd_addr_2d;
    assign w_data = rd_valid_d ? wbuf_q : {DATA_WIDTH{1'b0}};
    assign w_en   = rd_valid_d;
    assign w_load = commit && (wload_state == STATE_READY);

endmodule

// weight_buffer_32_bank : ARRAY_ROWS independent weight_buffer banks, one per
// pe_array row. Bank r holds every weight PE row r ever loads, so each
// bank's read port feeds exactly one row and all ARRAY_ROWS rows prefetch
// in parallel - one weight per row per cycle, a whole block in ARRAY_COLS
// cycles (see weight_loader.v).
//
// Layout: W[k][n] (WEIGHT_ROWS x WEIGHT_COLS, K = PE-row axis, N = PE-column
// axis) lives in bank k % ARRAY_ROWS at byte offset (k / ARRAY_ROWS)*N + n -
// each bank holds its own rows of W back to back, row-major. Block (k_blk,
// n_blk) therefore starts at the same offset, k_blk*N + n_blk*ARRAY_COLS,
// in every bank, and its ARRAY_COLS weights per row are contiguous from
// there, so one shared read offset walks the whole block.
//
// The write side is WIDE (WR_DATA_WIDTH) so an Avalon host moves a full
// word per access; the read side stays DATA_WIDTH (byte) wide because each
// row takes one weight per cycle. wraddress is a linear *word* address,
// split the same way input_buffer_32_bank's is: upper BANK_SEL_WIDTH bits
// pick the bank (PE row), low BANK_WADDR_WIDTH bits are the word offset
// within it. Per-bank byte regions are a power-of-two multiple of the word
// size, so a wide write never straddles two banks.
//
// rd_byteenable is one bit per bank (bit i gates bank i's rden, on top of
// the shared rden) - weight_loader uses it to skip the banks of rows past a
// partial block's valid_rows.
//
// NB: the "32" in the module name only mirrors input_buffer_32_bank /
// output_buffer_32_bank - the bank count is ARRAY_ROWS.
`timescale 1ps/1ps

module weight_buffer_32_bank #(
    parameter ARRAY_ROWS      = 16,
    parameter DATA_WIDTH      = 8,   // int8 weight entries
    parameter BANK_ADDR_WIDTH = 10,  // per-bank READ (byte) address width - match the generated weight_buffer IP's depth (1024)
    parameter WR_DATA_WIDTH   = 32,  // Avalon-side write width
    parameter BANK_SEL_WIDTH  = $clog2(ARRAY_ROWS),
    // per-bank WRITE (word) address width: the same region addressed in
    // WR_DATA_WIDTH/DATA_WIDTH-sized words instead of bytes
    parameter BANK_WADDR_WIDTH = BANK_ADDR_WIDTH - $clog2(WR_DATA_WIDTH/DATA_WIDTH)
) (
    input                                               clock,
    input       [WR_DATA_WIDTH-1:0]                     data,
    input       [ARRAY_ROWS*BANK_ADDR_WIDTH-1:0]        rdaddress,
    input                                               rden,
    input       [ARRAY_ROWS-1:0]                        rd_byteenable,
    input       [BANK_SEL_WIDTH+BANK_WADDR_WIDTH-1:0]   wraddress,
    input                                               wren,

    output wire [DATA_WIDTH*ARRAY_ROWS-1:0]             q

);

    wire [BANK_SEL_WIDTH-1:0]   wr_bank_sel = wraddress[BANK_SEL_WIDTH+BANK_WADDR_WIDTH-1:BANK_WADDR_WIDTH];
    wire [BANK_WADDR_WIDTH-1:0] wr_bank_off = wraddress[BANK_WADDR_WIDTH-1:0];

    genvar i;
    generate
        for(i = 0; i < ARRAY_ROWS; i = i + 1) begin : BANK
            weight_buffer u_weight_buffer (
                .clock     (clock),
                .data      (data),
                .rdaddress (rdaddress[i*BANK_ADDR_WIDTH +: BANK_ADDR_WIDTH]),
                .rden      (rden & rd_byteenable[i]),
                .wraddress (wr_bank_off),
                .wren      (wren & (wr_bank_sel == i)),
                .q         (q[i*DATA_WIDTH +: DATA_WIDTH])
            );
        end
    endgenerate

endmodule

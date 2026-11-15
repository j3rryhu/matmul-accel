// bias_buffer_32_bank : ARRAY_ROWS independent bias_buffer banks, one per
// output_loader lane (pe_array p_out row). Holds one int32 bias per output
// canvas row, in accumulator units (scale input_scale*weight_scale), added
// to the raw sum by output_loader on the final contraction pass when
// ACT_CFG.bias_en is set (see output_loader.v).
//
// Layout: canvas row R's bias lives in bank R % ARRAY_ROWS at entry
// R / ARRAY_ROWS - the same row -> bank mapping as output_buffer_32_bank,
// so lane r (which only ever writes canvas rows in bank r) reads its own
// bank, with the canvas row block as the address, every lane in parallel.
// 16 banks x 16 entries = 256 rows of bias.
//
// The write side is WIDE (WR_DATA_WIDTH) so an Avalon host moves a full
// word per access: software writes the bias vector as one contiguous
// image of little-endian int32s, bias[R] at byte 4*R, so host word w holds
// rows ROWS_PER_WORD*w .. ROWS_PER_WORD*w + ROWS_PER_WORD-1. Those rows sit
// in ROWS_PER_WORD neighbouring banks, all at entry w / BANK_GROUPS: the
// low WORD_SEL_WIDTH bits of wraddress pick the group of banks, the rest
// is the entry, and lane j of the word is bank (group*ROWS_PER_WORD + j)'s
// data. A lane is written only when all of its byte enables are set, so a
// bias entry is always written whole. Write-only from the host, like
// weight_buffer_32_bank/input_buffer_32_bank.
//
// rden/rdaddress are per bank, like output_buffer_32_bank's loader port.
//
// NB: the "32" in the module name only mirrors input_buffer_32_bank /
// weight_buffer_32_bank / output_buffer_32_bank - the bank count is
// ARRAY_ROWS.
`timescale 1ps/1ps

module bias_buffer_32_bank #(
    parameter ARRAY_ROWS      = 16,
    parameter DATA_WIDTH      = 32,   // int32 bias entries (accumulator units)
    parameter BANK_ADDR_WIDTH = 4,    // per-bank address width - match the generated bias_buffer IP's depth (16)
    parameter WR_DATA_WIDTH   = 128,  // Avalon-side write width
    parameter ROWS_PER_WORD   = WR_DATA_WIDTH / DATA_WIDTH,          // bias rows per host word (4)
    parameter WORD_SEL_WIDTH  = $clog2(ARRAY_ROWS / ROWS_PER_WORD),  // picks the group of banks a word fills
    // host WRITE (word) address width: entry index above the group select
    parameter WADDR_WIDTH     = BANK_ADDR_WIDTH + WORD_SEL_WIDTH
) (
    input                                               clock,
    input       [WR_DATA_WIDTH-1:0]                     data,
    input       [WADDR_WIDTH-1:0]                       wraddress,
    input                                               wren,
    input       [WR_DATA_WIDTH/8-1:0]                   wr_byteenable,
    input       [ARRAY_ROWS*BANK_ADDR_WIDTH-1:0]        rdaddress,
    input       [ARRAY_ROWS-1:0]                        rden,

    output wire [DATA_WIDTH*ARRAY_ROWS-1:0]             q

);

    localparam BYTES_PER_ENTRY = DATA_WIDTH / 8;

    // a host word must split into whole entries, over more than one group
    generate
        if ((WR_DATA_WIDTH % DATA_WIDTH) != 0 || (ARRAY_ROWS % ROWS_PER_WORD) != 0
                || ROWS_PER_WORD >= ARRAY_ROWS) begin : g_width_mismatch
            BIAS_BUFFER_needs_whole_entries_per_word_and_more_than_one_group u_err();
        end
    endgenerate

    wire [WORD_SEL_WIDTH-1:0]  wr_grp = wraddress[WORD_SEL_WIDTH-1:0];
    wire [BANK_ADDR_WIDTH-1:0] wr_off = wraddress[WADDR_WIDTH-1:WORD_SEL_WIDTH];

    genvar i;
    generate
        for(i = 0; i < ARRAY_ROWS; i = i + 1) begin : BANK
            // bank i = group i / ROWS_PER_WORD, lane i % ROWS_PER_WORD of the word
            wire lane_be = &wr_byteenable[(i % ROWS_PER_WORD)*BYTES_PER_ENTRY +: BYTES_PER_ENTRY];

            bias_buffer u_bias_buffer (
                .clock     (clock),
                .data      (data[(i % ROWS_PER_WORD)*DATA_WIDTH +: DATA_WIDTH]),
                .rdaddress (rdaddress[i*BANK_ADDR_WIDTH +: BANK_ADDR_WIDTH]),
                .rden      (rden[i]),
                .wraddress (wr_off),
                .wren      (wren & lane_be & (wr_grp == i / ROWS_PER_WORD)),
                .q         (q[i*DATA_WIDTH +: DATA_WIDTH])
            );
        end
    endgenerate

endmodule

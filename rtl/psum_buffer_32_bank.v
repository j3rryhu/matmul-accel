// psum_buffer_32_bank : NUM_BANKS independent psum_buffer banks, one per
// pe_array p_out lane (array row), holding each output element's raw
// 32-bit running sum between contraction blocks (see output_loader.v).
//
// Internal to output_loader: it is the only reader and writer, through a
// full read port and write port per bank (read-modify-write accumulate),
// both one 32-bit slot wide. The host never sees it - finished results go
// to output_buffer as int8, and that is what the host reads.
//
// Layout mirrors output_buffer's: canvas row R in bank R % NUM_BANKS, at
// slot (R / NUM_BANKS) * pitch + column, pitch = the canvas width rounded up
// to a multiple of 4 slots.
//
// NB: the "32" in the module name only mirrors the other *_buffer_32_bank
// wrappers - the bank count is NUM_BANKS.
`timescale 1ps/1ps

module psum_buffer_32_bank #(
    parameter DATA_WIDTH      = 32,   // one raw running sum (pe_array ACC_WIDTH)
    parameter NUM_BANKS       = 16,
    parameter BANK_ADDR_WIDTH = 12    // per-bank address width (slots) - match the generated psum_buffer IP's depth (4096)
)(
    input clock,

    input      [NUM_BANKS-1:0]                   bank_rden,
    input      [NUM_BANKS*BANK_ADDR_WIDTH-1:0]   bank_rdaddress,
    output     [NUM_BANKS*DATA_WIDTH-1:0]        bank_q,
    input      [NUM_BANKS-1:0]                   bank_wren,
    input      [NUM_BANKS*BANK_ADDR_WIDTH-1:0]   bank_waddr,
    input      [NUM_BANKS*DATA_WIDTH-1:0]        bank_wdata
);

    genvar i;
    generate
        for (i = 0; i < NUM_BANKS; i = i + 1) begin : BANK
            psum_buffer u_psum_buffer (
                .clock     (clock),
                .data      (bank_wdata[i*DATA_WIDTH +: DATA_WIDTH]),
                .rdaddress (bank_rdaddress[i*BANK_ADDR_WIDTH +: BANK_ADDR_WIDTH]),
                .rden      (bank_rden[i]),
                .wraddress (bank_waddr[i*BANK_ADDR_WIDTH +: BANK_ADDR_WIDTH]),
                .wren      (bank_wren[i]),
                .q         (bank_q[i*DATA_WIDTH +: DATA_WIDTH])
            );
        end
    endgenerate

endmodule

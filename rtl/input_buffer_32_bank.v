// input_buffer_32_bank : ARRAY_ROWS independent input_buffer banks, one per
// pe_array/input_dispatch row - each bank holds that row's activation
// values contiguously, so a single shared rdaddress pulls one element per
// row every cycle with no reshuffling needed downstream (see
// input_dispatch.v).
//
// The write side is WIDE (AVS_DATA_WIDTH, 32) so an Avalon host moves a
// full word per access and its byte address advances by 4; the read side
// stays DATA_WIDTH (byte) wide because input_dispatch pulls one element per
// row per cycle. wraddress is therefore a linear *word* address, split the
// same way output_buffer_32_bank's is: upper BANK_SEL_WIDTH bits pick the
// bank (row), low BANK_WADDR_WIDTH bits are the word offset within it.
// Per-bank byte regions are a power-of-two multiple of 4 bytes, so bank
// boundaries stay word aligned and a wide write never straddles two banks.
//
// NB: the "32" in the module name is historical - the bank count is
// ARRAY_ROWS and is fully parameterized (kept as-is so the Qsys/Platform
// Designer fileset and tb/Makefile don't need renaming).
//
// rd_byteenable is one bit per bank (bit i gates bank i's rden, on top of
// the shared rden) - lets a caller mask out specific rows' banks from a
// given read (e.g. rows beyond a partial block's valid_rows) without
// touching the shared rdaddress/rden that every other bank still uses.
module input_buffer_32_bank #(
    parameter ARRAY_ROWS      = 16,
    parameter DATA_WIDTH      = 8,   // int8 activation entries
    parameter BANK_ADDR_WIDTH = 8,   // per-bank READ (byte) address width - match the generated input_buffer IP's depth (256)
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
            input_buffer u_input_buffer (
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

// input_buffer_32_bank : 32 independent input_buffer banks, one per
// pe_array/input_dispatch row - each bank holds that row's activation
// values contiguously, so a single shared rdaddress pulls one element per
// row every cycle with no reshuffling needed downstream (see
// input_dispatch.v). wraddress is a single linear address split the same
// way output_buffer_32_bank's is: upper bits pick the bank (row), low bits
// are the offset within it.
//
// rd_byteenable is one bit per bank (bit i gates bank i's rden, on top of
// the shared rden) - lets a caller mask out specific rows' banks from a
// given read (e.g. rows beyond a partial block's valid_rows) without
// touching the shared rdaddress/rden that every other bank still uses.
module input_buffer_32_bank #(
    parameter ARRAY_ROWS = 32
) (
    input                       clock,
    input       [ 7:0]          data,
    input       [ARRAY_ROWS*8-1:0]        rdaddress,
    input                       rden,
    input       [ARRAY_ROWS-1:0]          rd_byteenable,
    input       [12:0]          wraddress,
    input                       wren,

    output wire [8 * ARRAY_ROWS - 1:0]  q

);

    genvar i;
    generate
        for(i = 0; i < 32; i++) begin
            input_buffer u_input_buffer (
                .clock     (clock),
                .data      (data),
                .rdaddress (rdaddress[i*8+:8]),
                .rden      (rden & rd_byteenable[i]),
                .wraddress (wraddress[7:0]),
                .wren      (wren & (wraddress[12:8] == i)),
                .q         (q[i * 8 +: 8])
            );
        end
    endgenerate

endmodule

// input_buffer_32_bank : 32 independent input_buffer banks, one per
// pe_array/input_dispatch row - each bank holds that row's activation
// values contiguously, so a single shared rdaddress pulls one element per
// row every cycle with no reshuffling needed downstream (see
// input_dispatch.v). wraddress is a single linear address split the same
// way output_buffer_32_bank's is: upper bits pick the bank (row), low bits
// are the offset within it.
module input_buffer_32_bank (
    input                       clock,
    input       [ 7:0]          data,
    input       [ 7:0]          rdaddress,
    input                       rden,
    input       [12:0]          wraddress,
    input                       wren,

    output wire [8 * 32 - 1:0]  q

);

    genvar i;
    generate
        for(i = 0; i < 32; i++) begin
            input_buffer u_input_buffer (
                .clock     (clock),
                .data      (data),
                .rdaddress (rdaddress),
                .rden      (rden),
                .wraddress (wraddress[7:0]),
                .wren      (wren & (wraddress[12:8] == i)),
                .q         (q[i * 8 +: 8])
            );
        end
    endgenerate

endmodule

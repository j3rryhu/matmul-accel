module input_buffer_8_bank (
    input                       clock,
    input       [ 7:0]          data,
    input       [ 9:0]          rdaddress,
    input                       rden,
    input       [12:0]          wraddress,
    input                       wren,

    output wire [8 * 8 - 1:0]   q

);

    genvar i;
    generate
        for(i = 0; i < 8; i++) begin
            input_buffer u_input_buffer (
                .clock     (clock),
                .data      (data),
                .rdaddress (rdaddress),
                .rden      (rden),
                .wraddress (wraddress[9:0]),
                .wren      (wren & (wraddress[12:10] == i)),
                .q         (q[i * 8 +: 8])
            );
        end
    endgenerate

endmodule
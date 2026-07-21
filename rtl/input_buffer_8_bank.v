module input_buffer_8_bank (
    input                       clock,
    input       [ 7:0]          data,
    input       [ 9:0]          rdaddress,
    input                       rden,
    input       [12:0]          wraddress,
    input                       wren,

    output wire [7 * 8 - 1:0]   q

);

    genvar i;
    generate
        for(i = 0; i < 8; i++) begin
            input_buffer u_input_buffer (
                .clock     (clk),
                .data      (ibuf_wdata),
                .rdaddress (rdaddress),   // TODO: drive from compute datapath
                .rden      (rden),    // TODO: drive from compute datapath
                .wraddress (ibuf_wraddress[9:0]),
                .wren      (ibuf_wren & ibuf_wraddress[12:10] == i),
                .q         (out_rdata[i * 8 +: 8])         // unused until compute datapath is wired up
            );
        end
    endgenerate

endmodule
// Input is the signed int8 value AFTER bias addition and requantization.
// LUT_FILE must match the agreed input and output real-value scales.
module gelu_int8 #(
    parameter LUT_FILE = "activation/example/gelu_int8.hex"
) (
    input wire clk,
    input wire reset_n,
    input wire signed [7:0] in_data,
    input wire in_valid,
    output wire in_ready,
    output reg signed [7:0] out_data,
    output reg out_valid,
    input wire out_ready
);
    reg [7:0] rom [0:255];
    initial $readmemh(LUT_FILE, rom);
    // One-entry output buffer: hold both data and valid during backpressure.
    assign in_ready = reset_n && (!out_valid || out_ready);
    always @(posedge clk or negedge reset_n) begin
        if (!reset_n) begin
            out_valid <= 1'b0;
            out_data <= 8'sd0;
        end else if (in_ready) begin
            out_valid <= in_valid;
            if (in_valid)
                out_data <= rom[$unsigned(in_data)];
        end
    end
endmodule

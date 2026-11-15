// LANES-wide (default 16 x int8 = 128-bit) bias + GELU stage.
// Contract: matmul_data and bias_data share the GELU LUT input scale.
// Lane l occupies bits [8*l+7 : 8*l] of every wide data bus.
// bias_data and in_mask are paired with matmul_data on every accepted transfer.
//
// in_mask[l] = 1: out lane l = GELU(sat8(matmul_l + bias_l)).
// in_mask[l] = 0: out lane l = matmul_l unchanged (bias_l is don't-care, may be X).
// out_mask is the accepted in_mask, aligned with out_data (like AXIS tkeep).
//
// Handshake and latency are identical to gelu_int8: one shared ready/valid
// pair for all lanes, one-entry output buffer, peak one wide word per clock.
module bias_gelu_int8 #(
    parameter LANES = 16,
    parameter LUT_FILE = "activation/example/gelu_int8.hex"
) (
    input wire clk,
    input wire reset_n,
    input wire [LANES*8-1:0] matmul_data,
    input wire [LANES*8-1:0] bias_data,
    input wire [LANES-1:0] in_mask,
    input wire in_valid,
    output wire in_ready,
    output wire [LANES*8-1:0] out_data,
    output reg [LANES-1:0] out_mask,
    output reg out_valid,
    input wire out_ready
);
    // Per-lane adder results, flattened so testbenches can probe them.
    wire [LANES*9-1:0] wide_sum;
    wire [LANES*8-1:0] clipped_sum;

    wire accept = in_valid && in_ready;
    assign in_ready = reset_n && (!out_valid || out_ready);

    // Control: valid and mask are reset.
    always @(posedge clk or negedge reset_n) begin
        if (!reset_n) begin
            out_valid <= 1'b0;
            out_mask <= {LANES{1'b0}};
        end else if (in_ready) begin
            out_valid <= in_valid;
            if (in_valid)
                out_mask <= in_mask;
        end
    end

    genvar l;
    generate
        for (l = 0; l < LANES; l = l + 1) begin : lane
            wire signed [7:0] m = matmul_data[l*8 +: 8];
            wire signed [7:0] b = bias_data[l*8 +: 8];
            wire signed [8:0] s = $signed({m[7], m}) + $signed({b[7], b});
            wire [7:0] c = (s > 9'sd127) ? 8'h7f :
                           (s < -9'sd128) ? 8'h80 : s[7:0];
            assign wide_sum[l*9 +: 9] = s;
            assign clipped_sum[l*8 +: 8] = c;

            // One ROM per lane so every lane can look up in the same cycle.
            // Plain registered read with clock enable and no reset keeps the
            // ROM inferable as block RAM, so the bypass byte gets its own
            // register and the lane select happens after both registers.
            reg [7:0] rom [0:255];
            reg [7:0] rom_q;
            reg [7:0] raw_q;
            initial $readmemh(LUT_FILE, rom);
            always @(posedge clk) begin
                if (accept)
                    rom_q <= rom[c];
            end
            // Reset to 0: after reset out_mask is 0, which selects raw_q.
            always @(posedge clk or negedge reset_n) begin
                if (!reset_n)
                    raw_q <= 8'h00;
                else if (accept)
                    raw_q <= m;
            end
            assign out_data[l*8 +: 8] = out_mask[l] ? rom_q : raw_q;
        end
    endgenerate
endmodule

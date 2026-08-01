// weight-stationary processing element (PE)
// mac : p_out = weight * a + b_in   (32-bit signed, wraps on overflow)
// b_in is the incoming partial sum from the previous PE in the chain;
// p_out (the new partial sum) is forwarded on to the next PE.
// operand a is buffered and also passed on to the next PE (a_out).
`timescale 1ps/1ps

module pe #(
    parameter DATA_WIDTH = 32
)(
    input                       clk,
    input                       rst_n,

    // input operand a (activation) - buffered, then forwarded to next PE
    input  [DATA_WIDTH-1:0]     a_in,
    input                       a_en,

    // incoming partial sum - buffered, accumulated into, then forwarded as p_out
    input  [DATA_WIDTH-1:0]     b_in,
    input                       b_en,

    // weight load path : prefetch register -> stationary weight register
    input  [DATA_WIDTH-1:0]     w_in,
    input                       w_en,       // latch w_in into the prefetch register
    input                       w_load,     // commit prefetched weight into the active weight register

    // outputs
    output reg [DATA_WIDTH-1:0] p_out,      // mac result : weight * a + b_in, passed to next PE
    output reg [DATA_WIDTH-1:0] a_out       // buffered a, passed on to the next PE
);

    // ---- internal registers ----
    reg [DATA_WIDTH-1:0] weight_reg;    // stationary weight used by the mac
    reg [DATA_WIDTH-1:0] weight_pf;     // weight prefetch register
    reg [DATA_WIDTH-1:0] a_buf;         // input a buffer
    reg [DATA_WIDTH-1:0] b_buf;         // input b (partial sum) buffer

    // ---- weight prefetch register ----
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n)
            weight_pf <= {DATA_WIDTH{1'b0}};
        else if (w_en)
            weight_pf <= w_in;
    end

    // ---- stationary weight register (loaded from the prefetch register) ----
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n)
            weight_reg <= {DATA_WIDTH{1'b0}};
        else if (w_load)
            weight_reg <= weight_pf;
    end

    // ---- input a buffer (also forwarded to the next PE) ----
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            a_buf <= {DATA_WIDTH{1'b0}};
            a_out <= {DATA_WIDTH{1'b0}};
        end
        else if (a_en) begin
            a_buf <= a_in;
            a_out <= a_in;
        end
    end

    // ---- input b (partial sum) buffer ----
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n)
            b_buf <= {DATA_WIDTH{1'b0}};
        else if (b_en)
            b_buf <= b_in;
    end

    // ---- 32-bit signed multiply-accumulate ----
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n)
            p_out <= {DATA_WIDTH{1'b0}};
        else
            p_out <= ($signed(weight_reg) * $signed(a_in)) + $signed(b_in);
    end

endmodule

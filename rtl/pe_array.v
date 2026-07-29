// weight-stationary systolic PE array
// activations (a) shift left -> right along each row
// partial sums  (b) shift top  -> bottom along each column
// a_en/b_en are broadcast per row/column (not per PE) since the array is a
// synchronous shift chain - a per-PE enable would need its own delay chain
// to stay aligned with data already in flight.
// weight prefetch is a single regfile-style write port: one PE's prefetch
// register is written per cycle, selected by w_addr (row*ARRAY_COLS+col)
// while w_en is high. w_load is a single global pulse that commits every
// PE's prefetched weight into its active weight register on the same cycle.
`timescale 1ps/1ps

module pe_array #(
    parameter DATA_WIDTH  = 32,
    parameter ARRAY_ROWS  = 32,
    parameter ARRAY_COLS  = 32
)(
    input                                           clk,
    input                                           rst_n,

    // activation input, left edge - one lane per row
    input  [ARRAY_ROWS*DATA_WIDTH-1:0]              a_in,
    input  [ARRAY_ROWS-1:0]                         a_en,

    // partial-sum enable, one lane per column (row 0's b_in is tied to 0 -
    // the array always starts accumulation fresh, no external cascade-in)
    input  [ARRAY_COLS-1:0]                         b_en,

    // weight prefetch write port, regfile-style: one PE per cycle
    input  [$clog2(ARRAY_ROWS*ARRAY_COLS)-1:0]      w_addr,  // row*ARRAY_COLS+col
    input  [DATA_WIDTH-1:0]                         w_data,
    input                                            w_en,
    input                                            w_load,

    // activation output, right edge - one lane per row (for tile cascading)
    output [ARRAY_ROWS*DATA_WIDTH-1:0]              a_out,

    // partial-sum output, bottom edge - one lane per column (final results)
    output [ARRAY_COLS*DATA_WIDTH-1:0]              p_out
);

    genvar r, c;

    generate
        for (r = 0; r < ARRAY_ROWS; r = r + 1) begin : ROW
            for (c = 0; c < ARRAY_COLS; c = c + 1) begin : COL

                wire [DATA_WIDTH-1:0] pe_a_in;
                wire [DATA_WIDTH-1:0] pe_b_in;
                wire [DATA_WIDTH-1:0] pe_a_out;
                wire [DATA_WIDTH-1:0] pe_p_out;

                // this PE's prefetch reg is written when the incoming
                // address matches its flat index and w_en is asserted
                wire pe_w_en = w_en && (w_addr == (r*ARRAY_COLS+c));

                if (c == 0)
                    assign pe_a_in = a_in[(r+1)*DATA_WIDTH-1 -: DATA_WIDTH];
                else
                    assign pe_a_in = ROW[r].COL[c-1].pe_a_out;

                if (r == 0)
                    assign pe_b_in = {DATA_WIDTH{1'b0}};
                else
                    assign pe_b_in = ROW[r-1].COL[c].pe_p_out;

                pe #(
                    .DATA_WIDTH (DATA_WIDTH)
                ) u_pe (
                    .clk     (clk),
                    .rst_n   (rst_n),
                    .a_in    (pe_a_in),
                    .a_en    (a_en[r]),
                    .b_in    (pe_b_in),
                    .b_en    (b_en[c]),
                    .w_in    (w_data),
                    .w_en    (pe_w_en),
                    .w_load  (w_load),
                    .p_out   (pe_p_out),
                    .a_out   (pe_a_out)
                );

                if (c == ARRAY_COLS-1)
                    assign a_out[(r+1)*DATA_WIDTH-1 -: DATA_WIDTH] = pe_a_out;

                if (r == ARRAY_ROWS-1)
                    assign p_out[(c+1)*DATA_WIDTH-1 -: DATA_WIDTH] = pe_p_out;

            end
        end
    endgenerate

endmodule

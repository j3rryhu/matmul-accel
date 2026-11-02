// buf_addr_remap : turns a contiguous host write address into the banked
// {bank, word offset} address of weight_buffer_32_bank / input_buffer_32_bank,
// so software can write a plain row-major matrix image.
//
// Host image: row-major, every row padded up to a whole number of
// WR_DATA_WIDTH words (row_words words per row), rows back to back. The
// padding is what makes this a pure address remap: every host word then
// belongs to exactly one row, so it lands whole in one bank at a
// word-aligned offset. Row p goes to bank p % NUM_BANKS, and each bank
// holds its rows back to back:
//
//   w    = host word address
//   p    = w / row_words,  c = w % row_words
//   bank = p % NUM_BANKS
//   off  = (p / NUM_BANKS) * row_words + c       (bank word offset)
//
// The divide is a multiply by a reciprocal: p = (w * ceil(2^S / R)) >> S
// with S = HOST_WADDR_WIDTH + ROWW_WIDTH. With e = ceil(2^S/R)*R - 2^S < R
// the error term w*e/2^S is < 1 for every w < 2^HOST_WADDR_WIDTH, so the
// floor is exact. The reciprocal comes from a constant table indexed by
// row_words and is registered - row_words is a control register that
// software programs before writing the buffer.
//
// Two register stages: write address/data/enable come out two cycles after
// they go in. That is safe against matmul_start - the start write is
// accepted at least one cycle after the last buffer write and takes several
// more cycles to reach the first buffer read.
`timescale 1ps/1ps

(* multstyle = "logic" *)
module buf_addr_remap #(
    parameter DATA_WIDTH       = 128,  // host write word width
    parameter HOST_WADDR_WIDTH = 10,   // contiguous host word address width
    parameter ROWW_WIDTH       = 7,    // width of row_words (row pitch in words)
    parameter BANK_SEL_WIDTH   = 4,    // log2(NUM_BANKS)
    parameter BANK_WADDR_WIDTH = 6     // per-bank word address width
)(
    input                                            clk,

    input      [HOST_WADDR_WIDTH-1:0]                host_waddr,
    input      [DATA_WIDTH-1:0]                      host_wdata,
    input                                            host_wren,
    input      [ROWW_WIDTH-1:0]                      row_words,   // >= 1; 0 is treated as 1

    output reg [BANK_SEL_WIDTH+BANK_WADDR_WIDTH-1:0] bank_waddr,  // {bank, word offset}
    output reg [DATA_WIDTH-1:0]                      bank_wdata,
    output reg                                       bank_wren
);

    localparam S       = HOST_WADDR_WIDTH + ROWW_WIDTH;
    localparam NUM_R   = 1 << ROWW_WIDTH;

    // ---- reciprocal table: RECIP[R] = ceil(2^S / R), RECIP[0] = RECIP[1] ----
    wire [S:0] RECIP [0:NUM_R-1];
    genvar g;
    generate
        for (g = 0; g < NUM_R; g = g + 1) begin : RECIP_ROM
            if (g == 0)
                assign RECIP[g] = (1 << S);
            else
                assign RECIP[g] = ((1 << S) + g - 1) / g;
        end
    endgenerate

    reg [S:0]            recip_r;
    reg [ROWW_WIDTH-1:0] rw_r;
    always @(posedge clk) begin
        recip_r <= RECIP[row_words];
        rw_r    <= (row_words == 0) ? {{(ROWW_WIDTH-1){1'b0}}, 1'b1} : row_words;
    end

    // ---- stage 1: row index p = w / row_words ----
    wire [HOST_WADDR_WIDTH+S:0] prod = host_waddr * recip_r;

    reg [HOST_WADDR_WIDTH-1:0] p_1;
    reg [HOST_WADDR_WIDTH-1:0] w_1;
    reg [DATA_WIDTH-1:0]       d_1;
    reg                        en_1;
    always @(posedge clk) begin
        p_1  <= prod[S +: HOST_WADDR_WIDTH];
        w_1  <= host_waddr;
        d_1  <= host_wdata;
        en_1 <= host_wren;
    end

    // ---- stage 2: bank = p % NUM_BANKS, off = (p / NUM_BANKS)*R + (w - p*R) ----
    wire [HOST_WADDR_WIDTH-1:0] col_w = w_1 - p_1 * rw_r;
    wire [HOST_WADDR_WIDTH-1:0] off   = (p_1 >> BANK_SEL_WIDTH) * rw_r + col_w;

    always @(posedge clk) begin
        bank_waddr <= {p_1[BANK_SEL_WIDTH-1:0], off[BANK_WADDR_WIDTH-1:0]};
        bank_wdata <= d_1;
        bank_wren  <= en_1;
    end

endmodule

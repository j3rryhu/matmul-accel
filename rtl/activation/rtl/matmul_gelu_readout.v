// Optional postprocessing of FINAL output-buffer reads. Never changes the
// partial sums stored in RAM. Operands and LUT input share a real-value scale.
module matmul_gelu_readout #(
    parameter DATA_WIDTH = 128,
    parameter BANK_ADDR_WIDTH = 7,
    parameter BANKS = 16,
    parameter LUT_FILE = "activation/example/gelu_int8.hex"
) (
    input clock,
    input reset_n,
    input config_write,
    input [15:0] config_address,
    input [DATA_WIDTH-1:0] config_data,
    input [DATA_WIDTH/8-1:0] config_byteenable,
    input [15:0] columns,
    input [$clog2(BANKS)+BANK_ADDR_WIDTH-$clog2(DATA_WIDTH/8)-1:0] read_word_address,
    input [DATA_WIDTH-1:0] raw_data,
    output reg [DATA_WIDTH-1:0] result
);
    localparam BYTES = DATA_WIDTH/8;
    localparam ADDRESS_BITS = $clog2(BANKS)+BANK_ADDR_WIDTH;
    reg [7:0] bias [0:255];
    reg [7:0] rom [0:255];
    reg enabled;
    reg [ADDRESS_BITS-1:0] read_byte_address;
    integer i, lane;
    wire [BANK_ADDR_WIDTH-1:0] bank_offset = read_byte_address[BANK_ADDR_WIDTH-1:0];
    // All live row segments must be wide-word aligned. One bias per word.
    wire geometry_ok = columns != 0 && (columns % BYTES) == 0;
    wire [BANK_ADDR_WIDTH-1:0] divisor = columns[BANK_ADDR_WIDTH-1:0];
    wire [BANK_ADDR_WIDTH-1:0] segment =
        columns >= (1 << BANK_ADDR_WIDTH) ? 0 :
        (divisor == 0 ? 0 : bank_offset / divisor);
    wire [15:0] row_index = (segment * BANKS) +
        (read_byte_address >> BANK_ADDR_WIDTH);
    wire signed [7:0] row_bias = bias[row_index[7:0]];
    reg signed [8:0] sum;
    reg [7:0] clipped;
    initial $readmemh(LUT_FILE, rom);
    always @(posedge clock) begin
        if (!reset_n) begin
            enabled <= 0;
            read_byte_address <= 0;
            for (i=0; i<256; i=i+1) bias[i] <= 0;
        end else begin
            // Match the existing output RAM's one-clock read latency.
            read_byte_address <= read_word_address * BYTES;
            if (config_write) begin
                for (i=0; i<BYTES; i=i+1) begin
                    if (config_byteenable[i]) begin
                        if (config_address[15:8] == 8'h80)
                            bias[(config_address[7:0] & (256-BYTES))+i] <= config_data[i*8 +: 8];
                        if ((config_address & ~(BYTES-1))+i == 16'h8100)
                            enabled <= config_data[i*8];
                    end
                end
            end
        end
    end
    always @* begin
        result = raw_data;
        sum = 0; clipped = 0;
        for (lane=0; lane<BYTES; lane=lane+1) begin
            if (enabled && geometry_ok && row_index < 256) begin
                sum = $signed({raw_data[lane*8+7], raw_data[lane*8 +: 8]}) +
                      $signed({row_bias[7], row_bias});
                clipped = sum > 9'sd127 ? 8'd127 :
                          (sum < -9'sd128 ? 8'h80 : sum[7:0]);
                result[lane*8 +: 8] = rom[clipped];
            end
        end
    end
endmodule

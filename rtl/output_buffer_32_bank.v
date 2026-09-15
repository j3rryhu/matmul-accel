// output_buffer_32_bank : NUM_BANKS independent output_buffer banks, one per
// pe_array column - mirrors input_buffer_32_bank's split-address-space
// pattern, but output_loader needs independent read+write per bank
// (read-modify-write accumulation) rather than input_buffer_32_bank's
// single shared read address, so each bank gets its own full port here.
//
// output_loader drives every bank's read+write directly (bank_r*/bank_w*
// buses below, one ARRAY_COLS-wide slice per bank). Avalon only ever
// reads (software reads results out after a matmul), never writes, and
// only does so once output_loader is idle - per the programming sequence,
// software polls STATUS.busy/done before reading, so there's no window
// where Avalon and output_loader both need the same bank's read port at
// once. Given that, each bank's read address/enable is simply muxed by
// `busy` (output_loader's own busy, not the whole matmul's): output_loader
// wins while busy, the Avalon-derived address wins once idle.
//
// The Avalon-facing read port is WIDE (AVS_DATA_WIDTH, 32) so a host read
// moves a full word and its byte address advances by 4; output_loader's own
// per-bank write side stays DATA_WIDTH (byte) wide because it writes one
// rescaled result per bank per cycle. ext_rdaddress is therefore a linear
// *word* address, split the same way input_buffer_32_bank's write address
// is: upper bits pick the bank, low bits are the word offset within it.
// Per-bank byte regions are a power-of-two multiple of 4 bytes, so bank
// boundaries stay word aligned and a wide read never straddles two banks.
//
// NB: the "32" in the module name is historical - the bank count is
// NUM_BANKS and is fully parameterized (kept as-is so the Qsys/Platform
// Designer fileset and tb/Makefile don't need renaming).
`timescale 1ps/1ps

module output_buffer_32_bank #(
    parameter DATA_WIDTH      = 8,   // int8 - output_loader rescales pe_array's ACC_WIDTH accumulator down to this before writing (see output_loader.v)
    parameter NUM_BANKS       = 16,
    parameter BANK_ADDR_WIDTH = 7,    // per-bank WRITE (byte) address width - match the generated output_buffer IP's depth
    parameter RD_DATA_WIDTH   = 32,   // Avalon-side read width
    // per-bank READ (word) address width: the same region addressed in
    // RD_DATA_WIDTH/DATA_WIDTH-sized words instead of bytes
    parameter BANK_RADDR_WIDTH = BANK_ADDR_WIDTH - $clog2(RD_DATA_WIDTH/DATA_WIDTH)
)(
    input clock,

    // ---- output_loader per-bank read+write port ----
    input                                        busy,             // gates the read-address mux, see header note
    input      [NUM_BANKS-1:0]                   bank_rden,
    input      [NUM_BANKS*BANK_RADDR_WIDTH-1:0]  bank_rdaddress,
    output     [NUM_BANKS*RD_DATA_WIDTH-1:0]     bank_q,
    input      [NUM_BANKS-1:0]                   bank_wren,
    input      [NUM_BANKS*BANK_ADDR_WIDTH-1:0]   bank_waddr,
    input      [NUM_BANKS*DATA_WIDTH-1:0]        bank_wdata,

    // ---- Avalon-facing external read port (wide) ----
    input                                                    ext_rden,
    input      [$clog2(NUM_BANKS)+BANK_RADDR_WIDTH-1:0]      ext_rdaddress,
    output     [RD_DATA_WIDTH-1:0]                           ext_q
);

    localparam BANK_SEL_WIDTH = $clog2(NUM_BANKS);

    wire [BANK_SEL_WIDTH-1:0]   ext_bank_sel = ext_rdaddress[BANK_SEL_WIDTH+BANK_RADDR_WIDTH-1:BANK_RADDR_WIDTH];
    wire [BANK_RADDR_WIDTH-1:0] ext_bank_off = ext_rdaddress[BANK_RADDR_WIDTH-1:0];

    genvar i;
    generate
        for (i = 0; i < NUM_BANKS; i = i + 1) begin : BANK
            wire [BANK_RADDR_WIDTH-1:0] rdaddr_mux = busy ? bank_rdaddress[i*BANK_RADDR_WIDTH +: BANK_RADDR_WIDTH] : ext_bank_off;
            wire                       rden_mux   = busy ? bank_rden[i] : (ext_rden && (ext_bank_sel == i));

            output_buffer u_output_buffer (
                .clock     (clock),
                .data      (bank_wdata[i*DATA_WIDTH +: DATA_WIDTH]),
                .rdaddress (rdaddr_mux),
                .rden      (rden_mux),
                .wraddress (bank_waddr[i*BANK_ADDR_WIDTH +: BANK_ADDR_WIDTH]),
                .wren      (bank_wren[i]),
                .q         (bank_q[i*RD_DATA_WIDTH +: RD_DATA_WIDTH])
            );
        end
    endgenerate

    // Avalon readout: q of whichever bank was selected while idle. One
    // cycle of registered-read latency applies same as any other bank
    // access - accel_top's obuf_rd_pending already handles that stall.
    reg [BANK_SEL_WIDTH-1:0] ext_bank_sel_d;
    always @(posedge clock)
        ext_bank_sel_d <= ext_bank_sel;

    assign ext_q = bank_q[ext_bank_sel_d*RD_DATA_WIDTH +: RD_DATA_WIDTH];

endmodule

// Simulation-only behavioral stand-ins for the buffer-RAM IP referenced by
// rtl/accel_top.sv, rtl/input_buffer_32_bank.v, and rtl/output_buffer_32_bank.v
// (weight_buffer, input_buffer, output_buffer).
//
// All three are MIXED-WIDTH simple dual-port RAMs: the Avalon-facing port is
// AVS_DATA_WIDTH (32) wide so a host access moves a full word and the byte
// address advances by 4, while the core-facing port stays byte-wide because
// weight_loader/input_dispatch read one byte per cycle and output_loader
// writes one byte per cycle.
//
//   weight_buffer : wide WRITE  (Avalon)      / narrow READ  (weight_loader)
//   input_buffer  : wide WRITE  (Avalon)      / narrow READ  (input_dispatch)
//   output_buffer : narrow WRITE (output_loader) / wide READ (Avalon)
//
// Byte ordering matches Altera's mixed-width altsyncram convention (and
// Avalon byte lanes): narrow element i of a wide word sits at bit
// [i*NARROW_WIDTH +: NARROW_WIDTH], so narrow address 4n+0 is the wide
// word's least significant byte.
//
// These are NOT meant for synthesis. Swap this file out of the compile
// fileset for the real generated IP (same module names/ports) once it
// exists - the generated On-Chip Memory must be configured with matching
// mixed port widths.
`timescale 1ps/1ps

// Wide write port, narrow read port.
module mixed_ram_wr_wide #(
    parameter NARROW_WIDTH = 8,
    parameter WIDE_WIDTH   = 32,
    parameter NARROW_DEPTH = 256
)(
    input                                       clock,
    input      [WIDE_WIDTH-1:0]                 data,
    input      [$clog2(NARROW_DEPTH)-1:0]       rdaddress,
    input                                       rden,
    input      [$clog2(NARROW_DEPTH*NARROW_WIDTH/WIDE_WIDTH)-1:0] wraddress,
    input                                       wren,
    output reg [NARROW_WIDTH-1:0]               q
);
    localparam RATIO = WIDE_WIDTH / NARROW_WIDTH;

    reg [NARROW_WIDTH-1:0] mem [0:NARROW_DEPTH-1];

    integer i;
    always @(posedge clock) begin
        if (wren)
            for (i = 0; i < RATIO; i = i + 1)
                mem[wraddress*RATIO + i] <= data[i*NARROW_WIDTH +: NARROW_WIDTH];
        if (rden)
            q <= mem[rdaddress];
    end
endmodule

// Narrow write port, wide read port.
module mixed_ram_rd_wide #(
    parameter NARROW_WIDTH = 8,
    parameter WIDE_WIDTH   = 32,
    parameter NARROW_DEPTH = 128
)(
    input                                       clock,
    input      [NARROW_WIDTH-1:0]               data,
    input      [$clog2(NARROW_DEPTH*NARROW_WIDTH/WIDE_WIDTH)-1:0] rdaddress,
    input                                       rden,
    input      [$clog2(NARROW_DEPTH)-1:0]       wraddress,
    input                                       wren,
    output reg [WIDE_WIDTH-1:0]                 q
);
    localparam RATIO = WIDE_WIDTH / NARROW_WIDTH;

    reg [NARROW_WIDTH-1:0] mem [0:NARROW_DEPTH-1];

    integer i;
    always @(posedge clock) begin
        if (wren)
            mem[wraddress] <= data;
        if (rden)
            for (i = 0; i < RATIO; i = i + 1)
                q[i*NARROW_WIDTH +: NARROW_WIDTH] <= mem[rdaddress*RATIO + i];
    end
endmodule

// weight_buffer: 16384 bytes. Avalon writes 32-bit words (4096 x 32,
// 12-bit word address); weight_loader reads bytes (16384 x 8, 14-bit).
module weight_buffer (
    input         clock,
    input  [127:0] data,
    input  [13:0] rdaddress,
    input         rden,
    input  [ 9:0] wraddress,
    input         wren,
    output [ 7:0] q
);
    mixed_ram_wr_wide #(
        .NARROW_WIDTH (8),
        .WIDE_WIDTH   (128),
        .NARROW_DEPTH (16384)
    ) u_ram (
        .clock (clock), .data (data),
        .rdaddress (rdaddress), .rden (rden),
        .wraddress (wraddress), .wren (wren), .q (q)
    );
endmodule

// input_buffer: one bank of input_buffer_32_bank, 256 bytes. Avalon writes
// 32-bit words (64 x 32, 6-bit word address); input_dispatch reads bytes
// (256 x 8, 8-bit).
module input_buffer (
    input         clock,
    input  [127:0] data,
    input  [ 7:0] rdaddress,
    input         rden,
    input  [ 3:0] wraddress,
    input         wren,
    output [ 7:0] q
);
    mixed_ram_wr_wide #(
        .NARROW_WIDTH (8),
        .WIDE_WIDTH   (128),
        .NARROW_DEPTH (256)
    ) u_ram (
        .clock (clock), .data (data),
        .rdaddress (rdaddress), .rden (rden),
        .wraddress (wraddress), .wren (wren), .q (q)
    );
endmodule

// output_buffer: one bank of output_buffer_32_bank, 128 bytes.
// output_loader writes bytes (128 x 8, 7-bit); Avalon reads 32-bit words
// (32 x 32, 5-bit word address).
module output_buffer (
    input         clock,
    input  [ 7:0] data,
    input  [ 2:0] rdaddress,
    input         rden,
    input  [ 6:0] wraddress,
    input         wren,
    output [127:0] q
);
    mixed_ram_rd_wide #(
        .NARROW_WIDTH (8),
        .WIDE_WIDTH   (128),
        .NARROW_DEPTH (128)
    ) u_ram (
        .clock (clock), .data (data),
        .rdaddress (rdaddress), .rden (rden),
        .wraddress (wraddress), .wren (wren), .q (q)
    );
endmodule

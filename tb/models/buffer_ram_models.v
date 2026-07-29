// Simulation-only behavioral stand-ins for the buffer-RAM IP referenced by
// rtl/accel_top.sv, rtl/input_buffer_8_bank.v, and rtl/output_buffer_32_bank.v
// (weight_buffer, input_buffer, output_buffer). Each is a plain flop array:
// synchronous write, registered (1-cycle-latency) read - matching the
// "synchronous on-chip RAM" assumption documented throughout weight_ctrl.v/
// weight_loader.v/output_loader.v.
//
// These are NOT meant for synthesis. Swap this file out of the compile
// fileset (and drop in the real generated IP, same module names/ports) once
// that IP exists - nothing elsewhere needs to change, since callers only
// know these three module names and their clock/data/rdaddress/rden/
// wraddress/wren/q interface.
`timescale 1ps/1ps

module flop_ram #(
    parameter WIDTH      = 8,
    parameter ADDR_WIDTH = 10
)(
    input                       clock,
    input      [WIDTH-1:0]      data,
    input      [ADDR_WIDTH-1:0] rdaddress,
    input                       rden,
    input      [ADDR_WIDTH-1:0] wraddress,
    input                       wren,
    output reg [WIDTH-1:0]      q
);
    reg [WIDTH-1:0] mem [0:(1<<ADDR_WIDTH)-1];

    always @(posedge clock) begin
        if (wren)
            mem[wraddress] <= data;
        if (rden)
            q <= mem[rdaddress];
    end
endmodule

// Stamps out a concretely-named module (matching a real IP's name exactly)
// as a thin flop_ram wrapper at the given address width, 8 bits wide.
//
// NB: the macro argument below is named AW, not ADDR_WIDTH - naive
// text-substitution macros rewrite *every* matching token, including the
// ADDR_WIDTH inside ".ADDR_WIDTH(...)"'s named parameter connection, so
// reusing that exact name here would mangle it into ".14(14)" or similar.
`define DEFINE_BUFFER_RAM(NAME, AW) \
module NAME ( \
    input               clock, \
    input      [7:0]    data, \
    input      [AW-1:0] rdaddress, \
    input               rden, \
    input      [AW-1:0] wraddress, \
    input               wren, \
    output     [7:0]    q \
); \
    flop_ram #( \
        .WIDTH      (8), \
        .ADDR_WIDTH (AW) \
    ) u_flop_ram ( \
        .clock     (clock), \
        .data      (data), \
        .rdaddress (rdaddress), \
        .rden      (rden), \
        .wraddress (wraddress), \
        .wren      (wren), \
        .q         (q) \
    ); \
endmodule

// weight_buffer: WBUF_SIZE=0x4000 (16384 bytes), 14-bit address
`DEFINE_BUFFER_RAM(weight_buffer, 14)

// input_buffer: one bank of input_buffer_8_bank, 1024 bytes/bank, 10-bit address
`DEFINE_BUFFER_RAM(input_buffer, 10)

// output_buffer: one bank of output_buffer_32_bank, 128 bytes/bank, 7-bit address
`DEFINE_BUFFER_RAM(output_buffer, 7)

`undef DEFINE_BUFFER_RAM

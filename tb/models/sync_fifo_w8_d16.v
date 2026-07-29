// Simulation-only stand-in for the sync_fifo_w8_d16 IP referenced by
// rtl/input_dispatch.v (one per array row). Just wraps the existing
// generic rtl/sync_fifo.v at width=8/depth=16, renamed/re-pinned to match
// whatever vendor FIFO IP this name is meant to represent (clock/data/
// rdreq/wrreq/full/empty/q, no reset pin).
//
// input_dispatch.v doesn't wire a reset into this instance (matching a
// real FIFO IP with no reset port), so this wrapper generates its own
// one-shot power-on reset for the underlying sync_fifo, which does need
// one. That's a simulation-only convenience - real hardware comes up in a
// known state after configuration without needing an explicit pin here.
//
// Not meant for synthesis - swap for the real generated IP (same module
// name/ports) once it exists.
`timescale 1ps/1ps

module sync_fifo_w8_d16 (
    input        clock,
    input  [7:0] data,
    input        rdreq,
    input        wrreq,
    output       full,
    output       empty,
    output [7:0] q
);

    reg int_rst_n;
    initial begin
        int_rst_n = 1'b0;
        #1 int_rst_n = 1'b1;
    end

    sync_fifo #(
        .FIFO_WR_WIDTH (8),
        .FIFO_WR_DEPTH (16),
        .FIFO_RD_WIDTH (8),
        .FIFO_RD_DEPTH (16)
    ) u_sync_fifo (
        .clk      (clock),
        .rst_n    (int_rst_n),
        .data_in  (data),
        .wr_en    (wrreq),
        .rd_en    (rdreq),
        .full     (full),
        .empty    (empty),
        .data_out (q)
    );

endmodule

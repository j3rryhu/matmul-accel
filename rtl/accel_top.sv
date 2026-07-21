// Top level: Avalon-MM slave <-> ctrl_rf_rf register file + weight/input/output
// buffer RAMs. Address decode is a flat range check per region; each region's
// local (offset-subtracted) address feeds that region's RAM/regfile directly.
//
// pe_array is instantiated but intentionally left unconnected to the buffer
// RAMs - the RAM<->array datapath (weight load sequencing, activation
// streaming, result writeback) is a separate, not-yet-designed piece of work.
//
// Requires rdl/ctrl_reg.sv/ctrl_rf_rf.sv in the same compile fileset.
`timescale 1ps/1ps

module accel_top #(
    parameter AVS_ADDR_WIDTH = 32,
    parameter AVS_DATA_WIDTH = 32,   // byte-lane mux below assumes 32
    parameter PE_DATA_WIDTH  = 8,
    parameter ARRAY_ROWS     = 32,
    parameter ARRAY_COLS     = 32
)(
    input                                  clk,
    input                                  rst_n,      // active-low

    // Avalon-MM slave
    input        [AVS_ADDR_WIDTH-1:0]      avs_address,
    input                                  avs_read,
    input                                  avs_write,
    input        [AVS_DATA_WIDTH-1:0]      avs_writedata,
    input        [AVS_DATA_WIDTH/8-1:0]    avs_byteenable,
    output logic [AVS_DATA_WIDTH-1:0]      avs_readdata,
    output logic                           avs_waitrequest
);

    // ============================================================
    // Memory map (byte addresses)
    // ============================================================
    localparam logic [31:0] CTRL_BASE = 32'h0000_0000, CTRL_SIZE = 32'h0000_0100;
    localparam logic [31:0] WBUF_BASE = 32'h0000_1000, WBUF_SIZE = 32'h0000_4000; // weight_buffer, 14-bit addr
    localparam logic [31:0] IBUF_BASE = 32'h0000_5000, IBUF_SIZE = 32'h0000_2000; // input_buffer,  13-bit addr
    localparam logic [31:0] OBUF_BASE = 32'h0000_7000, OBUF_SIZE = 32'h0000_1000; // output_buffer, 12-bit addr

    wire sel_ctrl = (avs_address >= CTRL_BASE) && (avs_address < CTRL_BASE + CTRL_SIZE);
    wire sel_wbuf = (avs_address >= WBUF_BASE) && (avs_address < WBUF_BASE + WBUF_SIZE);
    wire sel_ibuf = (avs_address >= IBUF_BASE) && (avs_address < IBUF_BASE + IBUF_SIZE);
    wire sel_obuf = (avs_address >= OBUF_BASE) && (avs_address < OBUF_BASE + OBUF_SIZE);

    // pick the enabled byte lane out of a 32-bit avalon write word
    function automatic logic [7:0] be_select_byte(input logic [31:0] wdata, input logic [3:0] be);
        logic [7:0] result;
        begin
            result = wdata[7:0];
            if (be[1]) result = wdata[15:8];
            if (be[2]) result = wdata[23:16];
            if (be[3]) result = wdata[31:24];
            be_select_byte = result;
        end
    endfunction

    // ============================================================
    // weight_buffer - write-only from Avalon
    // ============================================================
    wire        wbuf_wren      = avs_write && sel_wbuf;
    wire [13:0] wbuf_wraddress = (avs_address - WBUF_BASE);
    wire [7:0]  wbuf_wdata     = be_select_byte(avs_writedata, avs_byteenable);

    weight_buffer u_weight_buffer (
        .clock     (clk),
        .data      (wbuf_wdata),
        .rdaddress (14'b0),   // TODO: drive from compute datapath
        .rden      (1'b0),    // TODO: drive from compute datapath
        .wraddress (wbuf_wraddress),
        .wren      (wbuf_wren),
        .q         ()         // unused until compute datapath is wired up
    );

    // ============================================================
    // input_buffer - write-only from Avalon
    // ============================================================
    wire        ibuf_wren      = avs_write && sel_ibuf;
    wire [12:0] ibuf_wraddress = (avs_address - IBUF_BASE);
    wire [7:0]  ibuf_wdata     = be_select_byte(avs_writedata, avs_byteenable);

    input_buffer_8_bank u_input_buffer (
        .clock     (clk),
        .data      (ibuf_wdata),
        .rdaddress (13'b0),   // TODO: drive from compute datapath
        .rden      (1'b0),    // TODO: drive from compute datapath
        .wraddress (ibuf_wraddress),
        .wren      (ibuf_wren),
        .q         ()         // unused until compute datapath is wired up
    );

    // ============================================================
    // output_buffer - read-only from Avalon; one cycle of read latency
    // (registered rdaddress -> q), stalled with avs_waitrequest
    // ============================================================
    wire        obuf_rden      = avs_read && sel_obuf;
    wire [11:0] obuf_rdaddress = (avs_address - OBUF_BASE);
    wire [7:0]  obuf_q;

    logic obuf_rd_pending;
    always_ff @(posedge clk or negedge rst_n)
        if (!rst_n)
            obuf_rd_pending <= 1'b0;
        else if (avs_read && sel_obuf && !obuf_rd_pending)
            obuf_rd_pending <= 1'b1;
        else
            obuf_rd_pending <= 1'b0;

    output_buffer u_output_buffer (
        .clock     (clk),
        .data      (8'b0),    // TODO: drive from compute datapath (avalon side is read-only)
        .rdaddress (obuf_rdaddress),
        .rden      (obuf_rden),
        .wraddress (12'b0),   // TODO: drive from compute datapath
        .wren      (1'b0),    // TODO: drive from compute datapath
        .q         (obuf_q)
    );

    // ============================================================
    // ctrl_rf_rf - control/status register file
    // ============================================================
    wire        ctrl_valid = (avs_read || avs_write) && sel_ctrl;
    wire [31:0] ctrl_rdata;

    ctrl_rf_rf #(
        .ADDR_OFFSET (CTRL_BASE),
        .ADDR_WIDTH  (AVS_ADDR_WIDTH),
        .DATA_WIDTH  (AVS_DATA_WIDTH)
    ) u_ctrl_rf (
        .clk                     (clk),
        .resetn                  (rst_n),

        .CONTROL_start_q         (),  // TODO: drive compute datapath
        .CONTROL_input_ready_q   (),  // TODO: drive compute datapath
        .CONTROL_weight_ready_q  (),  // TODO: drive compute datapath

        .STATUS_out_ready_wdata  (1'b0),   // TODO: drive from compute datapath
        .STATUS_out_count_wdata (16'b0),   // TODO: drive from compute datapath

        .valid  (ctrl_valid),
        .read   (avs_read),
        .addr   (avs_address),
        .wdata  (avs_writedata),
        .wmask  (avs_byteenable),
        .rdata  (ctrl_rdata)
    );

    // ============================================================
    // Avalon read mux / waitrequest
    // ============================================================
    assign avs_waitrequest = avs_read && sel_obuf && !obuf_rd_pending;
    assign avs_readdata    = obuf_rd_pending ? {4{obuf_q}} : (sel_ctrl ? ctrl_rdata : 32'h0);

    // ============================================================
    // pe_array - instantiated, not yet wired to the buffer RAMs
    // ============================================================
    pe_array #(
        .DATA_WIDTH (PE_DATA_WIDTH),
        .ARRAY_ROWS (ARRAY_ROWS),
        .ARRAY_COLS (ARRAY_COLS)
    ) u_pe_array (
        .clk     (clk),
        .rst_n   (rst_n),
        .a_in    ({(ARRAY_ROWS*PE_DATA_WIDTH){1'b0}}),
        .a_en    ({ARRAY_ROWS{1'b0}}),
        .b_en    ({ARRAY_COLS{1'b0}}),
        .w_in    ({(ARRAY_ROWS*ARRAY_COLS*PE_DATA_WIDTH){1'b0}}),
        .w_en    ({(ARRAY_ROWS*ARRAY_COLS){1'b0}}),
        .w_load  (1'b0),
        .a_out   (),
        .p_out   ()
    );

endmodule

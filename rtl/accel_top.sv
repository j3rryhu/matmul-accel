// Top level: Avalon-MM slave <-> ctrl_rf_rf register file + weight/input/output
// buffer RAMs. Address decode is a flat range check per region; each region's
// local (offset-subtracted) address feeds that region's RAM/regfile directly.
//
// input_dispatch is wired up: input_buffer's read side and pe_array's
// activation-in edge are both driven by it. weight_ctrl (weight_buffer ->
// pe_array weight-load port, block-by-block) is wired up too, and drives
// input_dispatch's per-block-row preload/compute-start handshake directly
// (matmul_start is the only software trigger left; everything past that
// is sequenced by weight_ctrl). output_loader is wired up too: it drives
// pe_array's b_en, reads p_out, and read-modify-write accumulates into
// output_buffer_32_bank (32 independent banks, one per array column - see
// output_buffer_32_bank.v; the per-bank output_buffer leaf RAM itself is
// still a placeholder depth pending the real generated IP).
//
// Requires rdl/ctrl_reg.sv/ctrl_rf_rf.sv in the same compile fileset.
`timescale 1ps/1ps

module accel_top #(
    parameter AVS_ADDR_WIDTH = 32,
    parameter AVS_DATA_WIDTH = 32,   // byte-lane mux below assumes 32
    parameter PE_DATA_WIDTH  = 8,    // activation/weight operand width (int8)
    parameter ACC_DATA_WIDTH = 32,   // pe_array accumulator width - kept wider
                                      // than PE_DATA_WIDTH so a full contraction
                                      // sum of int8 x int8 products can't
                                      // overflow before software rescales it
                                      // back down under the int8 quantization
                                      // scheme (see pe.v)
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

    wire [13:0] wbuf_rdaddress;
    wire        wbuf_rden;
    wire [7:0]  wbuf_q;

    weight_buffer u_weight_buffer (
        .clock     (clk),
        .data      (wbuf_wdata),
        .rdaddress (wbuf_rdaddress),
        .rden      (wbuf_rden),
        .wraddress (wbuf_wraddress),
        .wren      (wbuf_wren),
        .q         (wbuf_q)
    );

    // ============================================================
    // input_buffer - write side from Avalon, read side driven by input_dispatch
    // ============================================================
    wire        ibuf_wren      = avs_write && sel_ibuf;
    wire [12:0] ibuf_wraddress = (avs_address - IBUF_BASE);
    wire [7:0]  ibuf_wdata     = be_select_byte(avs_writedata, avs_byteenable);

    wire [255:0] ibuf_q;
    wire [7:0]  ibuf_rdaddress;
    wire        ibuf_rden;

    input_buffer_32_bank u_input_buffer (
        .clock     (clk),
        .data      (ibuf_wdata),
        .rdaddress (ibuf_rdaddress),
        .rden      (ibuf_rden),
        .wraddress (ibuf_wraddress),
        .wren      (ibuf_wren),
        .q         (ibuf_q)
    );

    // ============================================================
    // output_buffer - banked 32-ways (one bank per array column, see
    // output_buffer_32_bank.v). output_loader drives each bank's
    // read+write directly; Avalon reads through a single muxed external
    // port, read-only, one cycle of read latency, stalled with
    // avs_waitrequest same as before. Entries are int8 (PE_DATA_WIDTH) -
    // output_loader rescales pe_array's wider ACC_DATA_WIDTH accumulator
    // down to int8 before it ever reaches output_buffer (see
    // output_loader.v).
    // ============================================================
    localparam OBUF_BANK_ADDR_WIDTH = 7;   // 4096-byte OBUF_SIZE / 32 banks = 128 entries/bank

    wire        obuf_ext_rden      = avs_read && sel_obuf;
    wire [11:0] obuf_ext_rdaddress = (avs_address - OBUF_BASE);
    wire [7:0]  obuf_ext_q;

    logic obuf_rd_pending;
    always_ff @(posedge clk or negedge rst_n)
        if (!rst_n)
            obuf_rd_pending <= 1'b0;
        else if (avs_read && sel_obuf && !obuf_rd_pending)
            obuf_rd_pending <= 1'b1;
        else
            obuf_rd_pending <= 1'b0;

    wire        output_loader_busy;
    wire [31:0]                             obuf_bank_rden;
    wire [32*OBUF_BANK_ADDR_WIDTH-1:0]      obuf_bank_rdaddress;
    wire [32*PE_DATA_WIDTH-1:0]             obuf_bank_q;
    wire [31:0]                             obuf_bank_wren;
    wire [32*OBUF_BANK_ADDR_WIDTH-1:0]      obuf_bank_waddr;
    wire [32*PE_DATA_WIDTH-1:0]             obuf_bank_wdata;

    output_buffer_32_bank #(
        .DATA_WIDTH      (PE_DATA_WIDTH),
        .NUM_BANKS       (32),
        .BANK_ADDR_WIDTH (OBUF_BANK_ADDR_WIDTH)
    ) u_output_buffer (
        .clock          (clk),

        .busy           (output_loader_busy),
        .bank_rden      (obuf_bank_rden),
        .bank_rdaddress (obuf_bank_rdaddress),
        .bank_q         (obuf_bank_q),
        .bank_wren      (obuf_bank_wren),
        .bank_waddr     (obuf_bank_waddr),
        .bank_wdata     (obuf_bank_wdata),

        .ext_rden       (obuf_ext_rden),
        .ext_rdaddress  (obuf_ext_rdaddress),
        .ext_q          (obuf_ext_q)
    );

    // ============================================================
    // ctrl_rf_rf - control/status register file
    // ============================================================
    wire        ctrl_valid = (avs_read || avs_write) && sel_ctrl;
    wire [31:0] ctrl_rdata;

    wire        matmul_start;          // pulse: latch WEIGHT_ROWS/WEIGHT_COLS, reset block position
    wire [15:0] weight_rows;           // K: weight rows (contraction dimension)
    wire [15:0] weight_cols;           // N: weight cols
    wire [9:0]  input_max_addr;        // max input_buffer read address
    wire [15:0] input_cols;            // number of columns of the input (activation) matrix
    wire [15:0] output_scale;          // Q0.16 rescale factor (value/65536, range [0,1)) for pe_array's ACC_WIDTH accumulator
    wire        weight_ctrl_busy;      // weight_ctrl/weight_loader busy, for STATUS.busy
    wire        weight_ctrl_done;      // level: last block committed and its input band fully streamed in

    // STATUS.busy: weight/input side sequencing OR output_loader still
    // draining the last block's psums (weight_ctrl_done can go high a few
    // pipeline cycles before output_loader finishes writing that block's
    // results - see output_loader.v's header note on drain latency).
    wire        matmul_busy = weight_ctrl_busy || output_loader_busy;

    // STATUS.done: weight_ctrl_done only means every block has been
    // *committed* (weight_ctrl.v's WC_DONE) - output_loader.done pulses
    // per committed block, not once for the whole matmul (see
    // output_loader.v), so the real "whole matmul done" level is
    // weight_ctrl_done still held true once output_loader has also gone
    // idle (i.e. the last block's drain has finished). Both sides are
    // levels, so this stays high until the next matmul_start drops
    // weight_ctrl_done again - safe for software to poll at any time.
    wire        matmul_done = weight_ctrl_done && !output_loader_busy;

    ctrl_rf_rf #(
        .ADDR_OFFSET (CTRL_BASE),
        .ADDR_WIDTH  (AVS_ADDR_WIDTH),
        .DATA_WIDTH  (AVS_DATA_WIDTH)
    ) u_ctrl_rf (
        .clk                     (clk),
        .resetn                  (rst_n),

        .CONTROL_matmul_start_q        (matmul_start),

        .WEIGHT_ROWS_value_q     (weight_rows),
        .WEIGHT_COLS_value_q     (weight_cols),
        .INPUT_MAX_ADDR_value_q  (input_max_addr),
        .INPUT_COLS_value_q      (input_cols),  // TODO: drive compute datapath (not yet consumed by input_dispatch)
        .OUTPUT_SCALE_value_q    (output_scale),  // Q0.16 rescale factor, consumed by output_loader's requantize stage

        .STATUS_busy_wdata              (matmul_busy),
        .STATUS_done_wdata              (matmul_done),

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
    assign avs_readdata    = obuf_rd_pending ? {4{obuf_ext_q}} : (sel_ctrl ? ctrl_rdata : 32'h0);

    // ============================================================
    // input_dispatch - streams input_buffer into the array's left edge,
    // one 32-row band at a time, driven by weight_ctrl below
    // ============================================================
    wire [ARRAY_ROWS*PE_DATA_WIDTH-1:0] a_out;
    wire [ARRAY_ROWS-1:0]               a_en;

    wire [7:0] input_band_base_addr;  // weight_ctrl -> input_dispatch: current block's per-bank base offset
    wire       input_band_start;      // weight_ctrl -> input_dispatch: pulse, begin preloading that band
    wire       input_compute_start;   // weight_ctrl -> input_dispatch: pulse, begin streaming into pe_array
    wire       input_fifos_primed;    // input_dispatch -> weight_ctrl: all row FIFOs hold >=1 element
    wire       input_band_done;       // input_dispatch -> weight_ctrl: current band fully drained

    input_dispatch #(
        .ARRAY_ROWS (ARRAY_ROWS),
        .DATA_WIDTH (PE_DATA_WIDTH)
    ) u_input_dispatch (
        .clock          (clk),
        .rst_n          (rst_n),

        .ibuf_q         (ibuf_q),
        .ibuf_rdaddress (ibuf_rdaddress),
        .ibuf_rden      (ibuf_rden),

        .a_out          (a_out),
        .a_en           (a_en),

        .i_max_addr       (input_max_addr[7:0]),
        .i_band_base_addr (input_band_base_addr),
        .i_band_start     (input_band_start),
        .i_start_compute  (input_compute_start),
        .fifos_primed     (input_fifos_primed),
        .band_done        (input_band_done)
    );

    // ============================================================
    // weight_ctrl - walks the weight matrix column-major (contraction-
    // dimension blocks fast, output-dimension blocks slow) as 32x32
    // blocks, prefetching each into pe_array and driving input_dispatch's
    // per-block handshake; matmul_start is the only software trigger,
    // everything else is sequenced internally
    // ============================================================
    wire [$clog2(ARRAY_ROWS*ARRAY_COLS)-1:0] w_addr;
    wire [PE_DATA_WIDTH-1:0]                 w_data;
    wire                                      w_en;
    wire                                      w_load;
    wire [15:0]                              committed_n_blk_idx;    // -> output_loader
    wire                                      committed_first_k_blk; // -> output_loader

    weight_ctrl #(
        .DATA_WIDTH      (PE_DATA_WIDTH),
        .ARRAY_ROWS      (ARRAY_ROWS),
        .ARRAY_COLS      (ARRAY_COLS),
        .WBUF_ADDR_WIDTH (14),
        .IBUF_ADDR_WIDTH (8),
        .DIM_WIDTH       (16)
    ) u_weight_ctrl (
        .clk               (clk),
        .rst_n             (rst_n),

        .matmul_start      (matmul_start),
        .weight_rows       (weight_rows),
        .weight_cols       (weight_cols),

        .busy              (weight_ctrl_busy),
        .done              (weight_ctrl_done),

        .wbuf_rden         (wbuf_rden),
        .wbuf_rdaddress    (wbuf_rdaddress),
        .wbuf_q            (wbuf_q),

        .w_addr            (w_addr),
        .w_data            (w_data),
        .w_en              (w_en),
        .w_load            (w_load),

        .input_max_addr       (input_max_addr[7:0]),
        .input_band_start     (input_band_start),
        .input_band_base_addr (input_band_base_addr),
        .input_compute_start  (input_compute_start),
        .input_fifos_primed   (input_fifos_primed),
        .input_band_done      (input_band_done),

        .committed_n_blk_idx   (committed_n_blk_idx),
        .committed_first_k_blk (committed_first_k_blk)
    );

    // ============================================================
    // output_loader - drives pe_array's b_en, reads p_out, and read-
    // modify-write accumulates into output_buffer_32_bank
    // ============================================================
    wire [ARRAY_COLS-1:0]                pe_b_en;
    wire [ARRAY_COLS*ACC_DATA_WIDTH-1:0] pe_p_out;
    wire                                 output_loader_done;

    output_loader #(
        .DATA_WIDTH     (PE_DATA_WIDTH),
        .ACC_WIDTH      (ACC_DATA_WIDTH),
        .ARRAY_COLS     (ARRAY_COLS),
        .ROW_ADDR_WIDTH (OBUF_BANK_ADDR_WIDTH),
        .DIM_WIDTH       (16)
    ) u_output_loader (
        .clock (clk),
        .rst_n (rst_n),

        .p_in  (pe_p_out),
        .b_en  (pe_b_en),

        .a_en_last             (a_en[ARRAY_COLS-1]),
        .total_rows            (input_cols[OBUF_BANK_ADDR_WIDTH-1:0]),
        .committed_n_blk_idx   (committed_n_blk_idx),
        .committed_first_k_blk (committed_first_k_blk),

        .output_scale   (output_scale),

        .obuf_rden      (obuf_bank_rden),
        .obuf_rdaddress (obuf_bank_rdaddress),
        .obuf_q         (obuf_bank_q),
        .obuf_wren      (obuf_bank_wren),
        .obuf_waddr     (obuf_bank_waddr),
        .obuf_wdata     (obuf_bank_wdata),

        .busy (output_loader_busy),
        .done (output_loader_done)
    );

    // ============================================================
    // pe_array - activation side driven by input_dispatch, weight side
    // driven by weight_ctrl, partial-sum-out side driven by output_loader
    // ============================================================
    pe_array #(
        .DATA_WIDTH (PE_DATA_WIDTH),
        .ACC_WIDTH  (ACC_DATA_WIDTH),
        .ARRAY_ROWS (ARRAY_ROWS),
        .ARRAY_COLS (ARRAY_COLS)
    ) u_pe_array (
        .clk     (clk),
        .rst_n   (rst_n),
        .a_in    (a_out),
        .a_en    (a_en),
        .b_en    (pe_b_en),
        .w_addr  (w_addr),
        .w_data  (w_data),
        .w_en    (w_en),
        .w_load  (w_load),
        .a_out   (),
        .p_out   (pe_p_out)
    );

endmodule

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
//
// Avalon-MM slave notes for Quartus IP wrapping: the reset port is named
// reset_n (not rst_n) so Platform Designer's signal-role auto-detection
// associates it with this slave's clock interface, and the read side
// carries avs_burstcount/avs_readdatavalid so it's recognized as
// burst-capable with a readdatavalid-qualified variable read latency (see
// the "Avalon-MM read burst engine" section below). Writes are always
// single-beat - only reads (reading results back out of output_buffer) are
// ever bursted in practice.
`timescale 1ps/1ps

module accel_top #(
    parameter AVS_ADDR_WIDTH       = 32,
    parameter AVS_DATA_WIDTH       = 32,   // byte-lane mux below assumes 32
    parameter AVS_BURSTCOUNT_WIDTH = 8,    // max read burst length = 2**AVS_BURSTCOUNT_WIDTH-1 beats (direct encoding: avs_burstcount's value IS the beat count)
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
    input                                  reset_n,    // active-low

    // Avalon-MM slave
    input        [AVS_ADDR_WIDTH-1:0]      avs_address,
    input                                  avs_read,
    input                                  avs_write,
    input        [AVS_DATA_WIDTH-1:0]      avs_writedata,
    input        [AVS_DATA_WIDTH/8-1:0]    avs_byteenable,
    input        [AVS_BURSTCOUNT_WIDTH-1:0] avs_burstcount,
    output logic [AVS_DATA_WIDTH-1:0]      avs_readdata,
    output logic                           avs_readdatavalid,
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
    wire [ARRAY_ROWS*8-1:0]  ibuf_rdaddress;
    wire        ibuf_rden;
    wire [ARRAY_ROWS-1:0] ibuf_byteenable;

    input_buffer_32_bank u_input_buffer (
        .clock         (clk),
        .data          (ibuf_wdata),
        .rdaddress     (ibuf_rdaddress),
        .rden          (ibuf_rden),
        .rd_byteenable (ibuf_byteenable),  // per-row masking, driven by input_dispatch (see input_dispatch.v)
        .wraddress     (ibuf_wraddress),
        .wren          (ibuf_wren),
        .q             (ibuf_q)
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

    // rd_issue/rd_issue_addr/rd_issue_sel_obuf are driven by the read burst
    // engine below (declared later in the file - fine for continuous
    // assignments in SystemVerilog).
    wire        obuf_ext_rden      = rd_issue && rd_issue_sel_obuf;
    wire [11:0] obuf_ext_rdaddress = rd_issue_addr - OBUF_BASE;
    wire [7:0]  obuf_ext_q;

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
    // ctrl_write/ctrl_read_issue are mutually exclusive (avs_write and
    // avs_read are never both asserted), so ctrl_addr can safely pick
    // between the raw Avalon address (writes, always single-beat) and the
    // read burst engine's current beat address (rd_issue_addr, declared
    // further down) without needing separate read/write address ports.
    wire        ctrl_write      = avs_write && sel_ctrl;
    wire        ctrl_read_issue = rd_issue && rd_issue_sel_ctrl;
    wire        ctrl_valid      = ctrl_write || ctrl_read_issue;
    wire [31:0] ctrl_addr       = ctrl_write ? avs_address : rd_issue_addr;
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
        .resetn                  (reset_n),

        .CONTROL_matmul_start_q        (matmul_start),

        .WEIGHT_ROWS_value_q     (weight_rows),
        .WEIGHT_COLS_value_q     (weight_cols),
        .INPUT_MAX_ADDR_value_q  (input_max_addr),
        .INPUT_COLS_value_q      (input_cols),  // TODO: drive compute datapath (not yet consumed by input_dispatch)
        .OUTPUT_SCALE_value_q    (output_scale),  // Q0.16 rescale factor, consumed by output_loader's requantize stage

        .STATUS_busy_wdata              (matmul_busy),
        .STATUS_done_wdata              (matmul_done),

        .valid  (ctrl_valid),
        .read   (ctrl_read_issue),
        .addr   (ctrl_addr),
        .wdata  (avs_writedata),
        .wmask  (avs_byteenable),
        .rdata  (ctrl_rdata)
    );

    // ============================================================
    // Avalon-MM read burst engine
    // ============================================================
    // Read-only burst support - writes stay single-beat, always accepted
    // immediately whenever no read burst is draining (matches this
    // design's actual traffic: software writes weight_buffer/input_buffer
    // byte-by-byte, then bursts results back out of output_buffer).
    //
    // Non-pipelined: at most one read command is in flight at a time -
    // avs_waitrequest stays asserted for the whole command (address-issue
    // *and* data-drain phases), so the next command can't be accepted
    // until the current one's last beat has been returned. Simple and
    // Avalon-spec-legal. avs_readdatavalid and avs_waitrequest are both
    // derived from burst_busy (one register apart - see rd_valid_q below),
    // so for every beat but the last, readdatavalid pulses while
    // waitrequest is still asserted (mid-burst); the *last* beat's
    // readdatavalid happens to land the same cycle waitrequest finally
    // drops - track readdatavalid independently rather than assuming any
    // fixed relationship to waitrequest.
    //
    // ctrl_rf_rf's rdata is combinational (0-latency); output_buffer's leaf
    // RAM has a fixed 1-cycle registered read latency (see
    // tb/models/buffer_ram_models.v / the real generated IP). ctrl_rdata_q
    // below adds one register stage to ctrl's path so both regions present
    // their data exactly one cycle after their address is issued, letting
    // a single burst pipeline serve either region uniformly.
    //
    // rd_issue/rd_issue_addr deliberately never read avs_address/avs_read
    // directly - cur_addr is latched from avs_address once, at accept, and
    // every beat (including the first) is issued from that registered
    // snapshot on the cycle *after* accept. This costs one extra cycle of
    // latency on the first beat, in exchange for rd_issue/rd_issue_addr
    // depending only on registered state (never on a live input that a
    // master is simultaneously free to change the moment it sees
    // avs_waitrequest deasserted for the *previous* command) - simpler to
    // reason about and safe regardless of exactly when in a cycle a master
    // updates avs_address after that.
    logic                            burst_busy;
    logic [AVS_ADDR_WIDTH-1:0]       cur_addr;        // latched beat address, valid while burst_busy
    logic [AVS_BURSTCOUNT_WIDTH-1:0] beats_remaining; // beats left to issue (including this cycle's), valid while burst_busy

    // rd_issue: a beat's (registered) address is being presented to the
    // RAMs/regfile this cycle. Consumed above by obuf_ext_rden/
    // obuf_ext_rdaddress and ctrl_read_issue/ctrl_addr.
    wire                      rd_issue          = burst_busy;
    wire [AVS_ADDR_WIDTH-1:0] rd_issue_addr     = cur_addr;
    wire                      rd_issue_sel_ctrl = (rd_issue_addr >= CTRL_BASE) && (rd_issue_addr < CTRL_BASE + CTRL_SIZE);
    wire                      rd_issue_sel_obuf = (rd_issue_addr >= OBUF_BASE) && (rd_issue_addr < OBUF_BASE + OBUF_SIZE);

    // data-phase pipeline: one cycle after rd_issue, that beat's data is valid
    logic                       rd_valid_q;
    logic [AVS_ADDR_WIDTH-1:0]  rd_addr_q;
    logic [AVS_DATA_WIDTH-1:0]  ctrl_rdata_q;

    always_ff @(posedge clk or negedge reset_n)
        if (!reset_n) begin
            burst_busy      <= 1'b0;
            cur_addr        <= '0;
            beats_remaining <= '0;
            rd_valid_q       <= 1'b0;
            rd_addr_q         <= '0;
            ctrl_rdata_q      <= '0;
        end else begin
            if (!burst_busy) begin
                if (avs_read) begin
                    burst_busy      <= 1'b1;
                    cur_addr        <= avs_address;
                    // computed inline, not via a separate "burstcount_eff"
                    // wire: icarus has a real quirk where a value change on
                    // a primary input (avs_burstcount) reaches a flip-flop
                    // that reads it *directly* in time for the very next
                    // edge, but lags an extra cycle if it's read through an
                    // intermediate continuous-assign wire first (confirmed
                    // via an isolated probe outside this design) - avoid
                    // that class of hazard entirely by never routing a
                    // primary input through a wire before a register uses
                    // it.
                    beats_remaining <= (avs_burstcount == AVS_BURSTCOUNT_WIDTH'(0)) ? AVS_BURSTCOUNT_WIDTH'(1) : avs_burstcount;
                end
            end else if (beats_remaining == AVS_BURSTCOUNT_WIDTH'(1)) begin
                burst_busy <= 1'b0;   // this cycle issues the last beat
            end else begin
                // +1, not a word stride: every region on this port
                // (weight_buffer/input_buffer/output_buffer, and hence the
                // only regions ever actually burst-read) is byte-addressed
                // - one avs_address per int8 entry, replicated across all
                // 4 avs_readdata lanes (see the {4{obuf_ext_q}} mux below) -
                // not word-addressed like ctrl_rf_rf. A word stride here
                // would skip 3 of every 4 bytes.
                cur_addr        <= cur_addr + AVS_ADDR_WIDTH'(1);
                beats_remaining <= beats_remaining - AVS_BURSTCOUNT_WIDTH'(1);
            end

            rd_valid_q   <= rd_issue;
            rd_addr_q    <= rd_issue_addr;
            ctrl_rdata_q <= ctrl_rdata;
        end

    wire rd_addr_q_sel_ctrl = (rd_addr_q >= CTRL_BASE) && (rd_addr_q < CTRL_BASE + CTRL_SIZE);
    wire rd_addr_q_sel_obuf = (rd_addr_q >= OBUF_BASE) && (rd_addr_q < OBUF_BASE + OBUF_SIZE);

    assign avs_waitrequest   = burst_busy;
    assign avs_readdatavalid = rd_valid_q;
    assign avs_readdata      = rd_addr_q_sel_ctrl ? ctrl_rdata_q :
                                rd_addr_q_sel_obuf ? {4{obuf_ext_q}} :
                                                      32'h0;

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
        .rst_n          (reset_n),

        .ibuf_q         (ibuf_q),
        .ibuf_rdaddress (ibuf_rdaddress),
        .ibuf_rden      (ibuf_rden),
        .ibuf_byteenable(ibuf_byteenable),

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
    wire [15:0]                              committed_k_blk_idx;
    wire                                      committed_first_k_blk; // -> output_loader
    wire [$clog2(ARRAY_ROWS+1):0]           valid_row;
    wire [$clog2(ARRAY_COLS+1):0]           valid_col;

    weight_ctrl #(
        .DATA_WIDTH      (PE_DATA_WIDTH),
        .ARRAY_ROWS      (ARRAY_ROWS),
        .ARRAY_COLS      (ARRAY_COLS),
        .WBUF_ADDR_WIDTH (14),
        .IBUF_ADDR_WIDTH (8),
        .DIM_WIDTH       (16)
    ) u_weight_ctrl (
        .clk               (clk),
        .rst_n             (reset_n),

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
        .committed_k_blk_idx   (committed_k_blk_idx),
        .committed_first_k_blk (committed_first_k_blk),

        .valid_row          (valid_row),
        .valid_col          (valid_col)
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
        .ARRAY_ROWS     (ARRAY_ROWS),
        .ROW_ADDR_WIDTH (OBUF_BANK_ADDR_WIDTH),
        .DIM_WIDTH       (16)
    ) u_output_loader (
        .clock (clk),
        .rst_n (reset_n),

        .p_in  (pe_p_out),
        .b_en  (pe_b_en),

        .a_en_last             (a_en[ARRAY_COLS-1]),
        .total_rows            (input_cols[OBUF_BANK_ADDR_WIDTH-1:0]),
        .committed_k_blk_idx   (committed_k_blk_idx),
        .committed_n_blk_idx   (committed_n_blk_idx),
        .committed_first_k_blk (committed_first_k_blk),

        .output_scale   (output_scale),

        .weight_rows    (valid_row),
        .weight_cols    (valid_col),

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
        .rst_n   (reset_n),
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

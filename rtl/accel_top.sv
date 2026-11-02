// Top level: Avalon-MM slave <-> ctrl_rf_rf register file + weight/input/output
// buffer RAMs. Address decode is a flat range check per region; each region's
// local (offset-subtracted) address feeds that region's RAM/regfile directly.
//
// input_dispatch is wired up: input_buffer's read side and pe_array's
// activation-in edge are both driven by it. weight_ctrl (weight_buffer_32_bank
// -> pe_array's per-row weight-load ports, block-by-block) is wired up too, and drives
// input_dispatch's per-block-row preload/compute-start handshake directly
// (matmul_start is the only software trigger left; everything past that
// is sequenced by weight_ctrl). output_loader is wired up too: it drives
// pe_array's b_en, reads p_out, and read-modify-write accumulates into
// output_buffer_32_bank (ARRAY_COLS independent banks, one per array column - see
// output_buffer_32_bank.v; the per-bank output_buffer leaf RAM itself is
// still a placeholder depth pending the real generated IP).
//
// NB: the datapath is square-only - pe_array's a_in/a_en lanes are
// indexed by column while its p_out/b_en lanes are indexed by row, and
// output_loader banks per p_out lane while naming that axis ARRAY_COLS.
// Every width lines up only while ARRAY_ROWS == ARRAY_COLS. Resizing the
// array means changing both together.
//
// Requires rdl/ctrl_reg.sv/ctrl_rf_rf.sv in the same compile fileset.
`timescale 1ps/1ps

module accel_top #(
    parameter AVS_ADDR_WIDTH = 32,
    parameter AVS_DATA_WIDTH = 128,   // byte-lane mux below assumes 32
    parameter PE_DATA_WIDTH  = 8,    // activation/weight operand width (int8)
    parameter ACC_DATA_WIDTH = 32,   // pe_array accumulator width - kept wider
                                      // than PE_DATA_WIDTH so a full contraction
                                      // sum of int8 x int8 products can't
                                      // overflow before software rescales it
                                      // back down under the int8 quantization
                                      // scheme (see pe.v)
    parameter ARRAY_ROWS     = 16,  // must equal ARRAY_COLS - the array is
    parameter ARRAY_COLS     = 16   // square-only, see header note
)(
    input                                  clk,
    input                                  reset_n,      // active-low

    // Avalon-MM slave
    input        [AVS_ADDR_WIDTH-1:0]      avs_address,
    input                                  avs_read,
    input                                  avs_write,
    input        [AVS_DATA_WIDTH-1:0]      avs_writedata,
    input        [AVS_DATA_WIDTH/8-1:0]    avs_byteenable,
    input        [10:0]                    avs_burstcount,
    output logic [AVS_DATA_WIDTH-1:0]      avs_readdata,
    output logic                           avs_readdatavalid,
    output logic                           avs_waitrequest
);

    // ============================================================
    // Memory map (byte addresses)
    // ============================================================
    localparam logic [31:0] CTRL_BASE = 32'h0000_0000, CTRL_SIZE = 32'h0000_0100;
    localparam logic [31:0] WBUF_BASE = 32'h0000_1000, WBUF_SIZE = 32'h0000_4000; // weight_buffer (banked), 14-bit addr
    localparam logic [31:0] IBUF_BASE = 32'h0000_5000, IBUF_SIZE = 32'h0000_4000; // input_buffer (banked),  14-bit addr
    localparam logic [31:0] OBUF_BASE = 32'h0000_9000, OBUF_SIZE = 32'h0000_4000; // output_buffer (banked), 14-bit addr

    // ---- buffer geometry ----
    // Per-bank depths are fixed by the generated RAM IP; the bank *count*
    // follows the array, so the flat (software-visible) address width of
    // each banked buffer scales with ARRAY_ROWS/ARRAY_COLS. All three
    // buffers use 1KB banks, which at 16 banks fill each 16KB window
    // exactly - enough for a 128x128 W times a 128x128 X (8 output-row
    // blocks x 128 columns per output bank, 8 contraction bands x 128
    // columns per input bank, 8 rows of 128 weights per weight bank).
    // Growing a depth or the array means growing the matching *_SIZE
    // window above (the check below catches a window that is too small).
    localparam int WBUF_BANK_ADDR_WIDTH = 10;  // weight_buffer IP depth = 1024 entries/bank
    localparam int IBUF_BANK_ADDR_WIDTH = 10;  // input_buffer IP depth  = 1024 entries/bank
    localparam int OBUF_BANK_ADDR_WIDTH = 10;  // output_buffer IP depth = 1024 entries/bank
    localparam int WBUF_ADDR_WIDTH      = $clog2(ARRAY_ROWS) + WBUF_BANK_ADDR_WIDTH;
    localparam int IBUF_ADDR_WIDTH      = $clog2(ARRAY_ROWS) + IBUF_BANK_ADDR_WIDTH;
    localparam int OBUF_ADDR_WIDTH      = $clog2(ARRAY_COLS) + OBUF_BANK_ADDR_WIDTH;

    // ---- output canvas ----
    // output_buffer holds one int8 canvas OBUF_CANVAS_W bytes wide and
    // OBUF_CANVAS_H rows tall - 128 x 128, the whole 16KB buffer. Canvas row
    // R lives in bank R % ARRAY_ROWS at byte offset
    // (R / ARRAY_ROWS)*OBUF_CANVAS_W + column, and the host reads the canvas
    // as one row-major image, row R at OBUF_BASE + R*OBUF_CANVAS_W. Each
    // matmul writes its K x M result as one tile of the canvas, placed by
    // TILE_CFG (see the tile latch next to ctrl_rf below). A bigger canvas
    // needs a deeper output_buffer IP (OBUF_BANK_ADDR_WIDTH), a bigger OBUF
    // window, and OBUF_CANVAS_W to match.
    localparam int OBUF_CANVAS_W = 128;   // bytes per canvas row - power of two, >= 2 host words
    localparam int OBUF_CANVAS_H = (ARRAY_ROWS << OBUF_BANK_ADDR_WIDTH) / OBUF_CANVAS_W;

    // Each window must cover every bank, or the top banks are unreachable
    // from the host. Elaboration-time check; fails the build, not the sim.
    generate
        if ((OBUF_CANVAS_W & (OBUF_CANVAS_W - 1)) != 0
            || OBUF_CANVAS_W < 2 * (AVS_DATA_WIDTH / PE_DATA_WIDTH)
            || OBUF_CANVAS_W > (1 << OBUF_BANK_ADDR_WIDTH)) begin : g_bad_canvas_width
            OBUF_CANVAS_W_must_be_a_power_of_two_from_2_words_to_one_bank u_err();
        end
        if ((1 << WBUF_ADDR_WIDTH) > WBUF_SIZE) begin : g_wbuf_window_too_small
            WBUF_SIZE_smaller_than_weight_buffer u_err();
        end
        if ((1 << IBUF_ADDR_WIDTH) > IBUF_SIZE) begin : g_ibuf_window_too_small
            IBUF_SIZE_smaller_than_input_buffer u_err();
        end
        if ((1 << OBUF_ADDR_WIDTH) > OBUF_SIZE) begin : g_obuf_window_too_small
            OBUF_SIZE_smaller_than_output_buffer u_err();
        end
    endgenerate

    // The buffers' Avalon-facing ports are AVS_DATA_WIDTH wide, so a host
    // access moves BYTES_PER_WORD bytes and the byte address advances by
    // BYTES_PER_WORD. Those ports are addressed in words; the core-facing
    // ports stay byte-addressed (weight_loader/input_dispatch read one byte
    // per cycle, output_loader writes one byte per cycle).
    localparam int BYTES_PER_WORD  = AVS_DATA_WIDTH / PE_DATA_WIDTH;
    localparam int WORD_SHIFT      = $clog2(BYTES_PER_WORD);
    localparam int WBUF_WADDR_WIDTH      = WBUF_ADDR_WIDTH      - WORD_SHIFT;
    localparam int IBUF_BANK_WADDR_WIDTH = IBUF_BANK_ADDR_WIDTH - WORD_SHIFT;
    localparam int OBUF_BANK_RADDR_WIDTH = OBUF_BANK_ADDR_WIDTH - WORD_SHIFT;
    localparam int IBUF_WADDR_WIDTH      = IBUF_ADDR_WIDTH      - WORD_SHIFT;
    localparam int OBUF_RADDR_WIDTH      = OBUF_ADDR_WIDTH      - WORD_SHIFT;

    wire sel_ctrl = (avs_address >= CTRL_BASE) && (avs_address < CTRL_BASE + CTRL_SIZE);
    wire sel_wbuf = (avs_address >= WBUF_BASE) && (avs_address < WBUF_BASE + WBUF_SIZE);
    wire sel_ibuf = (avs_address >= IBUF_BASE) && (avs_address < IBUF_BASE + IBUF_SIZE);
    wire sel_obuf = (avs_address >= OBUF_BASE) && (avs_address < OBUF_BASE + OBUF_SIZE);
    logic accepted;
    logic burst_busy;
    logic sel_ctrl_r;
    logic sel_obuf_r;
    logic rd_burst_on;

    logic [10:0] burst_cnt_buf;
    logic [AVS_ADDR_WIDTH-1:0] addr_buf;
    wire  read_valid;
    logic obuf_data_valid;   // an OBUF beat is in obuf_ext_q this cycle (output_buffer section)
    assign read_valid = rf_data_valid || obuf_data_valid;
    // One read in flight at a time (maximumPendingReadTransactions = 1): a
    // new read stalls a cycle before it is accepted, and burst_busy holds
    // off every command for the rest of a multi-beat burst.
    assign avs_waitrequest = (avs_read && !accepted) || burst_busy;

    // An Avalon master holds write/address/writedata stable until
    // avs_waitrequest drops, so avs_write on its own means "a write is
    // being presented", not "a write was accepted" - using it directly as
    // a write enable commits the access once per stalled cycle. That is
    // invisible for the buffers (the same word is rewritten to the same
    // address) but not for a pulse field: a CONTROL.matmul_start write
    // stalled behind an in-flight burst read fires the start strobe once
    // for every cycle the master held it. Gate every write enable on the
    // accepting cycle instead. No combinational loop here - avs_waitrequest
    // depends only on avs_read/accepted/burst_busy, never on the write
    // enables.
    wire avs_write_accepted = avs_write && !avs_waitrequest;

    always@(posedge clk)begin
        if(!reset_n)begin
            burst_cnt_buf <= 0;
            accepted <= 0;
            burst_busy <= 0;
            addr_buf <= 0;
            sel_ctrl_r <= 0;
            sel_obuf_r <= 0;
            rd_burst_on <= 0;
            avs_readdatavalid <= 0;
            avs_readdata <= 0;
        end
        else begin
            // Avalon forbids readdatavalid in the cycle a read is accepted
            // (at least one cycle of latency), so every beat leaves through
            // these registers one cycle after read_valid/read_data. They
            // update every cycle, not only while rd_burst_on is set -
            // otherwise the last beat's valid stays high after the burst.
            avs_readdatavalid <= read_valid;
            avs_readdata      <= read_data;

            if(avs_read && !rd_burst_on)begin
                addr_buf <= avs_address + BYTES_PER_WORD;
                accepted <= 1;
                burst_cnt_buf <= avs_burstcount == 0 ? 1 : avs_burstcount;
                sel_ctrl_r <= sel_ctrl;
                sel_obuf_r <= sel_obuf;
                rd_burst_on <= 1;
            end

            if(accepted)begin
                burst_busy <= 1;
            end

            if(rd_burst_on)begin
                burst_cnt_buf <= burst_cnt_buf - 1;
                addr_buf <= addr_buf + BYTES_PER_WORD;
                if(burst_cnt_buf == 1)begin
                    sel_ctrl_r <= 0;
                    sel_obuf_r <= 0;
                    rd_burst_on <= 0;
                    accepted <= 0;
                    burst_busy <= 0;
                end
            end
        end
    end

    // ============================================================
    // Host-side address remap for the two banked write-only buffers.
    // Software writes each matrix as ONE contiguous row-major image, every
    // row zero-padded up to a whole number of AVS_DATA_WIDTH words:
    //   WBUF: W (K x N), row pitch = roundup(WEIGHT_COLS,    BYTES_PER_WORD)
    //   IBUF: X (N x M), row pitch = roundup(INPUT_MAX_ADDR, BYTES_PER_WORD)
    // Every host word then belongs to one row, so it lands whole in one
    // bank at a word-aligned offset: one RAM write per host write, no
    // stall. buf_addr_remap scatters row r to bank r % ARRAY_ROWS using
    // those live register values, so WEIGHT_COLS / INPUT_MAX_ADDR must be
    // programmed BEFORE the buffers are written. Inside a bank, rows keep
    // the padded pitch (weight_ctrl's block/band bases use it too). Writes
    // land two cycles after they are accepted (see buf_addr_remap.v).
    // output_buffer is read back as a fixed-pitch canvas instead - see its
    // section below.
    // ============================================================
    logic [15:0] weight_cols;           // N: weight cols (from ctrl_rf, below)
    logic [9:0]  input_max_addr;        // M: per-band input reads (from ctrl_rf, below)
    logic [15:0] input_cols;            // M: output columns per matmul (from ctrl_rf, below) - the tile width

    localparam int WBUF_ROWW_WIDTH = WBUF_BANK_ADDR_WIDTH - WORD_SHIFT + 1;  // row pitch up to a whole bank
    localparam int IBUF_ROWW_WIDTH = IBUF_BANK_ADDR_WIDTH - WORD_SHIFT + 1;

    wire [16:0] wbuf_pitch_words = ({1'b0, weight_cols} + (BYTES_PER_WORD-1)) >> WORD_SHIFT;
    wire [IBUF_BANK_ADDR_WIDTH:0] ibuf_pitch_words =
        ({1'b0, input_max_addr[IBUF_BANK_ADDR_WIDTH-1:0]} + (BYTES_PER_WORD-1)) >> WORD_SHIFT;

    // ============================================================
    // weight_buffer - banked ARRAY_ROWS-ways (one bank per PE row, see
    // weight_buffer_32_bank.v). Write-only from Avalon through the remap;
    // the read side is driven by weight_ctrl, every bank in parallel
    // ============================================================
    wire [WBUF_WADDR_WIDTH-1:0] wbuf_host_waddr = (avs_address - WBUF_BASE) >> WORD_SHIFT;
    wire [WBUF_WADDR_WIDTH-1:0] wbuf_wraddress;
    wire [AVS_DATA_WIDTH-1:0]   wbuf_wdata;
    wire                        wbuf_wren;

    buf_addr_remap #(
        .DATA_WIDTH       (AVS_DATA_WIDTH),
        .HOST_WADDR_WIDTH (WBUF_WADDR_WIDTH),
        .ROWW_WIDTH       (WBUF_ROWW_WIDTH),
        .BANK_SEL_WIDTH   ($clog2(ARRAY_ROWS)),
        .BANK_WADDR_WIDTH (WBUF_BANK_ADDR_WIDTH - WORD_SHIFT)
    ) u_wbuf_remap (
        .clk        (clk),
        .host_waddr (wbuf_host_waddr),
        .host_wdata (avs_writedata),
        .host_wren  (avs_write_accepted && sel_wbuf),
        .row_words  (wbuf_pitch_words[WBUF_ROWW_WIDTH-1:0]),
        .bank_waddr (wbuf_wraddress),
        .bank_wdata (wbuf_wdata),
        .bank_wren  (wbuf_wren)
    );

    logic [ARRAY_ROWS*WBUF_BANK_ADDR_WIDTH-1:0] wbuf_rdaddress;
    logic                                       wbuf_rden;
    logic [ARRAY_ROWS-1:0]                      wbuf_byteenable;
    logic [ARRAY_ROWS*PE_DATA_WIDTH-1:0]        wbuf_q;

    weight_buffer_32_bank #(
        .ARRAY_ROWS      (ARRAY_ROWS),
        .DATA_WIDTH      (PE_DATA_WIDTH),
        .BANK_ADDR_WIDTH (WBUF_BANK_ADDR_WIDTH),
        .WR_DATA_WIDTH   (AVS_DATA_WIDTH)
    ) u_weight_buffer (
        .clock         (clk),
        .data          (wbuf_wdata),
        .rdaddress     (wbuf_rdaddress),
        .rden          (wbuf_rden),
        .rd_byteenable (wbuf_byteenable),  // per-row masking, driven by weight_loader
        .wraddress     (wbuf_wraddress),
        .wren          (wbuf_wren),
        .q             (wbuf_q)
    );

    // ============================================================
    // input_buffer - write side from Avalon through the remap, read side
    // driven by input_dispatch
    // ============================================================
    wire [IBUF_WADDR_WIDTH-1:0] ibuf_host_waddr = (avs_address - IBUF_BASE) >> WORD_SHIFT;
    wire [IBUF_WADDR_WIDTH-1:0] ibuf_wraddress;
    wire [AVS_DATA_WIDTH-1:0]   ibuf_wdata;
    wire                        ibuf_wren;

    buf_addr_remap #(
        .DATA_WIDTH       (AVS_DATA_WIDTH),
        .HOST_WADDR_WIDTH (IBUF_WADDR_WIDTH),
        .ROWW_WIDTH       (IBUF_ROWW_WIDTH),
        .BANK_SEL_WIDTH   ($clog2(ARRAY_ROWS)),
        .BANK_WADDR_WIDTH (IBUF_BANK_ADDR_WIDTH - WORD_SHIFT)
    ) u_ibuf_remap (
        .clk        (clk),
        .host_waddr (ibuf_host_waddr),
        .host_wdata (avs_writedata),
        .host_wren  (avs_write_accepted && sel_ibuf),
        .row_words  (ibuf_pitch_words[IBUF_ROWW_WIDTH-1:0]),
        .bank_waddr (ibuf_wraddress),
        .bank_wdata (ibuf_wdata),
        .bank_wren  (ibuf_wren)
    );

    logic [ARRAY_ROWS*PE_DATA_WIDTH-1:0]       ibuf_q;
    logic [ARRAY_ROWS*IBUF_BANK_ADDR_WIDTH-1:0] ibuf_rdaddress;
    logic                                      ibuf_rden;
    logic [ARRAY_ROWS-1:0]                     ibuf_byteenable;

    input_buffer_32_bank #(
        .ARRAY_ROWS      (ARRAY_ROWS),
        .DATA_WIDTH      (PE_DATA_WIDTH),
        .BANK_ADDR_WIDTH (IBUF_BANK_ADDR_WIDTH),
        .WR_DATA_WIDTH   (AVS_DATA_WIDTH)
    ) u_input_buffer (
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
    // output_buffer - banked ARRAY_COLS-ways (one bank per p_out lane, see
    // output_buffer_32_bank.v), holding the output canvas (OBUF_CANVAS_W,
    // above). output_loader drives each bank's read+write directly; Avalon
    // reads through a single muxed external port, read-only. Entries are
    // int8 (PE_DATA_WIDTH) - output_loader rescales pe_array's wider
    // ACC_DATA_WIDTH accumulator down to int8 before it ever reaches
    // output_buffer (see output_loader.v).
    //
    // The host sees the canvas as one row-major image with a fixed
    // OBUF_CANVAS_W-byte pitch. That pitch is a power of two, so turning a
    // host word address into {bank, offset} is pure wiring - no divide, no
    // pipeline, and a read keeps the RAM's one cycle of latency:
    //   host word = {row block, bank, word in row}   (canvas row = {row block, bank})
    //   RAM addr  = {bank, row block, word in row}
    // ============================================================
    localparam int OBUF_ROW_WORDS_SHIFT = $clog2(OBUF_CANVAS_W) - WORD_SHIFT;  // log2(host words per canvas row)
    localparam int OBUF_BANK_SEL_WIDTH  = $clog2(ARRAY_COLS);

    // host word fetched this cycle: the burst's first word while the read
    // is still stalled, then one word per delivering cycle
    wire                        obuf_ext_rden  = (avs_read && sel_obuf) || (rd_burst_on && sel_obuf_r);
    wire [OBUF_RADDR_WIDTH-1:0] obuf_host_word =
        (rd_burst_on ? (addr_buf - OBUF_BASE) : (avs_address - OBUF_BASE)) >> WORD_SHIFT;
    wire [OBUF_RADDR_WIDTH-1:0] obuf_ext_rdaddress = {
        obuf_host_word[OBUF_ROW_WORDS_SHIFT +: OBUF_BANK_SEL_WIDTH],                       // bank = canvas row % banks
        obuf_host_word[OBUF_RADDR_WIDTH-1 : OBUF_ROW_WORDS_SHIFT + OBUF_BANK_SEL_WIDTH],   // canvas row / banks
        obuf_host_word[OBUF_ROW_WORDS_SHIFT-1 : 0]                                         // word within the row
    };
    logic [AVS_DATA_WIDTH-1:0] obuf_ext_q;

    // The delivering window is exactly `rd_burst_on && sel_obuf_r`: it goes
    // high the cycle the read is accepted, by which time obuf_ext_q holds the
    // word fetched during the stall cycle, and drops after burst_cnt_buf
    // beats. The fetch enable above is not the valid - it also fires during
    // the stall and one word past the end of the burst.
    assign obuf_data_valid = rd_burst_on && sel_obuf_r;

    logic        output_loader_busy;
    logic [ARRAY_COLS-1:0]                        obuf_bank_rden;
    logic [ARRAY_COLS*OBUF_BANK_RADDR_WIDTH-1:0]  obuf_bank_rdaddress;
    logic [ARRAY_COLS*AVS_DATA_WIDTH-1:0]         obuf_bank_q;
    logic [ARRAY_COLS-1:0]                        obuf_bank_wren;
    logic [ARRAY_COLS*OBUF_BANK_ADDR_WIDTH-1:0]   obuf_bank_waddr;
    logic [ARRAY_COLS*PE_DATA_WIDTH-1:0]          obuf_bank_wdata;

    output_buffer_32_bank #(
        .DATA_WIDTH      (PE_DATA_WIDTH),
        .NUM_BANKS       (ARRAY_COLS),
        .BANK_ADDR_WIDTH (OBUF_BANK_ADDR_WIDTH),
        .RD_DATA_WIDTH   (AVS_DATA_WIDTH)
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
    // `valid` must mark the cycle the access actually COMMITS, not merely
    // the cycle the master is presenting it. avs_waitrequest is still high
    // on the first cycle of a read (`accepted` is not set yet), so a raw
    // `avs_read && sel_ctrl` term asserts the register file's read strobe
    // one cycle before the master samples avs_readdata. That is harmless
    // for a plain status bit, but it destroys a read-to-clear field:
    // STATUS.done would be cleared on that stalled cycle and read back as 0
    // on the next. The accepted-read path (rd_burst_on && sel_ctrl_r) is
    // already asserted from the delivering cycle onward - and covers every
    // beat of a burst - so use only that for reads.
    wire        ctrl_valid = (avs_write_accepted && sel_ctrl) || (rd_burst_on && sel_ctrl_r);
    logic [AVS_DATA_WIDTH-1:0] ctrl_rdata;
    // ctrl_valid is the register file's *access* strobe, so it also covers
    // writes - it must not be reused as the read-data valid or every CTRL
    // write puts a phantom beat on the read channel. A write issued the
    // cycle after a read is accepted (which Avalon allows) then looks like
    // that read's valid stretching one cycle longer than the burst.
    wire        rf_data_valid = rd_burst_on && sel_ctrl_r;

    logic        matmul_start;          // pulse: latch WEIGHT_ROWS/WEIGHT_COLS, reset block position
    logic [15:0] weight_rows;           // K: weight rows (contraction dimension)
    // weight_cols / input_max_addr / input_cols are declared above, with the buffer remap
    logic [15:0] output_scale;          // Q0.16 rescale factor (value/65536, range [0,1)) for pe_array's ACC_WIDTH accumulator
    logic [14:0] tile_r_cfg;            // TILE_CFG.tile_r       - output tile row index (see the tile latch below)
    logic [14:0] tile_c_cfg;            // TILE_CFG.tile_c       - output tile column index
    logic        use_prev_out_cfg;      // TILE_CFG.use_prev_out - accumulate onto the canvas from the first block
    logic        weight_ctrl_busy;      // weight_ctrl/weight_loader busy, for STATUS.busy
    logic        weight_ctrl_done;      // PULSE: last block committed and its input band fully streamed in

    // STATUS.busy: weight/input side sequencing OR output_loader still
    // draining the last block's psums (weight_ctrl_done can go high a few
    // pipeline cycles before output_loader finishes writing that block's
    // results - see output_loader.v's header note on drain latency).
    wire        matmul_busy = weight_ctrl_busy || output_loader_busy;

    // STATUS.done is a read-to-clear sticky bit in the register file (see
    // rdl/ctrl_reg.rdl), set by a one-cycle write-enable pulse. Producing
    // that pulse takes two steps, because neither half of "the matmul
    // finished" is sufficient on its own:
    //
    //   - weight_ctrl's done pulse only means every block has been
    //     *committed*. output_loader is typically still draining the last
    //     block's psums for a few more cycles at that point (see
    //     output_loader.v's header note on drain latency), so firing here
    //     would let software start reading output_buffer before the final
    //     results have landed.
    //   - output_loader's own done pulses once *per committed block*, not
    //     once per matmul, so it can't be used directly either.
    //
    // So: latch weight_ctrl's pulse, then emit a single completion pulse on
    // the rising edge of "all blocks committed AND the drain has finished".
    logic all_blocks_committed;
    always_ff @(posedge clk or negedge reset_n) begin
        if (!reset_n)
            all_blocks_committed <= 1'b0;
        else if (matmul_start)          // a new matmul supersedes the old result
            all_blocks_committed <= 1'b0;
        else if (weight_ctrl_done)
            all_blocks_committed <= 1'b1;
    end

    wire  matmul_complete = all_blocks_committed && !output_loader_busy;
    logic matmul_complete_d;
    always_ff @(posedge clk or negedge reset_n) begin
        if (!reset_n) matmul_complete_d <= 1'b0;
        else          matmul_complete_d <= matmul_complete;
    end

    // one cycle, once per matmul
    wire matmul_done_pulse = matmul_complete && !matmul_complete_d;

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
        .TILE_CFG_tile_r_q       (tile_r_cfg),
        .TILE_CFG_tile_c_q       (tile_c_cfg),
        .TILE_CFG_use_prev_out_q (use_prev_out_cfg),

        .STATUS_busy_wdata              (matmul_busy),
        .STATUS_done_we                 (matmul_done_pulse),
        .STATUS_done_wdata              (1'b1),

        .valid  (ctrl_valid),
        .read   (avs_read),
        .addr   (avs_address),
        .wdata  (avs_writedata),
        .wmask  (avs_byteenable),
        .rdata  (ctrl_rdata)
    );

    // ============================================================
    // Output tile placement (TILE_CFG), latched at matmul_start so software
    // can program the next tile while this matmul runs.
    //
    // A matmul writes its K x M result into the canvas with its top-left
    // corner at canvas row tile_r*K, column tile_c*M, where K = WEIGHT_ROWS
    // and M = INPUT_COLS of that same matmul. So every tile of one output
    // uses the same K and M: zero-pad an edge tile rather than shrinking it.
    // The row offset must be a multiple of ARRAY_ROWS so that PE row r keeps
    // writing bank r - with tile_r > 0, K must be a multiple of ARRAY_ROWS
    // (the low bits are dropped). The column offset can be anything:
    // output_loader writes single bytes. Nothing checks that the tile lies
    // inside the canvas.
    //
    // use_prev_out makes the matmul's first contraction block accumulate
    // into what the canvas already holds, instead of overwriting it, so a
    // contraction too long for one matmul can be split across several.
    // ============================================================
    // operands cut to the canvas index width: a tile inside the canvas never
    // needs more, and it keeps both multipliers small
    localparam int CANVAS_IDX_W = $clog2(OBUF_CANVAS_H > OBUF_CANVAS_W ? OBUF_CANVAS_H : OBUF_CANVAS_W) + 1;
    logic [15:0]                     tile_row_blk;   // first canvas row / ARRAY_ROWS
    logic [OBUF_BANK_ADDR_WIDTH-1:0] tile_col_off;   // first canvas column (bytes)
    logic                            use_prev_out;
    always_ff @(posedge clk or negedge reset_n) begin
        if (!reset_n) begin
            tile_row_blk <= '0;
            tile_col_off <= '0;
            use_prev_out <= 1'b0;
        end
        else if (matmul_start) begin
            tile_row_blk <= (tile_r_cfg[CANVAS_IDX_W-1:0] * weight_rows[CANVAS_IDX_W-1:0]) >> $clog2(ARRAY_ROWS);
            tile_col_off <= tile_c_cfg[CANVAS_IDX_W-1:0] * input_cols[CANVAS_IDX_W-1:0];
            use_prev_out <= use_prev_out_cfg;
        end
    end

    // ============================================================
    // Avalon read mux / waitrequest
    // ============================================================
    // OBUF and CTRL beats never share a cycle: each belongs to its own read,
    // and the read FSM handles one read at a time.
    wire [AVS_DATA_WIDTH-1:0] read_data;
    assign read_data = obuf_data_valid ? obuf_ext_q :
                       rf_data_valid   ? ctrl_rdata : '0;

    // ============================================================
    // input_dispatch - streams input_buffer into the array's left edge,
    // one ARRAY_ROWS-row band at a time, driven by weight_ctrl below
    // ============================================================
    wire [ARRAY_ROWS*PE_DATA_WIDTH-1:0] a_out;
    wire [ARRAY_ROWS-1:0]               a_en;

    wire [IBUF_BANK_ADDR_WIDTH-1:0] input_band_base_addr;  // weight_ctrl -> input_dispatch: current block's per-bank base offset
    wire       input_band_start;      // weight_ctrl -> input_dispatch: pulse, begin preloading that band
    wire       input_compute_start;   // weight_ctrl -> input_dispatch: pulse, begin streaming into pe_array
    wire       input_fifos_primed;    // input_dispatch -> weight_ctrl: all row FIFOs hold >=1 element
    wire       input_band_done;       // input_dispatch -> weight_ctrl: current band fully drained

    input_dispatch #(
        .ARRAY_ROWS      (ARRAY_ROWS),
        .DATA_WIDTH      (PE_DATA_WIDTH),
        .IBUF_ADDR_WIDTH (IBUF_BANK_ADDR_WIDTH)
    ) u_input_dispatch (
        .clock          (clk),
        .rst_n          (reset_n),

        .ibuf_q         (ibuf_q),
        .ibuf_rdaddress (ibuf_rdaddress),
        .ibuf_rden      (ibuf_rden),
        .ibuf_byteenable(ibuf_byteenable),

        .a_out          (a_out),
        .a_en           (a_en),

        .i_max_addr       (input_max_addr[IBUF_BANK_ADDR_WIDTH-1:0]),
        .i_band_base_addr (input_band_base_addr),
        .i_band_start     (input_band_start),
        .i_start_compute  (input_compute_start),
        .fifos_primed     (input_fifos_primed),
        .band_done        (input_band_done)
    );

    // ============================================================
    // weight_ctrl - walks the weight matrix column-major (contraction-
    // dimension blocks fast, output-dimension blocks slow) as
    // ARRAY_ROWS x ARRAY_COLS
    // blocks, prefetching each into pe_array and driving input_dispatch's
    // per-block handshake; matmul_start is the only software trigger,
    // everything else is sequenced internally
    // ============================================================
    wire [ARRAY_ROWS*$clog2(ARRAY_ROWS*ARRAY_COLS)-1:0] w_addr;  // one port per PE row
    wire [ARRAY_ROWS*PE_DATA_WIDTH-1:0]                 w_data;
    wire                                      w_en;
    wire                                      w_load;
    wire [15:0]                              committed_n_blk_idx;    // -> output_loader
    wire [15:0]                              committed_k_blk_idx;
    wire                                      committed_first_k_blk; // -> output_loader
    wire [$clog2(ARRAY_ROWS+1)-1:0]         valid_row;
    wire [$clog2(ARRAY_COLS+1)-1:0]         valid_col;

    weight_ctrl #(
        .DATA_WIDTH      (PE_DATA_WIDTH),
        .ARRAY_ROWS      (ARRAY_ROWS),
        .ARRAY_COLS      (ARRAY_COLS),
        .WBUF_BANK_ADDR_WIDTH (WBUF_BANK_ADDR_WIDTH),
        .ROW_ALIGN       (BYTES_PER_WORD),
        .IBUF_ADDR_WIDTH (IBUF_BANK_ADDR_WIDTH),
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
        .wbuf_byteenable   (wbuf_byteenable),
        .wbuf_q            (wbuf_q),

        .w_addr            (w_addr),
        .w_data            (w_data),
        .w_en              (w_en),
        .w_load            (w_load),

        .input_max_addr       (input_max_addr[IBUF_BANK_ADDR_WIDTH-1:0]),
        .input_band_start     (input_band_start),
        .input_band_base_addr (input_band_base_addr),
        .input_compute_start  (input_compute_start),
        .input_fifos_primed   (input_fifos_primed),
        .input_band_done      (input_band_done),

        .output_drain_busy    (output_loader_busy),  // no commit until the previous block has left the array

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
        .CANVAS_W_SHIFT ($clog2(OBUF_CANVAS_W)),
        .RD_DATA_WIDTH  (AVS_DATA_WIDTH),
        .DIM_WIDTH       (16)
    ) u_output_loader (
        .clock (clk),
        .rst_n (reset_n),

        .p_in  (pe_p_out),
        .b_en  (pe_b_en),

        .a_en_last             (a_en[ARRAY_ROWS-1]),   // last row's a_en - see output_loader.v header
        .total_rows            (input_cols[OBUF_BANK_ADDR_WIDTH-1:0]),
        .committed_k_blk_idx   (committed_k_blk_idx),
        .committed_n_blk_idx   (committed_n_blk_idx),
        .committed_first_k_blk (committed_first_k_blk),

        .tile_row_blk   (tile_row_blk),
        .tile_col_off   (tile_col_off),
        .use_prev_out   (use_prev_out),

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
// dispatches activation data read from the input_buffer_32_bank into the
// systolic array's row-wise activation inputs (a_in/a_en on pe_array).
//
// input_buffer_32_bank gives every array row its own dedicated bank, so a
// single shared rdaddress pulls exactly one element per row per cycle -
// ibuf_q's lane i is always row i's next value, no reshuffling needed.
// input_buffer holds the x matrix row-major, but only ever a handful of
// 32-row bands at a time - not the whole matrix. weight_ctrl re-triggers
// this module once per weight block: i_band_start pulses with a new
// i_band_base_addr (that block's offset into every row's bank) each time
// weight_ctrl moves to a different contraction-dimension block
// (k_blk_idx). Preload fills the row FIFOs from that band; once every FIFO
// holds >=1 element (fifos_primed) weight_ctrl fires i_start_compute once
// its own weight block is also ready, and this module streams the primed
// rows into the array (OFFLOAD). band_done pulses once that band is fully
// drained, which is weight_ctrl's cue that the *next* i_band_start is safe
// to issue (the row FIFOs aren't double-buffered, so the next band can't
// be preloaded until this one has finished draining).
module input_dispatch #(
    parameter ARRAY_ROWS   = 32,
    parameter DATA_WIDTH   = 32
)(
    input                                     clock,
    input                                     rst_n,

    // input_buffer_32_bank read-side connections - one dedicated 8-bit
    // lane per row, all banks sharing the same rdaddress
    input       [ARRAY_ROWS*8-1:0]            ibuf_q,
    output reg  [ARRAY_ROWS*8-1:0]            ibuf_rdaddress,
    output reg                                ibuf_rden,
    output reg  [ARRAY_ROWS-1:0]              ibuf_byteenable,


    // systolic array activation input, left edge - one lane per row
    output      [ARRAY_ROWS*DATA_WIDTH-1:0]   a_out,
    output reg  [ARRAY_ROWS-1:0]              a_en,

    // per-band read window, driven by weight_ctrl
    input       [7:0]                         i_max_addr,       // per-band bound (number of reads to issue)
    input       [7:0]                         i_band_base_addr, // this band's base offset in every row's bank
    input                                     i_band_start,     // pulse: begin preloading i_band_base_addr's band
    input                                     i_start_compute,  // pulse: begin streaming the preloaded band into the array
    output                                    fifos_primed,     // level: every row FIFO holds >=1 element
    output reg                                band_done         // pulse: this band fully drained
);

    reg  [ARRAY_ROWS-1:0]     fifo_wren;
    wire [ARRAY_ROWS-1:0]     ififo_full;
    wire [ARRAY_ROWS-1:0]     ififo_empty;
    reg  [ARRAY_ROWS-1:0]     fifo_rden;

    assign fifos_primed = ~(|ififo_empty[ARRAY_ROWS-1:0]);
    integer fifo_wren_idx;

    always@(posedge clock)begin
        if(~rst_n)begin
            fifo_wren <= 0;
        end
        else begin
            for(fifo_wren_idx = 0; fifo_wren_idx < ARRAY_ROWS; fifo_wren_idx=fifo_wren_idx+1)begin
                fifo_wren[fifo_wren_idx] <= ibuf_rden && ibuf_byteenable[fifo_wren_idx];
            end
        end
    end

    genvar ififo_gen_idx;

    generate
        for(ififo_gen_idx = 0; ififo_gen_idx < ARRAY_ROWS; ififo_gen_idx=ififo_gen_idx+1)begin
            sync_fifo_w8_d32 input_fifo (
                .clock  (clock),
                .data   (ibuf_q[ififo_gen_idx * 8 +: 8]),
                .rdreq  (fifo_rden[ififo_gen_idx]),
                .wrreq  (fifo_wren[ififo_gen_idx]),
                .empty  (ififo_empty[ififo_gen_idx]),
                .full   (ififo_full[ififo_gen_idx]),
                .q      (a_out[ififo_gen_idx * DATA_WIDTH +: DATA_WIDTH])
            );

            always @(posedge clock) begin
                if(~rst_n)begin
                    a_en[ififo_gen_idx] <= 0;
                end
                else begin
                    a_en[ififo_gen_idx] <= fifo_rden[ififo_gen_idx];
                end
            end
        end

    endgenerate

    localparam STATE_IDLE = 0,
               STATE_PRELOAD = 1,
               STATE_OFFLOAD = 2,
               STATE_DONE = 3;

    reg  [ 1:0]  dispatch_state;

    reg  [ARRAY_ROWS*8-1:0]  read_count;    // reads issued this band
    reg  [ 6:0]  skewed_start_cnt;
    wire [ARRAY_ROWS-1:0]    ibuf_rd_done;
    integer ififo_idx;

    integer i;
    // input buffer read out logic - every accepted read pulls one element
    // per row (all banks read together at the same offset), so the offset
    genvar ibufdone_idx;
    generate
        for(ibufdone_idx = 0; ibufdone_idx < ARRAY_ROWS; ibufdone_idx = ibufdone_idx + 1)begin
            assign ibuf_rd_done[ibufdone_idx] = (read_count[ibufdone_idx*8 +: 8] == i_max_addr);
        end
    endgenerate

    // ---- per-row fill-level tracking, to gate new reads without relying
    // on ififo_full[i] alone ----
    // ififo_full[i] lags real fifo occupancy: a read decided this cycle
    // (read_count[i] incremented, ibuf_byteenable[i] set) doesn't actually
    // land in that row's fifo until 2 cycles later (1 cycle for
    // input_buffer's synchronous read latency, +1 for the fifo_wren
    // register), so ififo_full[i] can still read 0 for up to 2 cycles
    // after the fifo has effectively committed to being full - reads
    // issued in that window get silently dropped by the fifo's own
    // internal wr_en && ~full gating (sync_fifo.v), since fifo depth
    // (32) is sized with zero spare margin for exactly this many
    // in-flight rows (see tb/models/sync_fifo_w8_d32.v's header note).
    // read_count[i] (pushes decided) and pop_count[i] (pops actually
    // accepted, real-time) are both driven by this module itself with no
    // such lag, so (read_count[i]-pop_count[i]) is an always-accurate fill
    // level - gating on it instead avoids the race entirely. ififo_full[i]
    // is kept as an extra AND term below purely as a backstop; it should
    // never be the binding condition once this is in place.
    reg  [ARRAY_ROWS*8-1:0]  pop_count;     // pops actually accepted so far this band, per row
    integer pop_idx;

    always@(posedge clock)begin
        if(~rst_n)begin
            pop_count <= 0;
        end
        else if(dispatch_state == STATE_IDLE)begin
            pop_count <= 0;   // mirrors read_count's per-band reset below
        end
        else begin
            for(pop_idx = 0; pop_idx < ARRAY_ROWS; pop_idx = pop_idx + 1)begin
                if(fifo_rden[pop_idx] && ~ififo_empty[pop_idx])
                    pop_count[pop_idx*8 +: 8] <= pop_count[pop_idx*8 +: 8] + 1'b1;
            end
        end
    end


    // just steps by 1 each cycle
    always@(posedge clock)begin
        if(~rst_n)begin
            ibuf_rdaddress <= 0;
            dispatch_state <= STATE_IDLE;
            fifo_rden <= 0;
            skewed_start_cnt <= 0;
            read_count <= 0;
            band_done <= 1'b0;
            ibuf_rden <= 0;
            ibuf_byteenable <= 32'hFFFFFFFF;
        end
        else begin
            band_done <= 1'b0;

            case(dispatch_state)
                STATE_IDLE: begin
                    read_count       <= 0;
                    if(i_band_start)begin
                        ibuf_rdaddress   <= {(ARRAY_ROWS){i_band_base_addr}};
                        ibuf_rden        <= 1'b1;
                        ibuf_byteenable  <= 32'hFFFFFFFF;
                        for(i=0; i < ARRAY_ROWS; i=i+1)begin
                            read_count[i*8 +: 8] <= read_count[i*8 +: 8] + 1;
                        end
                        dispatch_state   <= STATE_PRELOAD;
                    end
                end

                STATE_PRELOAD: begin
                    for(i=0; i < ARRAY_ROWS; i=i+1)begin
                        if(read_count[i*8+:8] < i_max_addr
                           && (read_count[i*8+:8] - pop_count[i*8+:8]) < 32
                           && ~ififo_full[i])begin
                            ibuf_rdaddress[i*8+:8] <= ibuf_rdaddress[i*8+:8] + 1'b1;
                            read_count[i*8+:8]     <= read_count[i*8+:8] + 1'b1;
                            ibuf_byteenable[i]     <= 1;
                        end
                        else begin
                            ibuf_byteenable[i]     <= 0;
                        end
                    end

                    if(&ibuf_rd_done || (&ififo_full))begin
                        ibuf_rden <= 0;
                    end
                    else begin
                        ibuf_rden <= 1;
                    end

                    if(i_start_compute)begin
                        dispatch_state <= STATE_OFFLOAD;
                    end
                end

                STATE_OFFLOAD: begin
                    for(i=0; i < ARRAY_ROWS; i=i+1)begin
                        if(read_count[i*8+:8] < i_max_addr
                           && (read_count[i*8+:8] - pop_count[i*8+:8]) < 32
                           && ~ififo_full[i])begin
                            ibuf_rdaddress[i*8+:8] <= ibuf_rdaddress[i*8+:8] + 1'b1;
                            read_count[i*8+:8]     <= read_count[i*8+:8] + 1'b1;
                            ibuf_byteenable[i]     <= 1;
                        end
                        else begin
                            ibuf_byteenable[i]     <= 0;
                        end
                    end

                    if(&ibuf_rd_done || (&ififo_full))begin
                        ibuf_rden <= 0;
                    end
                    else begin
                        ibuf_rden <= 1;
                    end

                    if(skewed_start_cnt < 'd32)begin
                        fifo_rden[skewed_start_cnt] <= 1;
                        skewed_start_cnt <= skewed_start_cnt + 1;
                    end

                    // if(skewed_start_cnt >= 'd32 && skewed_start_cnt < 'd64)begin
                    //     fifo_rden[skewed_start_cnt-32] <= 0;
                    //     skewed_start_cnt <= skewed_start_cnt + 1;
                    // end

                    for(ififo_idx = 0; ififo_idx < 32; ififo_idx = ififo_idx + 1)begin
                        if(ififo_empty[ififo_idx])
                            fifo_rden[ififo_idx] <= 0;
                    end

                    if(&ififo_empty == 1)begin
                        dispatch_state <= STATE_DONE;
                        band_done <= 1'b1;
                    end
                end

                STATE_DONE: begin
                    ibuf_rden <= 0;
                    skewed_start_cnt <= 0;
                    dispatch_state <= STATE_IDLE;
                    read_count <= 0;
                end


            endcase
        end
    end

endmodule

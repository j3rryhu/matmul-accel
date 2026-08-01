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
    output reg  [7:0]                         ibuf_rdaddress,
    output reg                                ibuf_rden,


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

    reg             fifo_wren;
    wire [31:0]     ififo_full;
    wire [31:0]     ififo_empty;
    reg  [31:0]     fifo_rden;

    assign fifos_primed = ~(|ififo_empty[ARRAY_ROWS-1:0]);

    always@(posedge clock)begin
        if(~rst_n)begin
            fifo_wren <= 0;
        end
        else begin
            fifo_wren <= ibuf_rden;
        end
    end

    genvar i;

    generate
        for(i = 0; i < ARRAY_ROWS; i=i+1)begin
            sync_fifo_w8_d32 input_fifo (
                .clock  (clock),
                .data   (ibuf_q[i * 8 +: 8]),
                .rdreq  (fifo_rden[i]),
                .wrreq  (fifo_wren),
                .empty  (ififo_empty[i]),
                .full   (ififo_full[i]),
                .q      (a_out[i * DATA_WIDTH +: DATA_WIDTH])
            );

            always @(posedge clock) begin
                if(~rst_n)begin
                    a_en[i] <= 0;
                end
                else begin
                    a_en[i] <= fifo_rden[i];
                end
            end
        end

    endgenerate

    localparam STATE_IDLE = 0,
               STATE_PRELOAD = 1,
               STATE_OFFLOAD = 2,
               STATE_DONE = 3;

    reg  [ 1:0]  dispatch_state;

    reg  [ 7:0]  read_count;    // reads issued this band
    reg  [ 6:0]  skewed_start_cnt;
    integer ififo_idx;


    // input buffer read out logic - every accepted read pulls one element
    // per row (all banks read together at the same offset), so the offset
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
        end
        else begin
            band_done <= 1'b0;

            case(dispatch_state)
                STATE_IDLE: begin
                    read_count       <= 0;
                    if(i_band_start)begin
                       ibuf_rdaddress   <= i_band_base_addr;
                       ibuf_rden        <= 1'b1;
                       read_count       <= read_count + 1;
                       dispatch_state   <= STATE_PRELOAD;
                    end
                end

                STATE_PRELOAD: begin
                    if(read_count < i_max_addr && ~(|ififo_full))begin
                        ibuf_rdaddress <= ibuf_rdaddress + 1'b1;
                        ibuf_rden      <= 1'b1;
                        read_count     <= read_count + 1'b1;
                    end

                    if(read_count == i_max_addr)begin
                        ibuf_rden <= 0;
                    end

                    if(i_start_compute)begin
                        dispatch_state <= STATE_OFFLOAD;
                    end
                end

                STATE_OFFLOAD: begin
                    if(read_count < i_max_addr && ~(|ififo_full))begin
                        ibuf_rdaddress <= ibuf_rdaddress + 1'b1;
                        ibuf_rden      <= 1'b1;
                        read_count     <= read_count + 1'b1;
                    end

                    if(skewed_start_cnt < 'd32)begin
                        fifo_rden[skewed_start_cnt] <= 1;
                        skewed_start_cnt <= skewed_start_cnt + 1;
                    end

                    if(skewed_start_cnt >= 'd32 && skewed_start_cnt < 'd64)begin
                        fifo_rden[skewed_start_cnt-32] <= 0;
                        skewed_start_cnt <= skewed_start_cnt + 1;
                    end

                    for(ififo_idx = 0; ififo_idx < 32; ififo_idx = ififo_idx + 1)begin
                        if(ififo_empty[ififo_idx])
                            fifo_rden[ififo_idx] <= 0;
                    end

                    if(read_count == i_max_addr)begin
                        ibuf_rden <= 0;
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
                end


            endcase
        end
    end

endmodule

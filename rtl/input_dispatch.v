// dispatches activation data read from the input_buffer_8_bank into the
// systolic array's row-wise activation inputs (a_in/a_en on pe_array)
module input_dispatch #(
    parameter NUM_BANKS    = 8,
    parameter BANK_WIDTH   = 8,
    parameter ARRAY_ROWS   = 32,
    parameter DATA_WIDTH   = 32
)(
    input                                     clock,
    input                                     rst_n,

    // input_buffer_8_bank read-side connections
    input       [NUM_BANKS*BANK_WIDTH-1:0]    ibuf_q,
    output reg  [9:0]                         ibuf_rdaddress,
    output                                    ibuf_rden,


    // systolic array activation input, left edge - one lane per row
    output      [ARRAY_ROWS*DATA_WIDTH-1:0]   a_out,
    output reg  [ARRAY_ROWS-1:0]              a_en,

    // max_addr for reading out input buffer
    input       [9:0]                         i_max_addr,    // the max addr of input matrix, derived from input dimensions W*H/8
    input                                     i_ibuf_rdy,
    input                                     i_start_compute
);

    reg             fifo_wren;
    wire [31:0]     ififo_full;
    wire [31:0]     ififo_empty;
    reg  [31:0]     fifo_rden;
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
            wire [7:0] data_in = (i >= bank_idx * 8 && i < (bank_idx * 8 + 8)) ? ibuf_q[(i - bank_idx * 8) * 8 +: 8] : 0;

            sync_fifo_w8_d16 input_fifo (
                .clock  (clock),
                .data   (data_in),
                .rdreq  (fifo_rden[i]),
                .wrreq  (fifo_wren && i >= bank_idx * 8 && i < (bank_idx * 8 + 8)),
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
    reg  [ 1:0]  bank_idx;

    reg  [ 5:0]  skewed_start_cnt;
    integer ififo_idx;


    // input buffer read out logic
    always@(posedge clock)begin
        if(~rst_n)begin
            ibuf_rdaddress <= 0; 
            dispatch_state <= STATE_IDLE;
            bank_idx <= 0;
            fifo_rden <= 0;
            skewed_start_cnt <= 0;
        end
        else begin
            case(dispatch_state)
                STATE_IDLE: begin
                    if(i_ibuf_rdy)begin
                       dispatch_state <= STATE_PRELOAD; 
                    end
                end

                STATE_PRELOAD: begin
                    if(ibuf_rdaddress < i_max_addr && ~(&ififo_full))begin
                        ibuf_rdaddress <= ibuf_rdaddress + 'd1;
                        ibuf_rden <= 1'b1;
                        bank_idx <= bank_idx + 1;
                    end

                    if(i_start_compute)begin
                        dispatch_state <= STATE_OFFLOAD;
                    end
                end

                STATE_OFFLOAD: begin
                    if(ibuf_rdaddress < i_max_addr && ~(&ififo_full))begin
                        ibuf_rdaddress <= ibuf_rdaddress + 'd1;
                        ibuf_rden <= 1'b1;
                        bank_idx <= bank_idx + 1;
                    end

                    if(skewed_start_cnt < 'd32)begin
                        fifo_rden[skewed_start_cnt] <= 1;
                        skewed_start_cnt <= skewed_start_cnt + 1;
                    end

                    for(ififo_idx = 0; ififo_idx < 32; ififo_idx = ififo_idx + 1)begin
                        if(ififo_empty[ififo_idx])
                            fifo_rden[ififo_idx] <= 0;
                    end

                    if(ibuf_rdaddress == i_max_addr && |ififo_empty == 0)begin
                        dispatch_state <= STATE_DONE;
                    end
                end

                STATE_DONE: begin
                    ibuf_rden <= 0;
                    ibuf_rdaddress <= 0;
                    bank_idx <= 0;
                    skewed_start_cnt <= 0;

                    if(i_ibuf_rdy)
                        dispatch_state <= STATE_PRELOAD;
                    else
                        dispatch_state <= STATE_IDLE;
                end


            endcase
        end
    end

endmodule

`timescale 1ns/1ps
module tb_gelu_int8;
    reg clk = 0;
    always #5 clk = ~clk;
    reg reset_n = 0;
    reg signed [7:0] in_data = 0;
    reg in_valid = 0;
    wire in_ready;
    wire signed [7:0] out_data;
    wire out_valid;
    reg out_ready = 0;
    reg [7:0] expected [0:255];
    integer sent = 0, received = 0, cycles = 0;
    integer queue [0:1023];
    reg stalled = 0;
    reg passed = 0;
    reg [7:0] held;
    gelu_int8 dut(.*);
    initial $readmemh("activation/example/gelu_int8.hex", expected);

    always @(posedge clk) begin
        if (reset_n) begin
            if (stalled && (!out_valid || out_data !== held))
                $fatal(1, "Output changed under backpressure");
            if (out_valid && out_ready) begin
                if (received >= sent)
                    $fatal(1, "Unexpected output");
                if (out_data !== expected[queue[received]])
                    $fatal(1, "Mismatch at sample %0d", received);
                received = received + 1;
            end
            if (in_valid && in_ready) begin
                queue[sent] = $unsigned(in_data);
                sent = sent + 1;
            end
            stalled = out_valid && !out_ready;
            held = out_data;
        end else begin
            sent = 0;
            received = 0;
            stalled = 0;
        end
    end

    task run_codes;
        input integer with_stalls;
        integer code;
        begin
            code = 0;
            while (code < 256) begin
                @(negedge clk);
                out_ready = !with_stalls || (cycles % 5 != 0 && cycles % 5 != 1);
                in_valid = !with_stalls || cycles % 7 != 0;
                in_data = code;
                cycles = cycles + 1;
                @(posedge clk);
                if (in_valid && in_ready) code = code + 1;
            end
            @(negedge clk);
            in_valid = 0;
            out_ready = 1;
            while (received < sent) @(negedge clk);
        end
    endtask

    initial begin
        repeat (3) @(negedge clk);
        reset_n = 1;
        run_codes(0);
        run_codes(1);
        if (received != 512) $fatal(1, "Missing samples");
        // Reset must discard a pending, stalled output.
        @(negedge clk);
        out_ready = 0;
        in_valid = 1;
        in_data = -32;
        @(negedge clk);
        in_valid = 0;
        reset_n = 0;
        @(negedge clk);
        if (out_valid !== 0 || in_ready !== 0)
            $fatal(1, "Reset did not clear/gate handshake");
        reset_n = 1;
        run_codes(1);
        if (received != 256) $fatal(1, "Post-reset sample count wrong");
        $display("PASS: 768 LUT lookups, bubbles, backpressure, reset");
        passed = 1;
        $finish;
    end
    initial begin
        #100000;
        $fatal(1, "Simulation timeout");
    end
endmodule

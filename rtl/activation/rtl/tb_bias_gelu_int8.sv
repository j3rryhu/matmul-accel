`timescale 1ns/1ps
module tb_bias_gelu_int8;
    localparam LANES = 16;
    localparam W = LANES*8;
    reg clk = 0;
    always #5 clk = ~clk;
    reg reset_n = 0;
    reg [W-1:0] matmul_data = 0, bias_data = 0;
    reg [LANES-1:0] in_mask = 0;
    reg in_valid = 0, out_ready = 0;
    wire in_ready, out_valid;
    wire [W-1:0] out_data;
    wire [LANES-1:0] out_mask;
    reg [7:0] lut [0:255];
    // At most one word is buffered in the DUT, so a small ring suffices.
    reg [W-1:0] exp_data [0:15];
    reg [LANES-1:0] exp_mask [0:15];
    integer sent = 0, received = 0, ticks = 0;
    integer l, sum, clipped;
    integer beat, dl, p, seed = 32'h5eed;
    reg stalled = 0;
    reg passed = 0;
    reg [W-1:0] held_data;
    reg [LANES-1:0] held_mask;
    reg [W-1:0] m, b;
    reg [LANES-1:0] mk;
    bias_gelu_int8 #(.LANES(LANES)) dut(.*);
    initial $readmemh("activation/example/gelu_int8.hex", lut);

    always @(posedge clk) begin
        if (reset_n) begin
            if (stalled && (!out_valid || out_data !== held_data ||
                            out_mask !== held_mask))
                $fatal(1, "Output changed while stalled");
            if (out_valid && out_ready) begin
                if (received >= sent)
                    $fatal(1, "Unexpected output word");
                if (out_mask !== exp_mask[received % 16] ||
                    out_data !== exp_data[received % 16])
                    $fatal(1, "Mismatch at word %0d: got %h mask %h, expected %h mask %h",
                           received, out_data, out_mask,
                           exp_data[received % 16], exp_mask[received % 16]);
                received = received + 1;
            end
            if (in_valid && in_ready) begin
                if (sent - received >= 16)
                    $fatal(1, "Scoreboard overflow");
                exp_mask[sent % 16] = in_mask;
                for (l = 0; l < LANES; l = l + 1) begin
                    if (in_mask[l]) begin
                        // Integer arithmetic independently models widened addition.
                        sum = $signed(matmul_data[l*8 +: 8]);
                        sum = sum + $signed(bias_data[l*8 +: 8]);
                        clipped = sum > 127 ? 127 : (sum < -128 ? -128 : sum);
                        if ($signed(dut.wide_sum[l*9 +: 9]) !== sum ||
                            $signed(dut.clipped_sum[l*8 +: 8]) !== clipped)
                            $fatal(1, "Lane %0d adder/saturation mismatch for %0d + %0d",
                                   l, $signed(matmul_data[l*8 +: 8]),
                                   $signed(bias_data[l*8 +: 8]));
                        exp_data[sent % 16][l*8 +: 8] = lut[clipped & 255];
                    end else begin
                        exp_data[sent % 16][l*8 +: 8] = matmul_data[l*8 +: 8];
                    end
                end
                sent = sent + 1;
            end
            stalled = out_valid && !out_ready;
            held_data = out_data;
            held_mask = out_mask;
        end else begin
            sent = 0;
            received = 0;
            stalled = 0;
        end
    end

    // Present one word with bubbles; hold all operands until accepted while
    // downstream applies randomized multi-cycle backpressure.
    task send_word(input [LANES-1:0] mask, input [W-1:0] md, input [W-1:0] bd);
        begin
            @(negedge clk);
            in_valid = 0;
            matmul_data = md;
            bias_data = bd;
            in_mask = mask;
            ticks = ticks + 1;
            out_ready = ticks % 5 != 0;
            if (ticks % 7 == 0) @(negedge clk);
            in_valid = 1;
            @(posedge clk);
            while (!in_ready) begin
                @(negedge clk);
                out_ready = ($random(seed) & 3) != 0;
                @(posedge clk);
            end
        end
    endtask

    task drain;
        begin
            @(negedge clk);
            in_valid = 0;
            out_ready = 1;
            while (received < sent) @(negedge clk);
        end
    endtask

    initial begin
        repeat (3) @(negedge clk);
        reset_n = 1;

        // Phase 1: full mask. Each lane walks all 65,536 (matmul, bias) pairs
        // from a different starting offset, so every lane ROM/adder is exhaustive.
        for (beat = 0; beat < 65536; beat = beat + 1) begin
            for (dl = 0; dl < LANES; dl = dl + 1) begin
                p = (beat + dl * 4099) & 16'hffff;
                m[dl*8 +: 8] = p[15:8];
                b[dl*8 +: 8] = p[7:0];
            end
            send_word({LANES{1'b1}}, m, b);
        end
        drain;
        if (received != 65536) $fatal(1, "Phase 1 transfer count %0d", received);

        // Phase 2: lane masks. All-zero, one-hot, and random masks. Masked lanes
        // must pass matmul through; on odd beats their bias is X (don't-care).
        for (beat = 0; beat < 4096; beat = beat + 1) begin
            case (beat % 8)
                0: mk = {LANES{1'b0}};
                1: mk = {{(LANES-1){1'b0}}, 1'b1} << ((beat / 8) % LANES);
                2: mk = ~({{(LANES-1){1'b0}}, 1'b1} << ((beat / 8) % LANES));
                default: mk = $random(seed);
            endcase
            for (dl = 0; dl < LANES; dl = dl + 1) begin
                m[dl*8 +: 8] = $random(seed);
                b[dl*8 +: 8] = $random(seed);
                if (!mk[dl] && beat % 2)
                    b[dl*8 +: 8] = 8'bx;
            end
            send_word(mk, m, b);
        end
        drain;
        if (received != 65536 + 4096) $fatal(1, "Phase 2 transfer count %0d", received);

        // Phase 3: reset must discard a pending, stalled output and clear the mask.
        @(negedge clk);
        out_ready = 0;
        in_valid = 1;
        in_mask = {LANES{1'b1}};
        matmul_data = {LANES{8'h20}};
        bias_data = {LANES{8'h00}};
        @(negedge clk);
        in_valid = 0;
        if (out_valid !== 1) $fatal(1, "Pending word not captured");
        reset_n = 0;
        @(negedge clk);
        if (out_valid !== 0 || in_ready !== 0 || out_mask !== 0 || out_data !== 0)
            $fatal(1, "Reset did not clear/gate handshake, mask and data");
        reset_n = 1;
        for (beat = 0; beat < 64; beat = beat + 1) begin
            for (dl = 0; dl < LANES; dl = dl + 1) begin
                m[dl*8 +: 8] = $random(seed);
                b[dl*8 +: 8] = $random(seed);
            end
            send_word($random(seed), m, b);
        end
        drain;
        if (received != 64) $fatal(1, "Post-reset transfer count %0d", received);

        $display("PASS: %0d lanes x 65536 int8 bias pairs, lane masks, backpressure, reset",
                 LANES);
        passed = 1;
        $finish;
    end
    initial begin
        #5000000;
        $fatal(1, "Simulation timeout");
    end
endmodule

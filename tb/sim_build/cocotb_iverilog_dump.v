module cocotb_iverilog_dump();
initial begin
    $dumpfile("sim_build/accel_top.fst");
    $dumpvars(0, accel_top);
end
endmodule

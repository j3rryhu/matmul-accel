# Run from the matmul-soc repository root:
# vsim -c -do activation/rtl/run_modelsim.do
onerror {quit -code 1 -force}
if {![file exists activation/rtl/work]} {vlib activation/rtl/work}
vmap work activation/rtl/work
vlog -sv activation/rtl/gelu_int8.sv activation/rtl/tb_gelu_int8.sv
vlog -sv activation/rtl/bias_gelu_int8.sv activation/rtl/tb_bias_gelu_int8.sv
vlog -sv activation/rtl/matmul_gelu_readout.v activation/rtl/tb_matmul_gelu_readout.sv
vsim -onfinish stop work.tb_gelu_int8
onbreak {resume}
run -all
if {[examine -radix unsigned sim:/tb_gelu_int8/passed] != 1} {quit -code 1 -force}
quit -sim
vsim -onfinish stop work.tb_bias_gelu_int8
run -all
if {[examine -radix unsigned sim:/tb_bias_gelu_int8/passed] != 1} {quit -code 1 -force}
quit -sim
vsim -onfinish stop work.tb_matmul_gelu_readout
run -all
if {[examine -radix unsigned sim:/tb_matmul_gelu_readout/passed] != 1} {quit -code 1 -force}
quit -code 0 -force

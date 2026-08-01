# cocotb + icarus verilog testbench for accel_top.
#
# Buffer-RAM IP (weight_buffer/input_buffer/output_buffer) and FIFO IP
# (sync_fifo_w8_d32) aren't in this repo - models/ provides simulation-only
# behavioral stand-ins with matching module names/ports (see
# models/buffer_ram_models.v, models/sync_fifo_w8_d32.v). Swap those two
# files out of VERILOG_SOURCES for the real generated IP once it exists;
# nothing else needs to change.
#
# Usage:
#   make        # run the test
#   make clean

SIM ?= icarus
TOPLEVEL_LANG ?= verilog

PWD        := $(shell pwd)
RTL_DIR    := $(PWD)/../rtl
RDL_SV_DIR := $(PWD)/../rdl/ctrl_reg.sv
MODEL_DIR  := $(PWD)/models

COMPILE_ARGS += -g2012   # accel_top.sv/ctrl_rf_rf.sv use logic/always_ff/always_comb

VERILOG_SOURCES += $(RDL_SV_DIR)/ctrl_rf_rf.sv

VERILOG_SOURCES += $(RTL_DIR)/sync_fifo.v
VERILOG_SOURCES += $(RTL_DIR)/pe.v
VERILOG_SOURCES += $(RTL_DIR)/pe_array.v
VERILOG_SOURCES += $(RTL_DIR)/weight_loader.v
VERILOG_SOURCES += $(RTL_DIR)/weight_ctrl.v
VERILOG_SOURCES += $(RTL_DIR)/input_dispatch.v
VERILOG_SOURCES += $(RTL_DIR)/input_buffer_32_bank.v
VERILOG_SOURCES += $(RTL_DIR)/output_loader.v
VERILOG_SOURCES += $(RTL_DIR)/output_buffer_32_bank.v
VERILOG_SOURCES += $(RTL_DIR)/accel_top.sv

# simulation-only models - see header note above
VERILOG_SOURCES += $(MODEL_DIR)/sync_fifo_w8_d32.v
VERILOG_SOURCES += $(MODEL_DIR)/buffer_ram_models.v

TOPLEVEL = accel_top
MODULE   = test_accel_top
WAVES = 1

export PYTHONPATH := $(PWD):$(PYTHONPATH)

include $(shell cocotb-config --makefiles)/Makefile.sim

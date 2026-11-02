from systemrdl import RDLCompiler
from peakrdl.verilog import VerilogExporter

rdlc = RDLCompiler()
rdlc.compile_file("ctrl_reg.rdl")
root = rdlc.elaborate()

exporter = VerilogExporter()
exporter.export(root, "ctrl_reg.sv", signal_overrides={})

# PeakRDL-verilog emits a field's hardware write and its software
# read-to-clear as two plain `if`s, so when both land in the same cycle the
# clear wins - and it ignores SystemRDL's `precedence` property. For
# STATUS.done that loses the completion event whenever a STATUS poll is
# accepted in the done pulse's cycle, and every later poll then hangs
# (test_done_set_wins_over_status_read). Patch the output so the hardware
# set wins.
RF = "ctrl_reg.sv/ctrl_rf_rf.sv"
OLD = '''        // Software read
        if (STATUS_sw_rd) begin
            STATUS_done_q <= '0;
        end'''
NEW = '''        // Software read
        // NB patched by rdl/python_gen.py: `else` so a hardware set in the
        // same cycle as a read-to-clear wins and the done event isn't lost
        // (PeakRDL emits a plain `if`, letting the clear win)
        else if (STATUS_sw_rd) begin
            STATUS_done_q <= '0;
        end'''
with open(RF) as f:
    src = f.read()
if src.count(OLD) != 1:
    raise SystemExit("python_gen.py: STATUS.done read-to-clear block not found in "
                     + RF + " - the generator output changed; re-check this patch")
with open(RF, "w") as f:
    f.write(src.replace(OLD, NEW))

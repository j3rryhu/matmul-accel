from systemrdl import RDLCompiler
from peakrdl.verilog import VerilogExporter

rdlc = RDLCompiler()
rdlc.compile_file("ctrl_reg.rdl")
root = rdlc.elaborate()

exporter = VerilogExporter()
exporter.export(root, "ctrl_reg.sv", signal_overrides={})

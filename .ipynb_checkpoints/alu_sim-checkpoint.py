from enum import Enum
import os
import cocotb
from cocotb.clock import Clock
from cocotb.runner import get_runner
from cocotb.triggers import RisingEdge

# Directory where the ALU simulation files are located
alu_sim_dir = os.path.abspath(os.path.join('.', 'alu_sim_dir'))

# Enum for ALU operations
class Funct3(Enum):
    ADD = 0
    SLL = 1
    SLT = 2
    SLTU = 3
    XOR = 4
    SRL = 5
    SRA = 5
    OR = 6
    AND = 7

# Helper function to apply inputs and wait for clock edge
async def apply_inputs(dut, funct3, s1, s2):
    dut.funct3.value = funct3.value
    dut.s1.value = s1
    dut.s2.value = s2
    await RisingEdge(dut.clk)

# Test for ADD operation
@cocotb.test()
async def test_add_operation(dut):
    clock = Clock(dut.clk, period=10, units='ns')
    cocotb.start_soon(clock.start(start_high=False))

    # Apply inputs for ADD (10 + 20)
    await apply_inputs(dut, Funct3.ADD, 10, 20)
    await RisingEdge(dut.clk)

    # Check result
    expected_result = 10 + 20
    assert dut.d.value == expected_result, f"ADD operation failed: Expected {expected_result}, got {int(dut.d.value)}"
    print(f"ADD operation passed: 10 + 20 = {int(dut.d.value)}")

# Test for XOR operation
@cocotb.test()
async def test_xor_operation(dut):
    clock = Clock(dut.clk, period=10, units='ns')
    cocotb.start_soon(clock.start(start_high=False))

    # Apply inputs for XOR (30 ^ 30)
    await apply_inputs(dut, Funct3.XOR, 30, 30)
    await RisingEdge(dut.clk)

    # Check result
    expected_result = 30 ^ 30
    assert dut.d.value == expected_result, f"XOR operation failed: Expected {expected_result}, got {int(dut.d.value)}"
    print(f"XOR operation passed: 30 ^ 30 = {int(dut.d.value)}")

# Test for NOT operation (simulated with XOR)
@cocotb.test()
async def test_not_operation(dut):
    clock = Clock(dut.clk, period=10, units='ns')
    cocotb.start_soon(clock.start(start_high=False))

    # Simulated NOT: s1 = 7, s2 = all ones (0xFFFFFFFF)
    const_ones = 2**32 - 1
    await apply_inputs(dut, Funct3.XOR, 7, const_ones)
    await RisingEdge(dut.clk)

    # Check result for NOT (~7)
    expected_result = 7 ^ const_ones
    assert dut.d.value == expected_result, f"NOT operation failed: Expected {expected_result}, got {int(dut.d.value)}"
    print(f"NOT operation passed: ~7 = {int(dut.d.value)}")

# Test for SUB operation (assuming we add functionality later)
# @cocotb.test()
async def test_sub_operation(dut):
    clock = Clock(dut.clk, period=10, units='ns')
    cocotb.start_soon(clock.start(start_high=False))

    # Apply inputs for SUB (20 - 10)
    await apply_inputs(dut, Funct3.ADD, 20, -10)
    await RisingEdge(dut.clk)

    # Check result for subtraction (20 - 10)
    expected_result = 20 - 10
    assert dut.d.value == expected_result, f"SUB operation failed: Expected {expected_result}, got {int(dut.d.value)}"
    print(f"SUB operation passed: 20 - 10 = {int(dut.d.value)}")

# Function to run cocotb tests with verilator
def test_via_cocotb():
    """
    Main entry point for cocotb
    """
    verilog_sources = [os.path.abspath(os.path.join('.', 'alu.v'))]
    runner = get_runner("verilator")
    runner.build(
        verilog_sources=verilog_sources,
        vhdl_sources=[],
        hdl_toplevel="RISCALU",
        build_args=["--threads", "2"],
        build_dir=alu_sim_dir,
    )
    runner.test(hdl_toplevel="RISCALU", test_module="alu_sim")

# Entry point to start tests
if __name__ == '__main__':
    test_via_cocotb()

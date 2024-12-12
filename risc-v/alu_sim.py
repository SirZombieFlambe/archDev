from enum import Enum
import os

import cocotb

from cocotb.clock import Clock
from cocotb.runner import get_runner
from cocotb.triggers import RisingEdge

alu_sim_dir = os.path.abspath(os.path.join('.', 'alu_sim_dir'))

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

def perform_not(dut, s1) -> None:
    """
    ~
    :param dut: DUT object from cocotb
    :param s1: Value to perform the boolean not
    :return: None
    """

    raise NotImplementedError("Implement not")

def perform_negate(dut, s1) -> None:
    """
    Perform the two's complement.

    :param dut: DUT object from cocotb
    :param s1: Value to perform the two's complement negation on.
    :return: None
    """
    raise NotImplementedError("Implement negate")

def perform_sub(dut, s1, s2) -> None:
    """
    sub rd, rs1, rs2

    :param dut: Dut object from cocotb
    :param s1: First value as described in R sub
    :param s2: Second value as described in R sub
    :return: None
    """
    raise NotImplementedError("Implement sub")

def set_gte(dut, s1, s2):
    """
    In the same format as slt rd, rs1, rs2 perform the operation to set the output LSB bit to rs1 >= rs2.

    :param dut: DUT object from cocotb
    :param s1: First value as described in R slt
    :param s2: Second value as described in R slt
    :return:
    """


async def apply_inputs(dut, funct3, s1, s2):
    dut.funct3.value = funct3.value
    dut.s1.value = s1
    dut.s2.value = s2
    await RisingEdge(dut.clk)
    await RisingEdge(dut.clk)


@cocotb.test()
async def run_alu_sim(dut):
    clock = Clock(dut.clk, period=10, units='ns') # This assigns the clock into the ALU
    cocotb.start_soon(clock.start(start_high=False))


    # XOR
    dut.funct3.value = Funct3.XOR.value
    dut.s1.value = 10
    dut.s2.value = 20

    await RisingEdge(dut.clk)
    assert 10 ^ 20 == int(dut.d.value)
    print(int(dut.d.value))



@cocotb.test()
async def run_alu_sim2(dut):
    """Test ADD operation"""
    clock = Clock(dut.clk, period=10, units='ns')  # This assigns the clock into the ALU
    cocotb.start_soon(clock.start(start_high=False))

    s1 = 10
    s2 = 20

    # Apply inputs for ADD (10 + 20)
    actual_result = await add(dut, s1, s2)

    # Check result
    expected_result = s1 + s2
    assert actual_result == expected_result, f"ADD failed: Expected {expected_result}, got {actual_result}"
    print(f"ADD operation passed: {s1} + {s2} = {int(dut.d.value)}")

async def add(dut, s1, s2):
    await apply_inputs(dut, Funct3.ADD, s1, s2)  # Use Funct3.ADD instead of 0
    return dut.d.value


# Test for XOR operation
@cocotb.test()
async def test_xor_operation(dut):
    """Test XOR operation"""
    clock = Clock(dut.clk, period=10, units='ns')  # This assigns the clock into the ALU
    cocotb.start_soon(clock.start(start_high=False))

    s1 = 30
    s2 = 30

    # Apply inputs for XOR (30 ^ 30)
    actual_result = await xor(dut, s1, s2)
    # Check result
    expected_result = s1 ^ s2

    assert actual_result == expected_result, f"XOR failed: Expected {expected_result}, got {actual_result}"
    print(f"XOR operation passed: {s1} ^ {s2} = {int(dut.d.value)}")


async def xor(dut, s1, s2):
    # Apply inputs for XOR (30 ^ 30)
    await apply_inputs(dut, Funct3.XOR, s1, s2)  # Use Funct3.XOR for XOR operation
    return dut.d.value

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

if __name__ == '__main__':
    test_via_cocotb()

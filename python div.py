async def pythonDiv(dividend, divisor):
    quotient = 0
    remainder = dividend

    # Left shift divisor until it's just below the dividend
    shift_count = 0
    while dividend >= divisor:
        divisor <<= 1
        shift_count += 1

    # Right shift back to start the division
    divisor >>= 1

    for _ in range(shift_count):

        if remainder >= divisor:
            remainder -= divisor
            quotient = (quotient << 1) | 1
        else:
            quotient = quotient << 1

        divisor >>= 1

    return quotient, remainder




async def mul(dut, multiplicand, multiplier):
    product = 0

    while multiplier > 0:
        # If the current LSB of the multiplier is 1, add the multiplicand to the product
        if multiplier & 1 == 1:
            product += multiplicand

        # Shift the multiplicand left by 1 (multiplicand *= 2)
        multiplicand <<= 1

        # Shift the multiplier right by 1 (divide multiplier by 2)
        multiplier >>= 1

    return product




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

@cocotb.test()
async def run_alu_sim(dut):
    clock = Clock(dut.clk, period=10, units='ns') # This assigns the clock into the ALU
    cocotb.start_soon(clock.start(start_high=False))

    # ADD
    dut.funct3.value = Funct3.ADD.value
    dut.s1.value = 10
    dut.s2.value = 20

    await RisingEdge(dut.clk)

    # XOR
    dut.funct3.value = Funct3.XOR.value
    dut.s1.value = dut.d.value
    dut.s2.value = dut.d.value

    await RisingEdge(dut.clk)
    assert 10 + 20 == dut.d.value
    print(int(dut.d.value))

    # Perform NOT
    dut.funct3.value = Funct3.XOR.value
    const_ones = 2**32 - 1
    dut.s1.value = 7
    dut.s2.value = const_ones

    await RisingEdge(dut.clk)
    assert 0 == dut.d.value
    print(int(dut.d.value))

    await RisingEdge(dut.clk)
    assert (7 ^ const_ones) == dut.d.value
    print(dut.d.value)

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

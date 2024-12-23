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

# Apply Functions
async def apply(dut, funct3, s1, s2):
    dut.funct3.value = funct3
    dut.s1.value = s1
    dut.s2.value = s2
    await RisingEdge(dut.clk)

    #I have no idea why this second RisingEdge is necessary but without it my tests fail
    await RisingEdge(dut.clk)

# Part 1 (Puzzles)

#Not Puzzle
async def perform_not(dut, s1) -> None:
    # Wait for clock cycle
    await RisingEdge(dut.clk)

    # Group of 32 1's
    all_ones = (2 ** 32) - 1

    # XOR with all 1's and input
    await apply(dut, Funct3.XOR.value, s1, all_ones)

    # Debugging output
    print(f"Input: {s1:#010x}, all_ones: {all_ones:#010x}, Result: {dut.d.value.integer:#010x}")

    return dut.d.value

#Not Puzzle Test
@cocotb.test()
async def test_not(dut):
    """Test the NOT operation."""
    # Initialize the clock
    clock = Clock(dut.clk, period=10, units='ns')
    cocotb.start_soon(clock.start(start_high=False))
    all_ones = 2**32-1

    test_vals = [0,2**32-1,24,63,-4]

    for val in test_vals:

        # Input value and expected output for NOT operation
        expected_output = (~val) & all_ones

        # Perform the NOT operation
        result = await perform_not(dut, val)

        # Validate the result
        assert result == expected_output, f"NOT operation failed: got  {result}, expected {expected_output}"

#Add Puzzle
async def add(dut, s1, s2):
    #Wait for clock, add, return value
    await RisingEdge(dut.clk)
    await apply(dut,Funct3.ADD.value,s1,s2)
    return dut.d.value

#Add Puzzle Test
@cocotb.test()
async def test_add(dut):
    # Initialize the clock
    clock = Clock(dut.clk, period=10, units='ns')
    cocotb.start_soon(clock.start(start_high=False))
    test_vals = [[1,2],[1,-1],[0,-5],[32,37]]

    for arr in test_vals:
        result = await add(dut,arr[0],arr[1])
        result_int = int(result)
        if result_int >= 2**31:
            result_int-=2**32
        assert result_int == arr[0]+arr[1], f"ADD operation failed: got {result}, expected {arr[0]+arr[1]}"

#Negate Puzzle
async def perform_negate(dut, s1) -> None:
    await RisingEdge(dut.clk)

    #Flip bits, add 1
    flipped = await perform_not(dut, s1)
    await add(dut,flipped,1)

    return dut.d.value

@cocotb.test()
async def test_negate(dut):
    clock = Clock(dut.clk,period=10, units='ns')
    cocotb.start_soon(clock.start(start_high=False))

    test_vals = [1,-1,6,2**31-1]

    for val in test_vals:
        result = await perform_negate(dut,val)
        result_int = int(result)
        if result_int >= 2**31:
            result_int-=2**32
        
        should_zero = result_int + val
        assert should_zero == 0, f"Negated {val} is {result_int}"


async def perform_sub(dut, s1, s2) -> None:
    await RisingEdge(dut.clk)

    # Negative s2
    await perform_negate(dut,s2)
    negated_s2 = dut.d.value

    # s1 + (-s2)
    await add(dut,s1,negated_s2)

    return dut.d.value

@cocotb.test()
async def test_sub(dut):
    clock = Clock(dut.clk,period=10, units='ns')
    cocotb.start_soon(clock.start(start_high=False))

    test_vals = [[3,2],[2,3],[5,5],[8,2**10],[2**31,4]]
    for arr in test_vals:
        result = await perform_sub(dut,arr[0],arr[1])
        result_int = int(result)
        if result_int >= 2**31:
            result_int-=2**32
        expected = arr[0]-arr[1]
        assert result_int == expected, f"Expected {expected},got {result_int}"


async def set_gte(dut, s1, s2) -> None:
    await RisingEdge(dut.clk)

    # Computes if s2 < s1 (same thing)
    await apply(dut,Funct3.SLT.value,s2,s1)

    return int(dut.d.value)

@cocotb.test()
async def test_gte(dut):
    clock = Clock(dut.clk,period=10, units='ns')
    cocotb.start_soon(clock.start(start_high=False))

    test_vals = [[1,-1],[5,4],[0,-9]]
    for arr in test_vals:
        expected = 0
        if arr[0] >= arr[1]:
            expected = 1

        result = await set_gte(dut,arr[0],arr[1])
        result_int = int(result)
        if result_int >= 2**31:
            result_int-=2**32

        assert result_int == expected, f"Expected {expected}, got {result_int}"

# Returns 1 if s1 == 0, otherwise 0
async def logical_not(dut, s1):
    await apply(dut, Funct3.SRL.value, s1, 31)
    temp = dut.d.value
    await perform_not(dut, s1)
    temp2 = await add(dut, dut.d.value, 1)
    await apply_inputs(dut, Funct3.SRL, temp2, 31)
    temp2 = await or_operation(dut, temp, dut.d.value)
    return await add(dut, temp2, 1)

# Returns 1 if equal, 0 otherwise
async def eq(dut,s1,s2):
    diff = await perform_sub(dut,s1,s2)
    return int(await logical_not(dut, diff))

# Multiply Function
async def perform_multiply(dut, s1, s2) -> None:
    prod = 0

    loop = await set_gte(dut,s2,1)
    while loop:
        await apply(dut,Funct3.AND.value,s1,1)
        if_enough = await eq(dut,dut.d.value,1)
        if if_enough:
            prod = await add(dut,prod,s1)
        
        await apply(dut,Funct3.SLL.value,s1,1)
        s1 = dut.d.value

        await apply(dut,Funct3.SRL.value,s2,1)
        s2 = dut.d.value

        loop = await set_gte(dut,s2,1)

    return prod

@cocotb.test()
async def test_multiply(dut):
    clock = Clock(dut.clk,period=10, units='ns')
    cocotb.start_soon(clock.start(start_high=False))

    test_vals = [[2**31,0],[1,2],[6,-1]]
    for arr in test_vals:
        expected = arr[0]*arr[1]
        result = await perform_multiply(dut,arr[0],arr[1])
        result_int = int(result)
        if result_int >= 2**31:
            result_int-=2**32

        assert result_int == expected, f"Expected {expected}, got {result_int}. Inputs: {arr[0]},{arr[1]}"


async def perform_division(dut, s1, s2):
    qtnt = 0
    rem = s1

    raise NotImplementedError("Implement div")

@cocotb.test()
async def run_alu_sim(dut):
    clock = Clock(dut.clk, period=10, units='ns') # This assigns the clock into the ALU
    cocotb.start_soon(clock.start(start_high=False))


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

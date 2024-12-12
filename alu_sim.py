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
    await RisingEdge(dut.clk)


# Test for ADD operation
@cocotb.test()
async def test_add_operation(dut):
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


# Test for NOT operation (simulated with XOR)
@cocotb.test()
async def test_not_operation(dut):
    """Test NOT operation"""
    clock = Clock(dut.clk, period=10, units='ns')  # This assigns the clock into the ALU
    cocotb.start_soon(clock.start(start_high=False))

    # Inputs for the NOT operation
    s1 = 7
    const_ones = 2 ** 32 - 1

    # Apply the NOT operation
    actual_result = await not_operation(dut, s1)

    # Expected result is the bitwise negation of s1 (in 32-bit unsigned format)
    expected_result = (~s1) & const_ones  # Apply mask to simulate 32-bit unsigned behavior

    assert actual_result == expected_result, f"NOT operation failed: Expected {bin(expected_result)}, got {actual_result}"

    # Print the result for confirmation
    print(f"NOT operation passed: ~{s1} = {actual_result}")


async def not_operation(dut, s1):
    """Simulate a NOT operation using XOR with all ones (0xFFFFFFFF)"""
    const_ones = 2 ** 32 - 1
    await apply_inputs(dut, Funct3.XOR, s1, const_ones)
    return dut.d.value


# Test for SUB operation (assuming we add functionality later)
@cocotb.test()
async def test_sub_operation(dut):
    """Test SUB operation"""
    clock = Clock(dut.clk, period=10, units='ns')  # This assigns the clock into the ALU
    cocotb.start_soon(clock.start(start_high=False))

    s1 = 5
    s2 = 10

    # Apply inputs for SUB (20 - 10)
    actual_result = await sub(dut, s1, s2)

    # Check result s1 - s2
    expected = await int_to_bin(dut, s1 - s2)

    assert actual_result == expected, f"SUB failed: Expected {expected}, got {actual_result}"
    # Print the result for confirmation
    print(f"SUB operation passed: {s1} - {s2} = {int(dut.d.value)}")


async def sub(dut, s1, s2):
    """Subtracts s2 from s1 by converting s2 to its two's complement and adding."""
    await not_operation(dut, s2)

    await apply_inputs(dut, Funct3.ADD, dut.d.value, 1)
    negated_s2 = dut.d.value

    await apply_inputs(dut, Funct3.ADD, s1, negated_s2)

    # Await the final result
    return dut.d.value


async def int_to_bin(dut, num, ):
    """Converts an integer to a binary string (signed)."""

    return await add(dut, num, 0) if await gte(dut, num, 0) else await twos_complement(dut, await abs_value(dut, num))

async def twos_complement(dut, s1):
    # Flip the bits and add 1 for two's complement
    s1 = await xor(dut, s1, 0xFFFFFFFF)  # Flip bits
    s1 = await add(dut, s1, 1)           # Add 1 for two's complement
    return s1

@cocotb.test()
async def test_or_operation(dut):
    """Test gte operation"""
    clock = Clock(dut.clk, period=10, units='ns')  # This assigns the clock into the ALU
    cocotb.start_soon(clock.start(start_high=False))

    # Apply inputs for OR operation
    s1 = 0b10101010  # Example value for s1
    s2 = 0b11001100  # Example value for s2
    expected_result = s1 | s2  # Expected result of OR operation

    actual_value = await or_operation(dut, s1, s2)

    # Check result
    assert actual_value == expected_result, f"OR operation failed: Expected {expected_result}, got {int(dut.d.value)}"
    print(f"OR operation passed: {bin(s1)} | {bin(s2)} = {bin(dut.d.value)}")


async def or_operation(dut, s1, s2):
    # Apply the OR operation
    await apply_inputs(dut, Funct3.OR, s1, s2)
    return dut.d.value


@cocotb.test()
async def test_gt_operation(dut):
    """Test greater-than operation"""
    clock = Clock(dut.clk, period=10, units='ns')  # This assigns the clock into the ALU
    cocotb.start_soon(clock.start(start_high=False))

    # Example values
    s1 = 15
    s2 = -25
    expected_result = 1 if s1 > s2 else 0  # GT result should be 1 if a is greater than b

    actual_result = await gt(dut, s1, s2)

    # Check result
    # SLT returns 1 if s1 < s2, so we invert the logic
    assert actual_result == expected_result, f"GT operation failed: Expected {expected_result}, got {int(dut.d.value)}"
    print(f"GT operation passed: {s1} > {s2} = {expected_result}")


async def gt(dut, s1, s2):
    # Apply inputs for greater-than operation
    await apply_inputs(dut, Funct3.SLT, s2, s1)  # Use SLT and flip inputs
    return int(dut.d.value)


@cocotb.test()
async def test_equ_operation(dut):
    """Test equal-to operation without using == operator"""
    clock = Clock(dut.clk, period=10, units='ns')  # This assigns the clock into the ALU
    cocotb.start_soon(clock.start(start_high=False))

    # Example values for equality test
    s1 = 15
    s2 = -12

    # Expected result: 1 if s1 == s2, otherwise 0
    expected_result = 1 if s1 == s2 else 0

    actual_result = await equ_operation(dut, s1, s2)

    # Compare the result to expected without using == operator
    assert actual_result == expected_result, f"Equality operation failed: Expected {expected_result}, got {actual_result}"

    # Print the final result for debugging
    print(f"EQ operation passed: {s1} == {s2} = {expected_result}")


async def equ_operation(dut, s1, s2):
    diff = await sub(dut, s1, s2)

    return int(await logical_not(dut, diff))

@cocotb.test()
async def test_not_equ_operation(dut):
    """Test equal-to operation without using == operator"""
    clock = Clock(dut.clk, period=10, units='ns')  # This assigns the clock into the ALU
    cocotb.start_soon(clock.start(start_high=False))

    # Example values for equality test
    s1 = 15
    s2 = 17

    # Expected result: 1 if s1 == s2, otherwise 0
    expected_result = 1 if s1 != s2 else 0

    actual_result = await not_equ_operation(dut, s1, s2)

    # Compare the result to expected without using == operator
    assert actual_result == expected_result, f"Not equality operation failed: Expected {expected_result}, got {actual_result}"

    # Print the final result for debugging
    print(f"Not_EQ operation passed: {s1} != {s2} = {expected_result}")

async def not_equ_operation(dut, s1, s2):
    """Returns True if value1 is not equal to value2."""
    # Check if value1 and value2 are equal
    is_equal = await equ_operation(dut, s1, s2)

    # Invert the result to get not equal
    is_not_equal = await logical_not(dut, is_equal)  # 1 if not equal, 0 if equal
    return is_not_equal



@cocotb.test()
async def test_not_logical_operation(dut):
    """Test not operation"""
    clock = Clock(dut.clk, period=10, units='ns')  # This assigns the clock into the ALU
    cocotb.start_soon(clock.start(start_high=False))

    # Example values for equality test
    s1 = 15

    expected_result = not s1

    actual_result = await logical_not(dut, s1)

    # Compare the result to expected without using == operator
    assert actual_result == expected_result, f"Equality operation failed: Expected {expected_result}, got {actual_result}"

    # Print the final result for debugging
    print(f"NOT operation passed: not {s1} = {expected_result}")


async def logical_not(dut, s1):
    """
    This function computes the logical NOT operation for the given input `s1`.
    The result is 1 if s1 == 0, otherwise 0.

    This is achieved by breaking the operation down into bitwise shifts, NOT operations,
    and checking whether `s1` is zero or non-zero using bitwise operations.
    """

    # Right shift the value of s1 by 31 bits. This will get the sign bit in case s1 is non-zero.
    # If s1 is non-zero, `temp` will contain a value of 0 or 1, depending on the most significant bit.
    await apply_inputs(dut, Funct3.SRL, s1, 31)
    temp = dut.d.value

    # Perform a bitwise NOT operation on s1 to flip all bits.
    # This will invert s1, turning non-zero values into negative values.
    await not_operation(dut, s1)

    # Add 1 to the result of the NOT operation to form two's complement, which will help in the next step.
    # This ensures that we are handling the zero and non-zero values correctly.
    temp2 = await add(dut, dut.d.value, 1)

    # Right shift the result by 31 bits to isolate the sign bit. This helps normalize the result to either 0 or 1.
    await apply_inputs(dut, Funct3.SRL, temp2, 31)

    # OR the shifted result (`temp`) with the right-shifted value from the last step.
    # This will yield `1` if `s1` was non-zero, and `0` if `s1` was zero.
    temp2 = await or_operation(dut, temp, dut.d.value)

    # Finally, add 1 to the result, effectively flipping it to complete the logical NOT operation.
    # The final result will be 1 if `s1` was zero, otherwise 0.
    return await add(dut, temp2, 1)


@cocotb.test()
async def test_gte_operation(dut):
    """Test gte operation"""
    clock = Clock(dut.clk, period=10, units='ns')  # This assigns the clock into the ALU
    cocotb.start_soon(clock.start(start_high=False))

    s1 = 20
    s2 = -30

    # Expected result: 1 if s1 >= s2, otherwise 0
    expected_result = 1 if s1 >= s2 else 0

    actual_result = await gte(dut, s1, s2)

    # Compare the result to expected without using == operator
    assert actual_result == expected_result, f"Greater than or equal to operation failed: Expected {expected_result}, got {actual_result}"

    # Print the final result for debugging
    print(f"GTE operation passed: {s1} >= {s2} = {expected_result}")


async def gte(dut, s1, s2):
    temp = await gt(dut, s1, s2)

    equ_value = await equ_operation(dut, s1, s2)

    await or_operation(dut, temp, equ_value)

    return int(dut.d.value)


@cocotb.test()
async def test_mul(dut):
    """Test multiplication operation"""
    clock = Clock(dut.clk, period=10, units='ns')  # This assigns the clock into the ALU
    cocotb.start_soon(clock.start(start_high=False))

    s1 = -7
    s2 = 3

    # Expected result: s1 * s2
    expected_result = await int_to_bin(dut, s1 * s2)

    product = await mul(dut, s1, s2)

    assert expected_result == product, f"Multiplication operation failed: Expected {expected_result}, got {product}"

    # Print the final result for debugging
    print(f"Multiplication operation passed: {s1} * {s2} = {expected_result}")


async def mul(dut, multiplicand, multiplier):
    product = 0

    loop_value = await gte(dut, multiplier, 1)
    while loop_value:
        # If the current LSB of the multiplier is 1, add the multiplicand to the product

        # Apply AND to the multiplier to check the least significant bit
        await apply_inputs(dut, Funct3.AND, multiplier, 1)
        if_value = await equ_operation(dut, dut.d.value, 1)

        if if_value:
            product = await add(dut, product, multiplicand)

        await apply_inputs(dut, Funct3.SLL, multiplicand, 1)
        multiplicand = dut.d.value  # Mask to 32 bits

        await apply_inputs(dut, Funct3.SRL, multiplier, 1)
        multiplier = dut.d.value  # Mask to 32 bits

        loop_value = await gte(dut, multiplier, 1)
    return product


@cocotb.test()
async def test_div(dut):
    """Test division operation"""
    clock = Clock(dut.clk, period=10, units='ns')  # This assigns the clock into the ALU
    cocotb.start_soon(clock.start(start_high=False))

    # Example values
    s1 = 15  # Dividend
    s2 = 5  # Divisor

    # Expected quotient and remainder
    expected_quotient = await int_to_bin(dut, s1 // s2)
    expected_remainder = await int_to_bin(dut, s1 % s2)

    # Perform the division
    quotient, remainder = await div(dut, s1, s2)

    # Assertions to check quotient and remainder
    assert quotient == expected_quotient, f"Division operation failed: Expected quotient {expected_quotient}, got {quotient}"
    assert remainder == expected_remainder, f"Division operation failed: Expected remainder {expected_remainder}, got {remainder}"

    # Print results for confirmation
    print(f"Division operation passed: {s1} / {s2} = {quotient} with remainder {remainder}")


async def div(dut, dividend, divisor):
    # Step 1: Determine the sign of the result
    is_signed_dividend = await gte(dut, 2, dividend)
    is_signed_divisor = await gte(dut, 2, divisor)

    negative_result = await not_equ_operation(dut, is_signed_dividend, is_signed_divisor)

    # Step 2: Work  absolute values for the division
    dividend = await abs_value(dut, dividend)
    divisor = await abs_value(dut, divisor)

    quotient = 0
    remainder = await add(dut, dividend, 0)

    # Left shift divisor until it's just below the dividend
    shift_count = 0
    loop_value = await gte(dut, dividend, divisor)

    # Align the divisor with the dividend by shifting left
    while loop_value:
        await apply_inputs(dut, Funct3.SLL, divisor, 1)
        divisor = dut.d.value
        shift_count = await add(dut, shift_count, 1)
        loop_value = await gte(dut, dividend, divisor)

    # Right shift back to start the division
    await apply_inputs(dut, Funct3.SRL, divisor, 1)
    divisor = dut.d.value

    # Division loop
    for _ in range(shift_count):
        if await gte(dut, remainder, divisor):
            remainder = await sub(dut, remainder, divisor)
            await apply_inputs(dut, Funct3.SLL, quotient, 1)
            quotient = await or_operation(dut, dut.d.value, 1)
        else:
            await apply_inputs(dut, Funct3.SLL, quotient, 1)
            quotient = dut.d.value

        await apply_inputs(dut, Funct3.SRL, divisor, 1)
        divisor = dut.d.value

    if negative_result:
        # Convert quotient to negative using two's complement
        quotient = await twos_complement(dut, quotient)

    return quotient, remainder


async def abs_value(dut, value):
    """Returns the absolute value of the given number."""
    # Check if the value is greater than or equal to 0
    if await gte(dut, value, 0):
        return value  # If non-negative, return the original value
    else:
        # If negative, convert to positive using two's complement
        return await twos_complement(dut, value)


@cocotb.test()
async def test_all_operations(dut):
    """Test all operations with multiple values"""
    clock = Clock(dut.clk, period=10, units='ns')  # This assigns the clock into the ALU
    cocotb.start_soon(clock.start(start_high=False))

    # Define test cases for each operation
    add_cases = [(10, 20), (5, -15), (0, 0)]
    sub_cases = [(20, 10), (5, -10), (0, 0)]
    xor_cases = [(30, 30), (0b1010, 0b1100), (0, 0), (-15, -15)]
    or_cases = [(0b10101010, 0b11001100), (5, 10), (0, 1), (-5, -6)]
    not_cases = [7, 0xFFFFFFFF, 0, -9]
    gt_cases = [(15, -25), (5, 5), (-1, 1)]
    gte_cases = [(20, -30), (5, 5), (0, 1)]
    mul_cases = [(15, 13), (-7, 3), (0, 100)]
    div_cases = [(100, 7), (15, 5), (30, -6), (-30, -5), (-30, 6)]
    equ_cases = [(15, 15), (5, 10), (0, 0)]
    logical_not_cases = [1, 0, -5]

    # Run ADD test cases
    for s1, s2 in add_cases:
        actual_result = await add(dut, s1, s2)
        expected_result = await int_to_bin(dut, s1 + s2)
        assert actual_result == expected_result, f"ADD failed: {s1} + {s2} = {actual_result}, expected {expected_result}"
        print(f"ADD operation passed: {s1} + {s2} = {actual_result}")
    print()
    # Run SUB test cases
    for s1, s2 in sub_cases:
        actual_result = await sub(dut, s1, s2)
        expected_result = await int_to_bin(dut, s1 - s2)
        assert actual_result == expected_result, f"SUB failed: {s1} - {s2} = {actual_result}, expected {expected_result}"
        print(f"SUB operation passed: {s1} - {s2} = {actual_result}")
    print()
    # Run XOR test cases
    for s1, s2 in xor_cases:
        actual_result = await xor(dut, s1, s2)
        expected_result = await int_to_bin(dut, s1 ^ s2)
        assert actual_result == expected_result, f"XOR failed: {s1} ^ {s2} = {actual_result}, expected {expected_result}"
        print(f"XOR operation passed: {s1} ^ {s2} = {actual_result}")
    print()
    # Run OR test cases
    for s1, s2 in or_cases:
        actual_result = await or_operation(dut, s1, s2)
        expected_result = await int_to_bin(dut, s1 | s2)
        assert actual_result == expected_result, f"OR failed: {s1} | {s2} = {actual_result}, expected {expected_result}"
        print(f"OR operation passed: {s1} | {s2} = {actual_result}")
    print()
    # Run NOT test cases
    const_ones = 2 ** 32 - 1
    for s1 in not_cases:
        actual_result = await not_operation(dut, s1)
        expected_result = (~s1) & const_ones  # Apply mask to simulate 32-bit unsigned behavior
        assert actual_result == expected_result, f"NOT failed: ~{s1} = {actual_result}, expected {expected_result}"
        print(f"NOT operation passed: ~{s1} = {actual_result}")
    print()
    # Run GT test cases
    for s1, s2 in gt_cases:
        actual_result = await gt(dut, s1, s2)
        expected_result = 1 if s1 > s2 else 0
        assert actual_result == expected_result, f"GT failed: {s1} > {s2} = {actual_result}, expected {expected_result}"
        print(f"GT operation passed: {s1} > {s2} = {actual_result}")
    print()
    # Run GTE test cases
    for s1, s2 in gte_cases:
        actual_result = await gte(dut, s1, s2)
        expected_result = 1 if s1 >= s2 else 0
        assert actual_result == expected_result, f"GTE failed: {s1} >= {s2} = {actual_result}, expected {expected_result}"
        print(f"GTE operation passed: {s1} >= {s2} = {actual_result}")
    print()
    # Run MUL test cases
    for s1, s2 in mul_cases:
        actual_result = await mul(dut, s1, s2)
        expected_result = await int_to_bin(dut, s1 * s2)
        assert actual_result == expected_result, f"MUL failed: {s1} * {s2} = {actual_result}, expected {expected_result}"
        print(f"MUL operation passed: {s1} * {s2} = {actual_result}")
    print()
    # Run DIV test cases
    for s1, s2 in div_cases:
        quotient, remainder = await div(dut, s1, s2)
        expected_quotient = await int_to_bin(dut, s1 // s2)
        expected_remainder = await int_to_bin(dut, s1 % s2)
        assert quotient == expected_quotient, f"Division operation failed: Expected quotient {expected_quotient}, got {quotient}"
        assert remainder == expected_remainder, f"Division operation failed: Expected remainder {expected_remainder}, got {remainder}"
        print(f"Division operation passed: {s1} / {s2} = {quotient} with remainder {remainder}")
    print()
    # Run EQU test cases
    for s1, s2 in equ_cases:
        actual_result = await equ_operation(dut, s1, s2)
        expected_result = 1 if s1 == s2 else 0
        assert actual_result == expected_result, f"EQU failed: {s1} == {s2} = {actual_result}, expected {expected_result}"
        print(f"EQU operation passed: {s1} == {s2} = {actual_result}")
    print()
    # Run Logical NOT test cases
    for s1 in logical_not_cases:
        actual_result = await logical_not(dut, s1)
        expected_result = not s1
        assert actual_result == expected_result, f"Logical NOT failed: not {s1} = {actual_result}, expected {expected_result}"
        print(f"NOT operation passed: not {s1} = {expected_result}")
    print()
    # Print a summary message if all assertions pass
    print("All operations passed with multiple test values.")


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

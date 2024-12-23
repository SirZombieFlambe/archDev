from decimal import Decimal, getcontext

# Set the precision high enough to avoid rounding issues
getcontext().prec = 50  # Adjust the precision as needed

# Function to convert a fractional decimal to binary using Decimal for high precision
def decimal_to_binary_fraction(decimal_fraction, steps=52):
    """
    Convert a fractional decimal number to binary and show the intermediate steps.

    Args:
    - decimal_fraction (float or Decimal): The fractional decimal to convert.
    - steps (int): The number of binary digits to calculate.

    Returns:
    - steps_list (list): A list of strings showing the intermediate steps.
    - binary_result (str): The resulting binary representation as a string.
    """
    steps_list = []  # To store the step-by-step explanation
    value = Decimal(decimal_fraction)  # Ensure the input is a Decimal for precision
    binary_result = ""  # To store the binary representation

    # Iteratively convert the fraction to binary
    for _ in range(steps):
        value *= 2
        integer_part = int(value)
        fractional_part = value - integer_part

        # Record the step
        steps_list.append(f"{value:.10f} → Integer part = {integer_part}, Fractional part = {fractional_part:.10f}")
        binary_result += str(integer_part)

        # Stop if there's no fractional part left
        if fractional_part == 0:
            break

        value = fractional_part

    return steps_list, binary_result


# Function to convert the integer part to binary
def integer_to_binary(integer_part):
    """
    Convert the integer part of a decimal number to binary.

    Args:
    - integer_part (int): The integer part to convert.

    Returns:
    - binary_result (str): The binary representation of the integer part.
    """
    binary_result = ""
    steps = []
    value = integer_part

    while value > 0:
        remainder = value % 2
        binary_result = str(remainder) + binary_result
        steps.append(f"{value} ÷ 2 = {value // 2}, Remainder = {remainder}")
        value //= 2

    return steps, binary_result


# Input: Decimal number to convert
decimal_input = "3.14159265"
decimal_number = Decimal(decimal_input)

# Separate the integer and fractional parts
integer_part = int(decimal_number)
fractional_part = decimal_number - integer_part

# Convert the integer and fractional parts
integer_steps, integer_binary = integer_to_binary(integer_part)
fractional_steps, fractional_binary = decimal_to_binary_fraction(fractional_part)

# Display the steps and result
print("Integer Part Conversion Steps:")
for step in integer_steps:
    print(step)

print("\nFractional Part Conversion Steps:")
for step in fractional_steps:
    print(step)

# Combine the integer and fractional parts
print(f"\nBinary Representation: {integer_binary}.{fractional_binary}")


def binary_to_decimal(binary_string):
    """
    Convert a binary number to its decimal equivalent.

    Args:
    - binary_string (str): Binary number as a string (e.g., "110.101").

    Returns:
    - decimal_result (float): The decimal equivalent of the binary number.
    """
    if '.' in binary_string:
        integer_part, fractional_part = binary_string.split('.')
    else:
        integer_part, fractional_part = binary_string, ''

    # Convert integer part
    decimal_integer = 0
    for i, digit in enumerate(reversed(integer_part)):
        decimal_integer += int(digit) * (2 ** i)

    # Convert fractional part
    decimal_fraction = 0
    for i, digit in enumerate(fractional_part):
        decimal_fraction += int(digit) * (2 ** -(i + 1))

    # Combine the integer and fractional parts
    return decimal_integer + decimal_fraction


# Example usage
binary_input = "11.0010010000111111011010100111100100011010100111100001"  # Example binary number
decimal_result = binary_to_decimal(binary_input)
print(f"Binary: {binary_input} → Decimal: {decimal_result}")



# Function to convert hex to decimal and compare precision loss
import struct

def hex_to_float(hex_value, precision_bits):
    """
    Convert hex representation to floating-point number.

    Args:
    - hex_value (str): Hex string representing the number.
    - precision_bits (int): Bit precision of the floating-point representation.

    Returns:
    - float: The converted floating-point value.
    """
    if precision_bits == 32:
        return struct.unpack('!f', bytes.fromhex(hex_value))[0]
    elif precision_bits == 64:
        return struct.unpack('!d', bytes.fromhex(hex_value))[0]
    elif precision_bits == 8:
        # 8-bit floats are non-standard, we'll assume a fixed-point-like scaling for demonstration
        # Treat the value as an integer and scale down (approximation)
        return int(hex_value, 16) / 256.0
    else:
        raise ValueError("Unsupported precision bits.")

# Hex values and their precisions
hex_values = {
    "32-bit": ("40490FDB", 32),
    "64-bit": ("400921FB53C8D4F1", 64),
    "8-bit": ("44", 8),
}

# Convert and store results
results = {}
for label, (hex_val, bits) in hex_values.items():
    results[label] = hex_to_float(hex_val, bits)

# Display results for comparison
print(results)


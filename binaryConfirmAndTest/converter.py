from decimal import Decimal, getcontext

# Set the precision high enough to avoid rounding issues
getcontext().prec = 50  # You can adjust the precision as needed

# Function to convert a fractional decimal to binary using Decimal for high precision
def decimal_to_binary_fraction(decimal_fraction, steps=52):
    result = []
    # Convert the input to a Decimal
    value = Decimal(decimal_fraction)
    binary_val = ''
    for _ in range(steps):
        value *= 2
        integer_part = int(value)
        fractional_part = value - integer_part
        result.append(f"{value:.8f} → Integer part = {integer_part}")
        binary_val += (str(integer_part))
        if fractional_part == 0:  # Stop if there's no fractional part left
            break

        value = fractional_part

    return result, binary_val


# Decimal fraction to convert
fractional_decimal = Decimal('0.14159265')  # Use Decimal for input value

# Convert and print the result
binary_steps, binary_val = decimal_to_binary_fraction(fractional_decimal)
for step in binary_steps:
    print(step)

print(binary_val)
class Pipeline:
    def __init__(self, instructions):
        # program counter
        self.pc = 0
        # store instructions (remove labels)
        self.instructions = []
        # 32 registers
        self.registers = {f'x{i}': 0 for i in range(32)}
        # Memory to store data (address -> value)
        self.memory = {}
        # list to store CSV rows
        self.csv_output = []
        # store the encoded binary instructions
        self.binary_instructions = []
        # map labels to instruction indices
        self.label_to_index = {}
        self.load_instructions(instructions)

    # load instructions into simulator
    def load_instructions(self, instructions):
        cleaned_instructions = []
        # Detect label (':') and remove label
        for instr in instructions:
            if ':' in instr:  # Detect label
                label = instr.replace(":", "").strip()
                self.label_to_index[label] = len(cleaned_instructions)
            else:
                cleaned_instructions.append(instr)

        # display instructions without label
        self.instructions = cleaned_instructions
        print(f"Label mapping: {self.label_to_index}")
        print(f"Cleaned instructions: {self.instructions}")

    # fetch next instruction from  instruction list
    def fetch(self):
        # check if program counter is with the length of the instructions
        if self.pc < len(self.instructions):
            # fetch instruction
            instruction = self.instructions[self.pc]
            self.pc += 1
            print(f"Fetched instruction: {instruction}")
            return instruction
        return None

    # decode instruction to each component
    def decode(self, instruction):
        parts = instruction.split(' ', 1)
        op = parts[0]

        '''
        Split operands
        Extract destination register
        Extract memory operation (address)
        '''

        decoded_parts = tuple(map(str.strip, parts[1].split(',')))
        print(f"Decoded {op}: {decoded_parts}")
        return op, *decoded_parts

    # execute a decoded instruction
    def execute(self, decoded_instruction):
        op = decoded_instruction[0]

        def get_register_value(reg):
            return int(self.registers.get(reg, 0))

        def parse_mem_op(mem_op):
            offset = int(mem_op.split('(')[0])
            base_reg = mem_op.split('(')[1][:-1]
            base_value = get_register_value(base_reg)
            return offset, base_value

        if op == 'lw' or op == 'sw':
            rd = decoded_instruction[1]
            mem_op = decoded_instruction[2]
            offset, base_value = parse_mem_op(mem_op)
            effective_address = offset + base_value

            if op == 'lw':
                value = self.memory.get(effective_address, 0)
                self.registers[rd] = value
                print(f"Executed lw: {rd} = {value} (from memory at {effective_address})")
            else:  # 'sw'
                value = get_register_value(rd)
                self.memory[effective_address] = value
                print(f"Executed sw: {rd} = {value} stored at memory address {effective_address}")

        elif op == 'addi':
            rd, rs, imm = decoded_instruction[1], decoded_instruction[2], int(decoded_instruction[3])
            self.registers[rd] = get_register_value(rs) + imm
            print(f"Executed addi: {rd} = {self.registers[rd]} (from {rs} + {imm})")

        elif op == 'bne':
            rs1, rs2, label = decoded_instruction[1], decoded_instruction[2], decoded_instruction[3]
            if get_register_value(rs1) != get_register_value(rs2):
                if label in self.label_to_index:
                    self.pc = self.label_to_index[label]
                    print(f"Executed bne: Branching to label {label}")
                else:
                    print(f"Error: Label {label} not found")
            else:
                print(f"Executed bne: No branch taken")

        elif op in ['add', 'sub', 'and', 'or']:
            rd, rs1, rs2 = decoded_instruction[1], decoded_instruction[2], decoded_instruction[3]
            if op == 'add':
                self.registers[rd] = get_register_value(rs1) + get_register_value(rs2)
            elif op == 'sub':
                self.registers[rd] = get_register_value(rs1) - get_register_value(rs2)
            elif op == 'and':
                self.registers[rd] = get_register_value(rs1) & get_register_value(rs2)
            else:  # 'or'
                self.registers[rd] = get_register_value(rs1) | get_register_value(rs2)

            print(
                f"Executed {op}: {rd} = {self.registers[rd]} (from {rs1} {'+' if op == 'add' else '-' if op == 'sub' else '&' if op == 'and' else '|'} {rs2})")

    # memory
    def memory_stage(self, decoded_instruction):
        if not decoded_instruction:
            return

        # extract operation type
        op = decoded_instruction[0]
        if op == 'lw':
            rd = decoded_instruction[1]
            print(f"Memory Stage for lw: {rd} = {self.registers[rd]}")

    # write-back
    def write_back(self, decoded_instruction):
        instruction_type, dest_reg = decoded_instruction[0], decoded_instruction[1] if len(
            decoded_instruction) > 1 else None

        def print_write_back(instruction_name):
            value = int(self.registers[dest_reg])
            print(f"Write back for {instruction_name}: {dest_reg} = {value}")

        # Handle instructions with write-back
        if instruction_type in ['lw', 'addi', 'sub', 'and', 'or', 'add']:
            if dest_reg:
                print_write_back(instruction_type)
            else:
                print(f"Error: Missing destination register for {instruction_type}")

        # Handle instructions without write-back
        elif instruction_type in ['sw', 'bne']:
            print(f"{instruction_type.capitalize()} instruction, no write-back to registers")

        else:
            print("Unrecognized instruction type in Write-Back")

        # Print the final register state
        print(f"Register state after write-back: {self.registers}")

    def check_forwarding(self, current_decoded, previous_rows):
        """
        check on forwarding base off of the previous instructions and
        returns fwd_a and fwd_b for their forwarding status
        """

        if not current_decoded or len(current_decoded) < 3:
            return '*', '*'

        fwd_a = '*'
        fwd_b = '*'

        # get source registers for current instruction
        rs1 = current_decoded[2] if len(current_decoded) > 2 else None
        rs2 = current_decoded[3] if len(current_decoded) > 3 else None

        # Check last two instructions
        if len(previous_rows) >= 1:
            prev_instr = previous_rows[-1]
            # forward from previous instruction's result (EX/MEM hazard)
            if prev_instr['RegWrite'] == '1' and prev_instr['Rd']:
                if rs1 == prev_instr['Rd']:
                    fwd_a = '10'
                if rs2 == prev_instr['Rd']:
                    fwd_b = '10'

        if len(previous_rows) >= 2:
            prev_prev_instr = previous_rows[-2]
            # forward from 2 instructions ago (MEM/WB hazard)
            if prev_prev_instr['RegWrite'] == '1' and prev_prev_instr['Rd']:
                if rs1 == prev_prev_instr['Rd'] and fwd_a == '*':
                    fwd_a = '01'
                if rs2 == prev_prev_instr['Rd'] and fwd_b == '*':
                    fwd_b = '01'

        return fwd_a, fwd_b

    def check_data_hazard(self, current_decoded, previous_rows):
        """
        check if instruction needs to be stalled because of a data hazard
        """

        if not current_decoded or not previous_rows:
            return False

        # get source registers for current instruction
        rs1, rs2 = (current_decoded[2] if len(current_decoded) > 2 else None,
                    current_decoded[3] if len(current_decoded) > 3 else None)

        # check for load-use hazard
        prev_instr = previous_rows[-1]  # Most recent instruction
        if prev_instr['MemRd'] == '1' and prev_instr['Rd']:
            if rs1 == prev_instr['Rd'] or rs2 == prev_instr['Rd']:
                return True

        return False

    # create a row for CSV output
    def create_row(self, cycle, instruction, decoded, previous_rows):
        if not decoded:
            return None

        # Check for forwarding hazards
        fwd_a, fwd_b = self.check_forwarding(decoded, previous_rows)

        # Define operation codes for different instructions
        fct3 = {
            'add': '000', 'sub': '000', 'addi': '000',
            'lw': '010', 'sw': '010',
            'bne': '001',
            'or': '110', 'and': '111'
        }.get(decoded[0], '*')

        # Return the row as a dictionary
        return {
            'Cycle': cycle,
            'Instr': instruction,
            'Op': decoded[0],
            'Fct3': fct3,
            'Rd': decoded[1] if len(decoded) > 1 else '',
            'Rs1': decoded[2] if len(decoded) > 2 else '',
            'Rs2': decoded[3] if len(decoded) > 3 else '',
            'RegWrite': '1' if decoded[0] not in ['sw', 'bne'] else '0',
            'ALUSrc': '1' if decoded[0] in ['addi', 'lw', 'sw'] else '0',
            'FwdA': fwd_a,
            'FwdB': fwd_b,
            'MemRd': '1' if decoded[0] == 'lw' else '0',
            'MemWr': '1' if decoded[0] == 'sw' else '0',
            'WBSel': '1' if decoded[0] == 'lw' else '0',
            'bne': '1' if decoded[0] == 'bne' else ''
        }

    def run(self):
        cycle = 0
        previous_rows = []
        done = False

        while not done:
            if self.pc < len(self.instructions):
                # fetch
                instruction = self.fetch()
                if instruction:
                    # decode
                    decoded = self.decode(instruction)
                    if decoded:
                        # check for hazard
                        if self.check_data_hazard(decoded, previous_rows):
                            # add one stall cycle to CSV
                            row = self.create_row(cycle, "stall", None, previous_rows)
                            if row:
                                previous_rows.append(row)
                                self.csv_output.append(row)
                            cycle += 1

                        # process instruction
                        row = self.create_row(cycle, instruction, decoded, previous_rows)
                        if row:
                            previous_rows.append(row)
                            self.csv_output.append(row)

                        # execute
                        self.execute(decoded)

                        # memory
                        self.memory_stage(decoded)

                        # write Back
                        self.write_back(decoded)

                        cycle += 1
            else:
                done = True

    # generate a CSV file
    def generate_csv(self, filename="pipeline_output.csv"):
        import csv

        with open(filename, mode='w', newline='') as file:
            fieldnames = ['Cycle', 'Instr', 'Op', 'Fct3', 'Rd', 'Rs1', 'Rs2',
                          'RegWrite', 'ALUSrc', 'FwdA', 'FwdB', 'MemRd', 'MemWr',
                          'WBSel', 'bne']
            writer = csv.DictWriter(file, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(self.csv_output)

            print(f"CSV simulation saved to {filename}")

    def encode_to_binary(self, instruction):
        # decode instruction
        decoded = self.decode(instruction)

        # addi instruction example: addi x6, x6, 1
        if decoded[0] == 'addi':
            opcode = '0010011'
            funct3 = '000'
            rd = f"{int(decoded[1][1:]):05b}"  # register x6 -> 6 -> 00110
            rs1 = f"{int(decoded[2][1:]):05b}"  # register x6 -> 6 -> 00110
            immediate = f"{int(decoded[3]):012b}"  # Immediate value 1 -> 000000000001

            # binary instruction in addi format
            binary_instruction = f"{immediate}{rs1}{funct3}{rd}{opcode}"
            return binary_instruction

        # lw instruction example: lw x7, 0(x10)
        elif decoded[0] == 'lw':
            opcode = '0000011'
            funct3 = '010'
            rd = f"{int(decoded[1][1:]):05b}"  # register x7 -> 7 -> 00111

            # Parse memory operand 0(x10)
            mem_op = decoded[2]
            # remove parenthesis
            offset, base_reg = mem_op.split('(')
            base_reg = base_reg.strip(')')

            rs1 = f"{int(base_reg[1:]):05b}"
            immediate = f"{int(offset):012b}"

            # binary instruction in lw format
            binary_instruction = f"{immediate}{rs1}{funct3}{rd}{opcode}"
            return binary_instruction

        # sw instruction example: sw x6, 0(x7)
        elif decoded[0] == 'sw':
            opcode = '0100011'
            funct3 = '010'
            rs2 = f"{int(decoded[1][1:]):05b}"  # register x6 -> 6 -> 00110

            # Parse memory operand 0(x7)
            mem_op = decoded[2]

            # Remove the closing parenthesis
            offset, base_reg = mem_op.split('(')
            base_reg = base_reg.strip(')')

            rs1 = f"{int(base_reg[1:]):05b}"
            immediate = f"{int(offset):012b}"

            # binary instruction in sw format
            binary_instruction = f"{immediate}{rs1}{funct3}{rs2}{opcode}"
            return binary_instruction

        # bne instruction example: bne x1, x2, Loop
        elif decoded[0] == 'bne':
            opcode = '1100011'
            funct3 = '001'

            rs1 = f"{int(decoded[1][1:]):05b}"  # register x1 -> 1 -> 00001
            rs2 = f"{int(decoded[2][1:]):05b}"  # register x2 -> 2 -> 00010

            # Label for branch target
            label = decoded[3]

            if label in self.label_to_index:
                # calculate offset in terms of instructions
                current_address = self.pc // 4
                label_address = self.label_to_index[label]
                # multiply by 2 for byte addressing
                offset = (label_address - current_address) * 2

                # handle negative offset
                if offset < 0:
                    # keep only lower 12 bits
                    offset &= 0xFFF

                # Extract bits for the immediate field
                imm_12 = f"{(offset >> 12) & 0x1:01b}"  # bit 12
                imm_11 = f"{(offset >> 11) & 0x1:01b}"  # bit 11
                imm_10_5 = f"{(offset >> 5) & 0x3F:06b}"  # bits 10:5
                imm_4_1 = f"{(offset >> 1) & 0xF:04b}"  # bits 4:1

                # Construct binary instruction in B-type format
                binary_instruction = f"{imm_12}{imm_10_5}{rs2}{rs1}{funct3}{imm_4_1}{imm_11}{opcode}"

                return binary_instruction

        # add instruction
        elif decoded[0] == 'add':
            opcode = '0110011'
            funct3 = '000'
            funct7 = '0000000'

            rd = f"{int(decoded[1][1:]):05b}"
            rs1 = f"{int(decoded[2][1:]):05b}"
            rs2 = f"{int(decoded[3][1:]):05b}"

            # binary instruction in R-format for add
            binary_instruction = f"{funct7}{rs2}{rs1}{funct3}{rd}{opcode}"

            return binary_instruction

        # sub instruction
        elif decoded[0] == 'sub':
            opcode = '0110011'
            funct3 = '000'
            funct7 = '0100000'

            rd = f"{int(decoded[1][1:]):05b}"
            rs1 = f"{int(decoded[2][1:]):05b}"
            rs2 = f"{int(decoded[3][1:]):05b}"

            # binary instruction in R-format for sub
            binary_instruction = f"{funct7}{rs2}{rs1}{funct3}{rd}{opcode}"

            return binary_instruction

        # and instruction
        elif decoded[0] == 'and':
            opcode = '0110011'
            funct3 = '111'
            funct7 = '0000000'

            rd = f"{int(decoded[1][1:]):05b}"
            rs1 = f"{int(decoded[2][1:]):05b}"
            rs2 = f"{int(decoded[3][1:]):05b}"

            # binary instruction in R-format for and
            binary_instruction = f"{funct7}{rs2}{rs1}{funct3}{rd}{opcode}"

            return binary_instruction

        # or instruction
        elif decoded[0] == 'or':
            opcode = '0110011'
            funct3 = '110'
            funct7 = '0000000'

            rd = f"{int(decoded[1][1:]):05b}"
            rs1 = f"{int(decoded[2][1:]):05b}"
            rs2 = f"{int(decoded[3][1:]):05b}"

            # binary instruction in R-format for or
            binary_instruction = f"{funct7}{rs2}{rs1}{funct3}{rd}{opcode}"

            return binary_instruction

        else:
            # handle unknown instruction type
            raise ValueError(f"Unknown instruction: {instruction}")

    def generate_binary_instructions(self, filename="unrolled_instructions.bin"):
        """
        Creates binary instructions from the pipeline state and writes them to a file.
        """
        # Generate all binary instructions
        self.binary_instructions = [self.encode_to_binary(instruction) for instruction in self.instructions]

        # Write the binary instructions to a binary file
        with open(filename, 'wb') as file:
            for binary_instruction in self.binary_instructions:
                # Convert binary string to integer
                instruction_int = int(binary_instruction, 2)

                # Mask to 32 bits for handling negative values
                instruction_int &= 0xFFFFFFFF

                # Write as 32-bit unsigned integer
                file.write(instruction_int.to_bytes(4, byteorder='big', signed=False))

        print(f"Binary instructions saved to {filename}")

    def write_instructions_to_file(self, filename):
        """Write instruction set to text file."""
        try:
            with open(filename, 'w') as file:
                for instruction in self.instructions:
                    file.write(instruction + '\n')
            print(f"Instructions successfully written to {filename}")
        except IOError as e:
            print(f"An error occurred while writing to the file: {e}")


def read_binary_file_as_binary(filename):
    with open(filename, 'rb') as file:
        # Read 4 bytes at a time
        while byte := file.read(4):
            # convert bytes to binary
            print(' '.join(f"{b:08b}" for b in byte))


# unrolled instructions
instructions1 = [
    "lw x7, 0(x10)",
    "lw x6, 0(x7)",
    "addi x6, x6, 1",
    "sw x6, 0(x7)",
    "lw x6, 4(x7)",
    "addi x6, x6, 1",
    "sw x6, 4(x7)",
    "lw x6, 8(x7)",
    "addi x6, x6, 1",
    "sw x6, 8(x7)"
]

# branch instructions
instructions2 = [
    "lw x7, 0(x10)",
    "addi x5, x0, 3",
    "Loop:",
    "lw x6, 0(x7)",
    "addi x6, x6, 1",
    "sw x6, 0(x7)",
    "addi x7, x7, 4",
    "addi x5, x5, -1",
    "bne x5, x0, Loop"
]

# hazards instructions
instructions3 = [
    "sub x2, x1, x3",
    "and x12, x2, x5",
    "or x13, x6, x2",
    "and x2, x12, x13",
    "add x14, x2, x2"
]

# dynamic instructions
instructions4 = [
    "lw x7, 0(x10)",
    "lw x6, 0(x7)",
    "lw x8, 4(x7)",
    "lw x9, 8(x7)",
    "addi x6, x6, 1",
    "addi x8, x8, 1",
    "addi x9, x9, 1",
    "sw x6, 0(x7)",
    "sw x8, 4(x7)",
    "sw x9, 8(x7)"
]

# Unrolled Simulation
pipeline1 = Pipeline(instructions1)
pipeline1.run()
pipeline1.generate_csv("unrolled_simulation.csv")
pipeline1.generate_binary_instructions("unrolled_instructions.bin")
read_binary_file_as_binary("../unrolled_instructions.bin")

# Branch Simulation
pipeline2 = Pipeline(instructions2)
pipeline2.run()
pipeline2.generate_csv("branch_simulation.csv")
pipeline2.generate_binary_instructions("branch_instructions.bin")
read_binary_file_as_binary("../branch_instructions.bin")

# Hazards Simulation
pipeline3 = Pipeline(instructions3)
pipeline3.run()
pipeline3.generate_csv("hazards_simulation.csv")
pipeline3.generate_binary_instructions("hazards_instructions.bin")
read_binary_file_as_binary("../hazards_instructions.bin")

# Dynamic Scheduling
pipeline4 = Pipeline(instructions4)
pipeline4.write_instructions_to_file("dynamic_re-order.txt")
pipeline4.run()
pipeline4.generate_csv("dynamic_simulation.csv")
pipeline4.generate_binary_instructions("dynamic_instructions.bin")
read_binary_file_as_binary("../dynamic_instructions.bin")

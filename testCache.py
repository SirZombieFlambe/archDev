# import numpy as np
# import random
#
# # Cache Configuration
# CACHE_SIZE = 4096  # 4 KB
# BLOCK_SIZE = 16  # Block size in bytes
# ASSOCIATIVITY = 2  # 2-way Set Associative
# NUM_BLOCKS = CACHE_SIZE // BLOCK_SIZE
# SET_COUNT = NUM_BLOCKS // ASSOCIATIVITY
#
# # Cache Initialization
# tags = np.full((SET_COUNT, ASSOCIATIVITY), -1, dtype=np.int32)  # Tag storage
# valid_bits = np.zeros((SET_COUNT, ASSOCIATIVITY), dtype=bool)  # Valid bits
#
#
# # Function to simulate cache access
# def cache_access(address):
#     global MISS_COUNT, HIT_COUNT, tags, valid_bits
#     block_offset_bits = int(np.log2(BLOCK_SIZE))
#     index_bits = int(np.log2(SET_COUNT))
#
#     block_address = address >> block_offset_bits
#     index = block_address & ((1 << index_bits) - 1)
#     tag = block_address >> index_bits
#
#     # Check cache hit
#     for i in range(ASSOCIATIVITY):
#         if valid_bits[index][i] and tags[index][i] == tag:
#             HIT_COUNT += 1
#             return "hit"
#
#     # Cache miss handling
#     MISS_COUNT += 1
#     replacement_index = MISS_COUNT % ASSOCIATIVITY  # Simple FIFO replacement
#     tags[index][replacement_index] = tag
#     valid_bits[index][replacement_index] = True
#     return "miss"
#
#
# # Function to simulate row-major and column-major accesses
# def simulate_accesses(array_shape, element_size, pattern="row-major", base_address=0):
#     global MISS_COUNT, HIT_COUNT, tags, valid_bits
#     MISS_COUNT, HIT_COUNT = 0, 0
#
#     # Reset cache state
#     tags.fill(-1)
#     valid_bits.fill(False)
#
#     # Generate addresses based on access pattern
#     if pattern == "row-major":
#         addresses = [
#             base_address + ((i * array_shape[1] + j) * element_size)
#             for i in range(array_shape[0])
#             for j in range(array_shape[1])
#         ]
#     elif pattern == "column-major":
#         addresses = [
#             base_address + ((j * array_shape[0] + i) * element_size)
#             for j in range(array_shape[1])
#             for i in range(array_shape[0])
#         ]
#
#     # Simulate accesses
#     for addr in addresses:
#         cache_access(addr)
#
#     total_accesses = len(addresses)
#     miss_rate = MISS_COUNT / total_accesses
#     return miss_rate
#
#
# # Cache Miss Rate Analysis for Different Data Types
# data_types = {
#     "char": (64, 64, 1),
#     "short": (32, 32, 2),  # Adjusted to match 4096 bytes
#     "int": (32, 32, 4),
#     "long": (16, 16, 8),  # Adjusted to match 4096 bytes
# }
#
# results = []
#
# # Generate a random aligned base address
# aligned_min = 0
# aligned_max = 2 ** 16  # Arbitrary limit for the address space
# base_address = random.randint(aligned_min // BLOCK_SIZE, aligned_max // BLOCK_SIZE) * BLOCK_SIZE
#
# # Run the simulation
# for dtype, (rows, cols, size) in data_types.items():
#     row_miss_rate = simulate_accesses((rows, cols), size, pattern="row-major", base_address=base_address)
#     col_miss_rate = simulate_accesses((rows, cols), size, pattern="column-major", base_address=base_address)
#     results.append((dtype, row_miss_rate, col_miss_rate))
#
# # Print Results
# print("Data Type Miss Rate Analysis")
# print("------------------------------------------------")
# print(f"{'Data Type':<10}{'Row Major':<15}{'Column Major':<15}")
# for dtype, row_miss, col_miss in results:
#     print(f"{dtype:<10}{row_miss * 100:>10.2f}%{col_miss * 100:>15.2f}%")
#
# # L2 Cache Size since L3 (18MiB) take ages
# L2_CACHE_SIZE = 10 * (2 ** 20)  # 10 MiB
# RANGE_LIMIT = L2_CACHE_SIZE // 2  # Half of L3 cache size
#
#
# def simulate_random_accesses(array_shape, element_size, iterations=30):
#     global MISS_COUNT, HIT_COUNT, tags, valid_bits
#     miss_rates = []
#
#     for _ in range(iterations):
#         MISS_COUNT, HIT_COUNT = 0, 0
#         tags.fill(-1)
#         valid_bits.fill(False)
#
#         # Generate random addresses
#         addresses = [random.randint(0, RANGE_LIMIT - 1) * element_size for _ in range(array_shape[0] * array_shape[1])]
#
#         # Simulate accesses
#         for addr in addresses:
#             cache_access(addr)
#
#         miss_rate = MISS_COUNT / len(addresses)
#         miss_rates.append(miss_rate)
#
#     # Average miss rate
#     return sum(miss_rates) / len(miss_rates)
#
#
# # Adjusted Data Types and Array Sizes for L3 Cache
# data_types = {
#     "char": (RANGE_LIMIT // 1, 1),  # Array size based on element size of 1 byte
#     "short": (RANGE_LIMIT // 2, 2),  # 2 bytes per element
#     "int": (RANGE_LIMIT // 4, 4),  # 4 bytes per element
#     "long": (RANGE_LIMIT // 8, 8),  # 8 bytes per element
# }
#
# random_results = []
#
# # Run Random Access Simulation
# for dtype, (elements, size) in data_types.items():
#     rows = int(elements ** 0.5)  # Create a roughly square array
#     cols = rows
#     avg_miss_rate = simulate_random_accesses((rows, cols), size)
#     random_results.append((dtype, avg_miss_rate))
#
# # Print Random Access Results
# print("\nRandom Access Miss Rate Analysis")
# print("------------------------------------------------")
# print(f"{'Data Type':<10}{'Random Access':<15}")
# for dtype, rand_miss in random_results:
#     print(f"{dtype:<10}{rand_miss:<15.4f}")


import numpy as np

class Cache:
    def __init__(self, BLOCK_SIZE, CACHE_SIZE, associativity):
        self.BLOCK_SIZE = BLOCK_SIZE
        self.associativity = associativity
        self.num_sets = CACHE_SIZE // (BLOCK_SIZE * associativity)

        # Initialize the cache data: valid bits and tags
        self.valid_bits = np.zeros((self.num_sets, associativity), dtype=np.int8)
        self.tags = np.zeros((self.num_sets, associativity), dtype=np.int32)
        self.lru = np.zeros((self.num_sets, associativity), dtype=np.int8)  # To track LRU

    def access(self, address):
        # Calculate the cache index and tag
        line = address // self.BLOCK_SIZE
        cache_index = line % self.num_sets
        tag = line // self.num_sets

        # Check if the tag is in the cache set
        for way in range(self.associativity):
            if self.valid_bits[cache_index, way] == 1 and self.tags[cache_index, way] == tag:
                # Cache hit
                self.update_lru(cache_index, way)
                return True

        # Cache miss
        self.replace_line(cache_index, tag)
        return False

    def update_lru(self, cache_index, way):
        # Update the LRU status for all ways in the set
        self.lru[cache_index] = np.where(self.lru[cache_index] > self.lru[cache_index, way],
                                         self.lru[cache_index] - 1,
                                         self.lru[cache_index])
        self.lru[cache_index, way] = self.associativity - 1

    def replace_line(self, cache_index, tag):
        # Find the least recently used (LRU) way to replace
        lru_way = np.argmin(self.lru[cache_index])
        self.tags[cache_index, lru_way] = tag
        self.valid_bits[cache_index, lru_way] = 1
        self.update_lru(cache_index, lru_way)

# Function to determine cache layout description
def describe_cache_layout(associativity, num_sets):
    if associativity == 1:
        return "Direct-Mapped Cache"
    elif num_sets == 1:
        return "Fully-Associative Cache"
    else:
        return f"{associativity}-Way Set-Associative Cache"

# Function to simulate cache accesses and calculate miss rate
def simulate_cache(array_size, element_size, base_address, BLOCK_SIZE, CACHE_SIZE, associativity):
    cache = Cache(BLOCK_SIZE, CACHE_SIZE, associativity)

    row_major_hits, column_major_hits = 0, 0

    # Row-major access
    for i in range(array_size):
        for j in range(array_size):
            address = base_address + (i * array_size + j) * element_size
            if cache.access(address):
                row_major_hits += 1

    row_major_miss_rate = 1 - (row_major_hits / (array_size * array_size))

    # Reset cache for column-major test
    cache = Cache(BLOCK_SIZE, CACHE_SIZE, associativity)

    # Column-major access
    for j in range(array_size):
        for i in range(array_size):
            address = base_address + (i * array_size + j) * element_size
            if cache.access(address):
                column_major_hits += 1

    column_major_miss_rate = 1 - (column_major_hits / (array_size * array_size))

    return row_major_miss_rate, column_major_miss_rate

def calculate_array_size(CACHE_SIZE, element_size):
    total_elements = CACHE_SIZE // element_size  # Total elements that fit in the cache
    array_size = int(total_elements**0.5)       # Determine the size of the square array
    return array_size

# Constants
BLOCK_SIZE = 64  # Bytes
CACHE_SIZE = 4092  # Bytes
L3_CACHE_SIZE = 18 * 1024 * 1024  # Bytes

associativity = 4
base_address = 0x8aa1000  # Start at this arbitrary address


print("Question 1:")
# Cache layout description
cache_layout = describe_cache_layout(associativity, CACHE_SIZE // (BLOCK_SIZE * associativity))
print(f"Cache Layout: {cache_layout}\n")

# Data types and configurations
data_types = [
    {"name": "char", "element_size": 1, "array_size": 64},
    {"name": "short", "element_size": 2, "array_size": calculate_array_size(L3_CACHE_SIZE, 2)},
    {"name": "int", "element_size": 4, "array_size": 32},
    {"name": "long", "element_size": 8, "array_size": calculate_array_size(L3_CACHE_SIZE, 8)},
]

# Simulate and report miss rates
# Header
print(f"{'Data Type':<12} | {'Element Size':<12} | {'Array Size':<15} | {'Row-Major Miss Rate':<22} | {'Column-Major Miss Rate':<22}")
print("-" * 90)

for data_type in data_types:
    row_miss, col_miss = simulate_cache(
        data_type["array_size"],
        data_type["element_size"],
        base_address,
        BLOCK_SIZE,
        CACHE_SIZE,
        associativity
    )
    array_size_str = f"{data_type['array_size']}x{data_type['array_size']}"
    print(f"{data_type['name']:<12} | {data_type['element_size']:<12} | {array_size_str:<15} | {row_miss * 100:>6.2f}%{'':<15} | {col_miss * 100:>6.2f}%{'':<15}")

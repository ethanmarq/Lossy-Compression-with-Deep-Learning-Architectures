import numpy as np

# Path to your MP4 file
file_path = 'output.mp4'

# Read the file in binary mode
with open(file_path, 'rb') as file:
    binary_data = file.read()

# Convert binary data to a NumPy array
np_array = np.frombuffer(binary_data, dtype=np.uint8)

# Measure the size of the NumPy array
size_in_bytes = np_array.size

print(f"The size of the MP4 file in bytes is: {size_in_bytes}")

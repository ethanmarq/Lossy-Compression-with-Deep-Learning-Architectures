#!/usr/bin/env python
from pathlib import Path
import json
import numpy as np
import sys
import os
import math
import matplotlib.pyplot as plt
from PIL import Image
import pandas as pd
import glob
from io import BytesIO
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
import csv
import time

import random
from matplotlib import image

#the sizes are based on the memory size of the elements in the numpy array. (why: so we don't take into account extra numpy metadata or any extra metadata that could change from version to version or computer to computer)

# Paths and directories
paths = ['Test/imgs/*.png', 'Train/imgs/*.png']
path_names = ['Test', 'Train']
srcpath = "/scratch/aniemcz/cavsMiniLossyCompressorsV2/mixed"
dest = 'output/'

# Function to calculate PSNR and SSIM
def calculate_metrics(original, compressed):
    original_array = np.array(original)
    compressed_array = np.array(compressed)
    psnr_value = peak_signal_noise_ratio(original_array, compressed_array)
    ssim_value = structural_similarity(original_array, compressed_array, channel_axis=2)
    return psnr_value, ssim_value

# Prepare CSV file
csv_file = 'compression_metrics.csv'
csv_header = ['filename', 'quality', 'uncompressed_size_bytes', 'compressed_size_bytes', 'psnr', 'ssim', 'split', 'time_to_compress', 'time_to_decompress', 'total_time']
with open(csv_file, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(csv_header)

for i, path in enumerate(paths):
    l = glob.glob( os.path.join(srcpath, path) )
    l.sort()
    
    path_name = path_names[i]
        
    print(f"{len(l)} is the number of images found for {path}")
    
    print(f"Processing {len(l)} images in {path}")
    for name in l:
        print(f"Processing {name}")
                            
        im1 = Image.open(name)
                
        imgNp = np.array(im1)
        uncompressed_size = imgNp.nbytes
                
        for Quality in range(0, 101, 10):
            buffer = BytesIO()
            
            # Measure compression time
            start_compress = time.time()
            im1.save(buffer, "JPEG", quality=Quality)
            end_compress = time.time()
            time_to_compress = end_compress - start_compress
            
            # Measure decompression time
            start_decompress = time.time()
            compressed_image = Image.open(buffer)
            compressed_image.load()  # This forces the image to be fully loaded
            end_decompress = time.time()
            time_to_decompress = end_decompress - start_decompress
            
            total_time = time_to_compress + time_to_decompress
            
            # Calculate metrics
            psnr, ssim = calculate_metrics(im1, compressed_image) #takes in pillow image and returns decimal values for psnr and ssim
            
            # Get compressed size
            #buffer.seek(0, 2)  # Go to the end of the buffer
            #compressed_size = buffer.tell()  # Get current position (size)
            compressed_size = np.frombuffer(buffer.getvalue(), dtype=np.uint8).nbytes #you can do the above too although this way it is consistent with sizes for lossy compressor script
            
            dest_2 = '/Train/rgb/' if path_name.capitalize() == "Train" else '/Test/rgb/'
            path = dest + 'Q' + str(Quality) + dest_2
            
            # Create directory if it doesn't exist
            #os.makedirs(path, exist_ok=True) #commented out for now since im not saving them
            
            filename = os.path.basename(name).split('/')[-1]
            filename = filename.split('.', 1)[0] + '.jpg'
            location = path + filename
            
            #print(f"Saving to {location}")
            
            # Save the compressed image
            #with open(location, "wb") as handle:
                #handle.write(buffer.getvalue())
            
            # Write metrics to CSV
            with open(csv_file, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([filename, Quality, uncompressed_size, compressed_size, psnr, ssim, path_name, time_to_compress, time_to_decompress, total_time])

print(f"Compression analysis complete. Results saved to {csv_file}")
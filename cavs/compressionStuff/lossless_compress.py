#!/usr/bin/env python

#NOTE:
#For whatever reason this script does not work as a sbatch job

from pathlib import Path
import json
import libpressio
import numpy as np
import sys
import os
import math
import numpy as np
#from ipywidgets import widgets,interact,IntProgress #I uncommented this
import matplotlib.pyplot as plt
from matplotlib import image
from skimage import morphology
from skimage.morphology import closing, square, reconstruction
from skimage import filters
from sklearn import preprocessing
#from OctCorrection import *
#from ImageProcessing import *
import pickle
from mpl_toolkits.mplot3d import Axes3D
import os
import glob
import re

from PIL import Image
import pandas as pd
#import cv2
from tifffile import imsave, imread
from numba import jit

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import rawpy
import imageio

from oct_converter.readers import FDS
from struct import unpack

import time
import cv2

if len(sys.argv) > 1:
    split = sys.argv[1].capitalize()
    print(f"split chosen is {split}")
else:
    raise Exception("Need to pass the test train split as arg")


def compression(srcpath, paths, compressor, modes):
    threads = [1]
    for thread in threads:
        mode_count = [1]
        for modes in mode_count:
            mode = "base"
            
            i = 0
            while i < len(paths):
                #print(len(paths))
                #size = sizes[i]
                
                l=glob.glob(srcpath+paths[i])
                l.sort()
                
                #l = l[0:10]
                
                #read in all the image data as one big numpy array
                input_data = np.concatenate([np.array(Image.open(name)).flatten() for name in l])
                input_data = np.squeeze(input_data)
                
                #Split the 1D numpy array of image data into batches
                #batch_size = 100
                #the batch size was found to be roughly the maximum amount of values I could encode at once before it would fail to encode
                #I may need to see as different encoders might have different limits
                batch_size = 100000000 #in image data value so for a rgb 100x100 img it would be 3x100x100 = 30000 pixels
                batches = [input_data[i:i + batch_size] for i in range(0, len(input_data), batch_size)]
                #batches = [input_data]
                
                print("Input data is of shape {}".format(input_data.shape))
                
                #Will run this for each compressor
                for comp in compressors:
                    
                    #NOTE: Only will be using compression level 9
                    j = 0
                    #bounds = [math.exp(-7),math.exp(-6),math.exp(-5),math.exp(-4),math.exp(-3),math.exp(-2),math.exp(-1)]
                    max_compression_level = 9 #so should go from 0 to 9 (so 10 compression_levels)
                    
                    #Run this for each compression level
                    while(j<=max_compression_level):
                        
                        #input_data = input_data.astype(np.float32) #commented this out since float32 is 32bits while images are originally stored as uint8 so only 8 bits so thats like 4x the bits so I think that partially why it struggled to encode

                        #i_data = input_data.copy()
                        #D_data = input_data.copy()
                        #diff_data = input_data.copy()
                        #decomp_data = input_data.copy()
                        #de_data = input_data.copy()
                        
                        #I think for the times I will store the times in a list and then take the average of them
                        #For the sizes I will add up the uncompressed size for each batch, same thing for compressed and decompressed
                        #compression ratio I will manually calculate at the end
                        
                        #Times:
                        batches_metrics = {
                            "start_times":[],
                            "normalize_times":[],
                            "diff_times":[],
                            "comp_times":[],
                            "encoding_times":[],
                            "de_comp_times":[],
                            "end_times":[],
                            "uncompressed_size":[],
                            "compressed_size":[],
                            "decompressed_size":[], #(not really needed for lossless)   
                        }
                        
                        #start timer
                        #start = time.time()
                        
                        counter = 0
                        counter = counter + 1

                        #Loop through the batches
                        for batch_idx, batch in enumerate(batches):
                            #batch = batch.astype(np.float32)
                            #i_data = batch.copy()
                            D_data = batch.copy()
                            #diff_data = batch.copy()
                            decomp_data = batch.copy()
                            #de_data = batch.copy()
                            
                            start = time.time()
                            batches_metrics["start_times"].append(start)
                            
                            batch = cv2.normalize(batch, None, 0, 1, cv2.NORM_MINMAX)
                            #input_data = (decomp_data - decomp_data.min())/ (decomp_data.max() - decomp_data.min())
                            
                            normalize_time = time.time() - start                
                            diff_time = time.time() - start - normalize_time   
                            batches_metrics["normalize_times"].append(normalize_time)
                            batches_metrics["diff_times"].append(diff_time)
                                                        
                            compressor = libpressio.PressioCompressor.from_config({
                                  # configure which compressor to use
                            "compressor_id": "blosc",
                                  # configure the set of metrics to be gathered
                                  "early_config": {
                                    "blosc:compressor": comp,
                                    "blosc:metric": "composite",
                                    "composite:plugins": ["time", "size", "error_stat", "external"]
                                    },
                                    "compressor_config": {
                                            "blosc:clevel": j,
                                    }
                            })
                            
                            batch = np.squeeze(batch)
                            
                            comp_data = compressor.encode(batch)
                            
                            comp_time = time.time() - diff_time - normalize_time - start
                            encoding_time = time.time() - start
                            batches_metrics["comp_times"].append(comp_time)
                            batches_metrics["encoding_times"].append(encoding_time)

                            #print(f"input comp data is of shape {comp_data.shape}")
                            #print(f"current batch is batch {batch_idx}")
                            decomp_data = compressor.decode(comp_data, decomp_data)
                            #print(f"output decomp data is of shape {decomp_data.shape}")
                            
                           

                            D_data = cv2.normalize(decomp_data, None, 255, 0, cv2.NORM_MINMAX, cv2.CV_8U)
                            #D_data = (decomp_data - decomp_data.min())/ (decomp_data.max() - decomp_data.min()) * 255

                            de_comp_time = time.time() - encoding_time - start
                            end_time = time.time() - start
                            batches_metrics["de_comp_times"].append(de_comp_time)
                            batches_metrics["end_times"].append(end_time)

                            #D_data = decomp_data.copy()

                            #print("The time of execution of above program is :",(end) * 10**3, "ms")
                            metrics = compressor.get_metrics()
                            D_data = D_data.astype(np.uint8)

                            #get size from libpressio
                            #we pick the size of the uncompressed dataset since that is how it was for the images
                            size = metrics['size:uncompressed_size'] # in bytes
                            
                            #returns size in bytes
                            uncompressed_size = metrics['size:uncompressed_size']
                            compressed_size = metrics['size:compressed_size']
                            decompressed_size = metrics['size:decompressed_size']
                            batches_metrics["uncompressed_size"].append(uncompressed_size)
                            batches_metrics["compressed_size"].append(compressed_size)
                            batches_metrics["decompressed_size"].append(decompressed_size)
                            
                            dest = '/scratch/aniemcz/cavsMiniLosslessCompressors/'
                            save = 3 #save 3 so no save at all
                            #save 0 is for getting cr for multiple images separately while save 1 is for getting it for the whole dataset

                            if save == 0:

                                if i == 0:
                                    dest_2 = '/rgb/'
                                elif i == 1:
                                    dest_2 = '/id/'

                                #dest = '/scratch/mfaykus/dissertation/datasets/rellis-images/compressed/'
                                #print(name)
                                filename = os.path.basename(name).split('/')[-1]

                                path = dest + f'{comp}/level_{j}' + dest_2

                                # make the result directory
                                if not os.path.exists(path):
                                    os.makedirs(path)

                                im = Image.fromarray(D_data)

                                print(path + str(filename))

                                im.save(path + str(filename))
                                imageio.imwrite(path + str(filename), D_data)
                            elif save == 1:
                                if i == 0:
                                    dest_2 = '/rgb_compressed_binary/'
                                elif i == 1:
                                    dest_2 = '/id_compressed_binary/'

                                folder_path = dest + f'{comp}/level_{j}' + dest_2 + "compressed_output/"
                                # make the result directory
                                if not os.path.exists(folder_path):
                                    os.makedirs(folder_path)

                                # Save the compressed data to a binary file
                                path = folder_path + "compressed_dataset.bin"

                                #checks for when I accidentally saved as folder
                                if os.path.exists(path) and os.path.isdir(path):
                                    os.rmdir(path)

                                #check on the first batch if there is already a file there if so then delete
                                #since we are appending to file and so need to make sure that at beginning 
                                #we are creating a new file to append to
                                if os.path.exists(path) and os.path.isfile(path) and batch_idx == 0:
                                    os.remove(path)
                                    
                                #append to existing file
                                with open(path, 'ab') as f:
                                    f.write(comp_data)
                            #End of batch
                        
                        #After part
                                                
                        #for these metrics it is being recorded as the average of the metrics for each batch
                        #in seconds
                        start_time = np.mean(batches_metrics["start_times"])
                        normalize_time = np.mean(batches_metrics["normalize_times"])
                        diff_time = np.mean(batches_metrics["diff_times"])
                        comp_time = np.mean(batches_metrics["comp_times"])
                        encoding_time = np.mean(batches_metrics["encoding_times"])
                        de_comp_time = np.mean(batches_metrics["de_comp_times"])
                        end_time = np.mean(batches_metrics["end_times"])
                        
                        #in bytes
                        compressed_dataset_size = sum(batches_metrics["compressed_size"])     
                        uncompressed_dataset_size = sum(batches_metrics["uncompressed_size"]) 
                        decompressed_dataset_size = sum(batches_metrics["decompressed_size"]) 
                        
                        compression_ratio = uncompressed_dataset_size / compressed_dataset_size
                        
                        #(compression ratio should be the same as taking mean of the compression ratio of each batch?)
                        size = uncompressed_dataset_size
                        
                        df.loc[len(df)] = [
                            paths[i], 
                            comp, 
                            (size/encoding_time)/1000000, #records in megabytes (1e6 bytes)
                            (size/de_comp_time)/1000000, #records in megabytes (1e6 bytes)
                            compression_ratio, #was originally size/metrics['size:compressed_size'],
                            normalize_time,
                            diff_time,
                            comp_time,
                            encoding_time,
                            de_comp_time,
                            end_time,
                            thread,
                            uncompressed_dataset_size, #uncompressed size of data in bytes
                            compressed_dataset_size, #compressed size of data in bytes
                            decompressed_dataset_size, #decompressed size of data in bytes (obv same as uncompressed b/c lossless)
                            j, #compression level (0 to 9 with 0 being no compression and 9 being max compression)
                            split
                        ]
                        j = j + 1
                i = i + 1
            mode_count = mode_count[0] + 1
    return df


#paths = ['id/*.jpg', 'rgb/*.jpg']
#paths = ['masks/*.png', 'imgs/*.png'] #assuming the masks are int map
paths = ['imgs/*.png'] #only compressing the rgb images and not the masks

#srcpath='/scratch/mfaykus/dissertation/datasets/rellis-images/train/'
#split = "Test"
srcpath = f'/scratch/aniemcz/cavsMiniLossyCompressorsV2/mixed/{split}/'

#compressors = ['zstd','blosclz','lz4','lz4hc','zlib'] #lossless compressors
#compressors = ['zstd'] #lossless compressors
compressors = ['zstd','blosclz','lz4','lz4hc','zlib']

#compressors = ['zfp', 'sz3'] #lossy compressors
#compressors = ['sz3']

df = pd.DataFrame({
            'filename':[],
            'compressor':[],
            'cBW':[],
            'dBW':[],
            'CR':[],
            'normalize_time':[],
            'diff_time':[],
            'comp_time':[],
            'encoding_time':[],
            'de_comp_time':[],
            'total_time':[],
            'thread':[],
            'uncompressed_size_bytes':[],
            'compressed_size_bytes':[],
            'decompressed_size_bytes':[],
            'compression_level':[], #specific to lossless compressors only
            'split':[],
})

mode = ["base"]

data = compression(srcpath, paths, compressors, mode)
data.to_csv(f'output/cavs_mini_lossless_binary_{split}_all.csv')
#data.to_csv('compression/rellis_base_lossy.csv')

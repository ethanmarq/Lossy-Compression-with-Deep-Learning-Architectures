#!/usr/bin/env python

from pathlib import Path
import json
import libpressio
import numpy as np
import sys
import os
import math
import numpy as np
from skimage import morphology
from skimage.morphology import closing, square, reconstruction
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage import filters
from sklearn import preprocessing

import pickle
import os
import glob
import re

from PIL import Image
import pandas as pd
#import cv2
from tifffile import imsave, imread
from numba import jit

#import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import rawpy
import imageio


from oct_converter.readers import FDS
from struct import unpack

from PIL import Image

from skimage.metrics import structural_similarity

import time
import cv2

if len(sys.argv) > 1:
    split = sys.argv[1].capitalize()
    print(f"split chosen is {split}")
else:
    raise Exception("Need to pass the test train split as arg")

def compression(srcpath, paths, compressor, modes, split):
    threads = [1]
    for thread in threads:
        mode_count = [1]
        for modes in mode_count:
            mode = "base"
            
            i = 0
            while i < len(paths):
                l=glob.glob(srcpath+paths[i])
                l.sort()
                
                print(f"{len(l)} is the number of images found for {paths[i]}")
                
                #print(len(paths))
                #size = sizes[i]
                for comp in compressors:
                    j = 0
                    #should add 'base' as another bounds
                    bounds = [math.exp(-7),math.exp(-6),math.exp(-5),math.exp(-4),math.exp(-3),math.exp(-2),math.exp(-1)]
                    bound_names = ["E-7", "E-6", "E-5", "E-4", "E-3", "E-2", "E-1"]
                    #these exponents are base e

                    while(j<len(bounds)):
                        
                        #print(l)
                        counter = 0
                        for name in l:
                            #print(name)
                            counter = counter + 1
                            
                            bound = bounds[j]
                            bound_name = bound_names[j]
                            
                            inp = Image.open(name)
                            input_data_uint8 = np.array(inp).astype(np.uint8)
                            
                            i_data = input_data_uint8.copy() #copy normalized uncompressed uint8 image data for ssim and psnr values
                            D_data = input_data_uint8.copy()
                            
                            input_data_float32 = input_data_uint8.astype(np.float32) # converts uint8 to float32 data type (8 to 32 bits)
                            decomp_data = input_data_float32.copy() #copies the float32 0-1 input image data
                            
                            #start timer
                            start = time.time()
                            
                            #normalize input
                            input_data_float32 = cv2.normalize(input_data_float32, None, 0, 1, cv2.NORM_MINMAX) #normalize values in array from 0-255 to 0-1 floating point (.astype step above is likely redundant although doesn't hurt it)
                            
                            comp_normalize_time = time.time() - start             

                            #config
                            compressor = libpressio.PressioCompressor.from_config({ 
                                # configure which compressor to use
                                "compressor_id": comp,
                                    # configure the set of metrics to be gathered
                                    "early_config": {
                                        "pressio:metric": "composite",
                                        "pressio:nthreads":thread,
                                        "composite:plugins": ["time", "size", "error_stat"]
                                    },
                                    # configure SZ/zfp
                                    "compressor_config": {
                                        "pressio:abs": bounds[6],
                                    }
                            })
                            
                            comp_config_time = time.time() - comp_normalize_time

                            #encode
                            comp_data = compressor.encode(input_data_float32) 
                            
                            encoding_time = time.time() - comp_config_time
                            
                            comp_time = time.time() - start

                            #decode
                            decomp_data = compressor.decode(comp_data, decomp_data)
                            
                            decoding_time = time.time() - comp_time
                            
                            #normalize output
                            D_data = cv2.normalize(decomp_data, None, 255, 0, cv2.NORM_MINMAX, cv2.CV_8U) #normalize the float32 decompressed data from 0-1 to 0-255. Returns a numpy array of data type uint8
                            
                            decomp_normalize_time = time.time() - decoding_time
                                                        
                            de_comp_time = time.time() - comp_time
                            
                            end = time.time() - start

                            #print("The time of execution of above program is :",(end) * 10**3, "ms")
                            metrics = compressor.get_metrics()
                            
                            #(ssim, diff) = structural_similarity(i_data[:,:,0], D_data[:,:,0], data_range=255, full=True)
                            print("size bytes i_data ", i_data.nbytes)
                            print("size bytes D_data ", D_data.nbytes)
                            print(i_data[0:10][0])
                            print(D_data[0:10][0])
                            ssim = structural_similarity(i_data, D_data, channel_axis=2)
                            
                            psnr_value = psnr(i_data, D_data, data_range=255)

                            #diff = (diff * 255).astype("uint8")
                            
                            print(f"ssim: {ssim}")
                            print(f"psnr_value: {psnr_value}")
                                                                
                            0/0
                            #get sizes from libpressio
                            #compressed_size = metrics["size:compressed_size"]     
                            #uncompressed_size = metrics["size:uncompressed_size"]
                            #decompressed_size = metrics["size:decompressed_size"]
                            #get the number of bytes the elements in this array take up (doesn't include numpy array metadata)
                            compressed_size_float32 = comp_data.nbytes
                            uncompressed_size = input_data_uint8.nbytes
                            uncompressed_size_float32 = input_data_float32.nbytes
                            decompressed_size = D_data.nbytes #uint8
                            decompressed_size_float32 = decomp_data.nbytes 
                            
                            df.loc[len(df)] = [
                                paths[i], 
                                bound, 
                                comp, 
                                ssim, 
                                psnr_value, 
                                (uncompressed_size/encoding_time)/1000000, #records in megabytes (1e6 bytes)
                                (uncompressed_size/de_comp_time)/1000000,  #records in megabytes (1e6 bytes)
                                uncompressed_size/compressed_size_float32, #cr
                                comp_normalize_time,
                                decomp_normalize_time,
                                comp_time,
                                comp_config_time,
                                encoding_time,
                                decoding_time,
                                de_comp_time,
                                end, #(total_time)
                                mode,
                                thread,
                                uncompressed_size, #uncompressed size of data in bytes
                                uncompressed_size_float32,
                                compressed_size_float32, #compressed size of data in bytes
                                decompressed_size, #decompressed size of data in bytes
                                decompressed_size_float32,
                                bound_name,
                                split
                            ]

                            print(name)
                            filename = os.path.basename(name).split('/')[-1]
                            save = 3
                            if save == 0:
                                dest = '/scratch/aniemcz/cavsMiniLossyCompressorsV2/'
                                
                                if i == 0:
                                    dest_2 = '/rgb/'
                                elif i == 1:
                                    dest_2 = '/id/'
                                
                                split = "Test"
                                                             
                                if(j == 0):
                                    path = dest + f'{comp}/' + '1E-7/' + split + dest_2 #technically not 1E-7 actually it should be e^-7
                                if(j == 1):
                                    path = dest + f'{comp}/' + '1E-6/' + split + dest_2
                                if(j == 2):
                                    path = dest + f'{comp}/' + '1E-5/' + split + dest_2
                                if(j == 3):
                                    path = dest + f'{comp}/' + '1E-4/' + split + dest_2
                                if(j == 4):
                                    path = dest + f'{comp}/' + '1E-3/' + split + dest_2
                                if(j == 5):
                                    path = dest + f'{comp}/' + '1E-2/' + split + dest_2
                                if(j == 6):
                                    path = dest + f'{comp}/' + '1E-1/' + split + dest_2

                                # make the result directory
                                if not os.path.exists(path):
                                    os.makedirs(path)
                                    
                                im = Image.fromarray(D_data)
                                im.save(path + str(filename))
                                
                                print(path + str(filename))
                                
                        j = j + 1
                i = i + 1
            mode_count = mode_count[0] + 1
    return df

paths = ['imgs/*.png'] #only compressing the rgb images and not the masks (we don't want to lose the accuracy of the masks)

#switch to use train and Test should change this script to do both at once maybe
srcpath = f'/scratch/aniemcz/cavsMiniLossyCompressorsV2/mixed/{split}/'

#compressors = ['zstd','blosclz','lz4','lz4hc','zlib'] #lossless compressors
compressors = ['sz3', 'zfp'] #lossy compressors

df = pd.DataFrame({
            'filename':[],
            'bound':[],
            'compressor':[],
            'ssim':[],
            'psnr':[],
            'cBW':[],
            'dBW':[],
            'CR':[],
            'comp_normalize_time':[],
            'decomp_normalize_time':[],
            'comp_time':[],
            'comp_config_time':[],
            'encoding_time':[],
            'decoding_time':[],
            'de_comp_time':[],
            'total_time':[],
            'diff':[],
            'thread':[],
            'uncompressed_size_bytes':[], #I added these bottom 3
            'uncompressed_size_float32_bytes':[],
            'compressed_size_float32_bytes':[],
            'decompressed_size_bytes':[],
            'decompressed_size_float32_bytes':[],
            'bound_name':[],
            'split':[],
})

mode = ["base"]

data = compression(srcpath, paths, compressors, mode, split)

data.to_csv(f'output/cavs_mini_lossy_v2_{split}_v3.csv')
#data.to_csv('compression/rellis_base_lossy.csv')

#!/usr/bin/env python

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
from skimage.metrics import peak_signal_noise_ratio as psnr
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

from skimage.metrics import structural_similarity

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
                for comp in compressors:
                    j = 0 #so we only pick the 1E-1 compressor for now
                    bounds = [math.exp(-7),math.exp(-6),math.exp(-5),math.exp(-4),math.exp(-3),math.exp(-2),math.exp(-1)]

                    while(j<len(bounds)):
                        l=glob.glob(srcpath+paths[i])
                        l.sort()
                        #print(l)
                        
                        #creates a numpy array containing the data of all the images combined and flattened into one dimension
                        input_data = np.concatenate([np.array(Image.open(fname)).flatten() for fname in l])
                        
                        counter = 0
                        #for name in l:
                        #print(name)
                        counter = counter + 1

                        bound = bounds[j]

                        inp = image.imread(name)
                        input_data = np.asarray(inp)

                        ori_data = input_data.copy()
                        #print(input_data.shape)

                        input_data = input_data.astype(np.float32)

                        i_data = input_data.copy()
                        D_data = input_data.copy()
                        diff_data = input_data.copy()
                        decomp_data = input_data.copy()
                        de_data = input_data.copy()

                        #start timer
                        start = time.time()

                        input_data = cv2.normalize(input_data, None, 0, 1, cv2.NORM_MINMAX)
                        #input_data = (decomp_data - decomp_data.min())/ (decomp_data.max() - decomp_data.min())

                        normalize_time = time.time() - start                
                        diff_time = time.time() - start - normalize_time

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
                                    "pressio:abs": bound,
                            }})

                        comp_data = compressor.encode(input_data)

                        comp_time = time.time() - diff_time - normalize_time - start
                        encoding_time = time.time()- start

                        decomp_data = compressor.decode(comp_data, decomp_data)
                                                
                        D_data = cv2.normalize(decomp_data, None, 255, 0, cv2.NORM_MINMAX, cv2.CV_8U)
                        #D_data = (decomp_data - decomp_data.min())/ (decomp_data.max() - decomp_data.min()) * 255

                        de_comp_time = time.time() - encoding_time - start
                        end = time.time() - start

                        #D_data = decomp_data.copy()

                        #print("The time of execution of above program is :",(end) * 10**3, "ms")
                        metrics = compressor.get_metrics()
                        D_data = D_data.astype(np.uint8)
                        (ssim, diff) = structural_similarity(i_data[:,:,0], D_data[:,:,0], data_range=255, full=True)
                        #diff = 0
                        #ssim = 0
                        psnr_value = psnr(i_data, D_data, data_range=255)

                        diff = (diff * 255).astype("uint8")

                        #print("SSIM: {}".format(ssim))
#get size from libpressio

                        #returns size in bytes
                        size = os.path.getsize(name)

                        df.loc[len(df)] = [
                            paths[i], 
                            bound, 
                            comp, 
                            ssim, 
                            psnr_value, 
                            (size/encoding_time)/1000000, #records in megabytes (1e6 bytes)
                            (size/de_comp_time)/1000000,  #records in megabytes (1e6 bytes)
                            size/metrics['size:compressed_size'],
                            normalize_time,
                            diff_time,
                            comp_time,
                            encoding_time,
                            de_comp_time,
                            end, #(total_time)
                            mode,
                            thread,
                            split
                        ]

                        print(name)
                        filename = os.path.basename(name).split('/')[-1]
                        save = 3 #3 so no save
                        dest = '/scratch/aniemcz/cavsMiniLossyCompressors/'
                        if i ==0:
                            dest_2 = '/rgb/'
                        elif i == 1:
                            dest_2 = '/id/'

                        if save == 0:
                            #dest = '/scratch/mfaykus/dissertation/datasets/rellis-images/compressed/'

                            if(j == 0):
                                path = dest + f'{comp}/' + '1E-7' + dest_2
                            if(j == 1):
                                path = dest + f'{comp}/' + '1E-6' + dest_2
                            if(j == 2):
                                path = dest + f'{comp}/' + '1E-5' + dest_2
                            if(j == 3):
                                path = dest + f'{comp}/' + '1E-4' + dest_2
                            if(j == 4):
                                path = dest + f'{comp}/' + '1E-3' + dest_2
                            if(j == 5):
                                path = dest + f'{comp}/' + '1E-2' + dest_2
                            if(j == 6):
                                path = dest + f'{comp}/' + '1E-1' + dest_2

                            # make the result directory
                            if not os.path.exists(path):
                                os.makedirs(path)

                            im = Image.fromarray(D_data)

                            print(path + str(filename))

                            im.save(path + str(filename))
                            imageio.imwrite(path + str(filename), D_data)
                        elif save == 1:

                            path = dest + f'{comp}/IE-{7-j}' + dest_2 + "compressed_output/compressed_image.bin"
                            # make the result directory
                            if not os.path.exists(path):
                                os.makedirs(path)
                            # Save the compressed data to a binary file
                            with open(path, 'wb') as f:
                                f.write(compressed_data)

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
compressors = ['zfp', 'sz3'] #lossy compressors
#compressors = ['sz3']

df = pd.DataFrame({
            'filename':[],
            'bound':[],
            'compressor':[],
            'ssim':[],
            'psnr':[],
            'cBW':[],
            'dBW':[],
            'CR':[],
            'normalize_time':[],
            'diff_time':[],
            'comp_time':[],
            'encoding_time':[],
            'de_comp_time':[],
            'total_time':[],
            'diff':[],
            'thread':[],
            'split':[],
})

mode = ["base"]

data = compression(srcpath, paths, compressors, mode)
data.to_csv(f'output/cavs_mini_lossy_binary_{split}_v3.csv')
#data.to_csv('compression/rellis_base_lossy.csv')

'''
1E-1 for right now for lossy
'''

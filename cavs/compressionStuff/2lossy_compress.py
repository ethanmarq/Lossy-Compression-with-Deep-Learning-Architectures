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
                    j = 0
                    bounds = [math.exp(-7),math.exp(-6),math.exp(-5),math.exp(-4),math.exp(-3),math.exp(-2),math.exp(-1)]

                    while(j<len(bounds)):
                        l=glob.glob(srcpath+paths[i])
                        l.sort()
                        #print(l)
                        counter = 0
                        for name in l:
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

                            diff = (diff * 255).astype("uint8")

                            #print("SSIM: {}".format(ssim))
    #get size from libpressio
    
                            size = os.path.getsize(name)

                            df.loc[len(df)] = [paths[i], bound, comp, ssim, metrics['error_stat:psnr'] , (size/encoding_time)/1000000, (size/de_comp_time)/1000000, size/metrics['size:compressed_size'],normalize_time,diff_time,comp_time,encoding_time,de_comp_time,end,mode,thread]

                            print(name)
                            filename = os.path.basename(name).split('/')[-1]
                            save = 0
                            if save == 0:
                                #dest = '/scratch/mfaykus/dissertation/datasets/rellis-images/compressed/'
                                dest = '/scratch/aniemcz/cavsMiniLossyCompressors/'
                                
                                if i ==0:
                                    dest_2 = '/id/'
                                elif i == 1:
                                    dest_2 = '/rgb/'
                                                             
                                if(j == 0):
                                    path = dest + '1E-7' + dest_2
                                if(j == 1):
                                    path = dest + '1E-6' + dest_2
                                if(j == 2):
                                    path = dest + '1E-5' + dest_2
                                if(j == 3):
                                    path = dest + '1E-4' + dest_2
                                if(j == 4):
                                    path = dest + '1E-3' + dest_2
                                if(j == 5):
                                    path = dest + '1E-2' + dest_2
                                if(j == 6):
                                    path = dest + '1E-1' + dest_2

                                # make the result directory
                                if not os.path.exists(path):
                                    os.makedirs(path)
                                    
                                im = Image.fromarray(D_data)
                                
                                print(path + str(filename))
                                
                                im.save(path + str(filename))
                                imageio.imwrite(path + str(filename), D_data)

                        j = j + 1
                i = i + 1
            mode_count = mode_count[0] + 1
    return df


#paths = ['id/*.jpg', 'rgb/*.jpg']
paths = ['masks/*.png', 'imgs/*.png'] #assuming the masks are int map

#srcpath='/scratch/mfaykus/dissertation/datasets/rellis-images/train/'
srcpath = '/scratch/aniemcz/CAT2/CAT/mixed/Train/'

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
            'thread':[]})

mode = ["base"]

data = compression(srcpath, paths, compressors, mode)
data.to_csv('output/cavs_mini_lossy.csv')
#data.to_csv('compression/rellis_base_lossy.csv')
